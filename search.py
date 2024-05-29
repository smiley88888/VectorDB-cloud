from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
import glob
import torch
import time
import tqdm
from qdrant_client.models import Filter, FieldCondition, MatchValue
import sys
import uuid
import argparse
import json
import QdrantCloud


def vectorize_texts(texts, tokenizer, model):
    vectors = []
    i = 0
    for text in tqdm.tqdm(texts):
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.pooler_output
        vectors.append(embeddings[0].numpy())
        i += 1
        # if i==50: break
    return np.array(vectors)


def search_similar_texts(query_text, user_id, site, lang, client, index_name, tokenizer, model, top_k):
    print("[SEARCH ARGS]:", query_text, user_id, site, lang, client, index_name, top_k, flush=True)
    query_vector = tokenizer(query_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**query_vector)
    query_embedding = outputs.pooler_output.numpy()
    field_conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if site is not None:
        field_conditions.append(FieldCondition(key="site", match=MatchValue(value=site)))
    if lang is not None:
        field_conditions.append(FieldCondition(key="lang", match=MatchValue(value=lang)))
    print("Field Conditions:", field_conditions)
    result = client.search(
        query_vector=query_embedding[0].tolist(), 
        query_filter=Filter(must=field_conditions),
        limit=top_k,
        collection_name=index_name
    )
    similar_ids, similar_texts, distances = list(),list(),list()
    for item in result:
        similar_uuid = item.id #not printed
        similar_ids.append(item.payload["text_id"])
        similar_texts.append(item.payload["text"])
        distances.append(item.score)
    return similar_ids, similar_texts, distances , result


def main(user_id, text, limit, index_name, emb_size):
    existing_indexes = [index.name for index in QdrantCloud.client.get_collections().collections]
    if index_name not in existing_indexes:
        print(f"Requested VDB {index_name} is not present")
        return

    start = time.time()
    # print(f"[Query]: {text}")
    similar_ids, similar_texts, distances, result = search_similar_texts(text, user_id, QdrantCloud.client, index_name, QdrantCloud.tokenizer, QdrantCloud.model, top_k=limit)
    # print(f"[Results]:")
    # for id, text, distance in zip(similar_ids, similar_texts, distances):
    #     print(f"\tID: {id},  Distance: {distance}, Text: {text.strip()}")
    # print(f"Search Time = {time.time()-start} seconds")
    result = [{"id":item.payload["text_id"], "string": item.payload["text"], "score": item.score} for item in result]
    return result


if __name__=="__main__":
    # Required
    parser = argparse.ArgumentParser(description="Handles insert and search")
    parser.add_argument("--user_id", type=int, required=True, help="User ID that will be used for filtering")
    parser.add_argument("--text", type=str, required=True, help="Text string to insert into VDB")
    parser.add_argument("--limit", type=int, required=True, help="Number of results")
    # Defaults
    parser.add_argument("--index_name", type=str, default="EverGrowingVDB", help="Name of Vector DB collection/index")
    parser.add_argument("--emb_size", type=int, default=384, help="Vector dimensionality")
    args = parser.parse_args()
    result = main(args.user_id, args.text, args.limit, args.index_name, args.emb_size)
    print(result)