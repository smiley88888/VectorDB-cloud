from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
import glob
import torch
import time
import tqdm
import QdrantCloud
from qdrant_client.models import PointStruct, Distance, VectorParams
import sys
import uuid
import argparse


def vectorize_texts(texts, tokenizer, model):
    vectors = []
    i = 0
    # for text in tqdm.tqdm(texts):
    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.pooler_output
        vectors.append(embeddings[0].numpy())
        i += 1
        # if i==50: break
    return np.array(vectors)


def add_to_index(client, index_name, text_ids, texts, user_ids, sites, langs, tokenizer, model, batch_size=100, log=False):
    assert len(text_ids)==len(texts)==len(user_ids)==len(sites)==len(langs), "Unequal lengths of IDs and texts"
    vectors = vectorize_texts(texts, tokenizer, model) #texts redefined so that we know order
    if log: print(f"First 5 rows of index:\n{vectors[:5]}\n\n")
    points = []
    for vector, text_id, text, user_id, site, lang in zip(vectors, text_ids, texts, user_ids, sites, langs):
        point = PointStruct(
            vector=vector.tolist(),
            # id=str(uuid.uuid3(uuid.NAMESPACE_DNS, text)),
            id = text_id,
            payload={"text_id": text_id, "text":text, "user_id": user_id, "site": site, "lang": lang}
        )
        points.append(point)
    # print(f"Inserting {len(points)} points to VDB")

    if len(points)==1: 
        point_to_print = {'id': points[0].id, 'text_id': points[0].payload["text_id"],  \
                         'user_id': points[0].payload["user_id"], 'text': points[0].payload["text"],\
                         'site': points[0].payload["site"], 'lang': points[0].payload["lang"]
                         }
        print(f"Inserting {point_to_print}")
        client.upsert(collection_name=index_name, points=points)
        return

    #For multiple points
    points_to_insert = []
    for point in points:
        points_to_insert.append(point)
        if len(points_to_insert)>=batch_size:
            client.upsert(collection_name=index_name, points=points_to_insert)
            points_to_insert = []
    if len(points_to_insert)>0:
        client.upsert(collection_name=index_name, points=points_to_insert)


def main(text_ids, user_ids, texts, index_name, emb_size):
    existing_indexes = [index.name for index in QdrantCloud.client.get_collections().collections]
    if index_name not in existing_indexes:
        # print("Creating index", index_name)
        QdrantCloud.client.create_collection(
            collection_name=index_name,
            vectors_config=VectorParams(size=emb_size, distance=Distance.EUCLID)
        )
    else:
        # print(f"Index {index_name} already present")
        pass

    insertion_status = False
    try:
        add_to_index(QdrantCloud.client, index_name, text_ids, texts, user_ids, QdrantCloud.tokenizer, QdrantCloud.model)
        insertion_status = True
    except Exception as e:
        print(f"Failed with exception: {e}")
    return insertion_status


if __name__=="__main__":
    # Required
    parser = argparse.ArgumentParser(description="Handles insert and search")
    parser.add_argument("--text_id", type=int, required=True, help="ID associated with the text.")
    parser.add_argument("--user_id", type=int, required=True, help="User ID that will be used for filtering")
    parser.add_argument("--text", type=str, required=True, help="Text string to insert into VDB")
    # Defaults
    parser.add_argument("--index_name", type=str, default="EverGrowingVDB", help="Name of Vector DB collection/index")
    parser.add_argument("--emb_size", type=int, default=384, help="Vector dimensionality")
    args = parser.parse_args()
    insertion_status = main([args.text_id],[ args.user_id], [args.text], args.index_name, args.emb_size)
    print(f"{insertion_status}")
