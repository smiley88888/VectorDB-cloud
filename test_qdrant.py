from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
import glob
import torch
import time
import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, InitFrom
import sys
import uuid


def get_data():
    xlsx_file = "files/Query - VDB Test - ever-growing.xlsx"
    sheet1 = pd.read_excel(io=xlsx_file, sheet_name="input_1")
    sheet2 = pd.read_excel(io=xlsx_file, sheet_name="Queries_for_input_1")
    sheet3 = pd.read_excel(io=xlsx_file, sheet_name="input_2")
    sheet4 = pd.read_excel(io=xlsx_file, sheet_name="Queries_for_input_2")
    sheet5 = pd.read_excel(io=xlsx_file, sheet_name="input_3")
    sheet6 = pd.read_excel(io=xlsx_file, sheet_name="Queries_for_input_3")
    return (sheet1["id"].astype(int).tolist(), sheet1["str"].tolist(), sheet2.to_numpy()[[0,2,4,7,8,10,11,13,14,15], 0].tolist()), \
           (sheet3["id"].astype(int).tolist(), sheet3["str"].tolist(), sheet4.to_numpy()[[0,2,4,7,8,10,11,13,14,15], 0].tolist()), \
           (sheet5["id"].astype(str).tolist(), sheet5["str"].tolist(), sheet6.to_numpy()[[0,2,4,6,7,9,10,12,13,14], 0].tolist())


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


def add_to_index(client, index_name, ids, texts, tokenizer, model, batch_size=100, log=False):
    if not os.path.isfile(f"files/{index_name}.npy"):
        print("Vectorizing...")
        vectors = vectorize_texts(texts, tokenizer, model) #texts redefined so that we know order
        np.save(f"files/{index_name}.npy", vectors)
    else:
        print("Loading from NPY file")
        vectors = np.load(f"files/{index_name}.npy")
    if log: print(f"First 5 rows of index:\n{vectors[:5]}\n\n")
    points = []
    for vector, id, text in zip(vectors, ids, texts):
        point = PointStruct(vector=vector.tolist(), \
                            id=str(uuid.uuid3(uuid.NAMESPACE_DNS, text)), \
                            payload={"text_id": id, "text":text}
                           )
        points.append(point)
        if len(points)>=batch_size:
            client.upsert(collection_name=index_name, points=points)
            points = []
    if len(points)>0:
        client.upsert(collection_name=index_name, points=points)



def search_similar_texts(query_text, client, index_name, tokenizer, model, top_k=5):
    query_vector = tokenizer(query_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**query_vector)
    query_embedding = outputs.pooler_output.numpy()
    result = client.search(query_vector=query_embedding[0].tolist(), limit=top_k, collection_name=index_name)
    similar_ids, similar_texts, distances = list(),list(),list()
    for item in result:
        similar_id = item.id
        similar_ids.append(item.payload["text_id"])
        similar_texts.append(item.payload["text"])
        distances.append(item.score)
    return similar_ids, similar_texts, distances 


def main():
    index_name = "EverGrowingVDBTest"
    updated_index_name = "EverGrowingVDBTestUpdated"
    index_v3_name = "EverGrowingVDBTestV3"
    emb_size = 384

    client = QdrantClient(host="localhost", port=6333)
    existing_indexes = [index.name for index in client.get_collections().collections]
    print("Existing Indexes:", existing_indexes)

    for item in existing_indexes: client.delete_collection(item)
    existing_indexes = [index.name for index in client.get_collections().collections]
    print("Current Indexes after deletion:", existing_indexes)

    (ids, texts, queries), (new_ids, new_texts, new_queries), (ids_3, texts_3, queries_3) = get_data()
    modify_queries = lambda Q: [Q[0],Q[1],Q[2], ' '.join(Q[3:5]), ' '.join(Q[5:7]), ' '.join(Q[7:])]
    queries, new_queries, queries_3 = modify_queries(queries), modify_queries(new_queries), modify_queries(queries_3)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    if index_name not in existing_indexes:
        print(index_name, "INSERTION")
        start = time.time()
        client.create_collection(
            collection_name=index_name,
            vectors_config=VectorParams(size=emb_size, distance=Distance.EUCLID)
        )
        add_to_index(client, index_name, ids, texts, tokenizer, model, log=True)
        print(f"Index creation Time = {time.time()-start} seconds\n\n")
    else:
        print(f"Index {index_name} already present")

    for query_text in queries:
        start = time.time()
        print(f"[Query]: {query_text}")
        similar_ids, similar_texts, distances = search_similar_texts(query_text, client, index_name, tokenizer, model, top_k=5)
        print(f"[Results]:")
        for id, text, distance in zip(similar_ids, similar_texts, distances):
            print(f"\tID: {id},  Distance: {distance}, Text: {text.strip()}")
        print(f"Search Time = {time.time()-start} seconds")
        print('\n'*2)

    if updated_index_name not in existing_indexes:
        print(updated_index_name, "UPDATION")
        start = time.time()
        client.create_collection(
            collection_name=updated_index_name,
            vectors_config=VectorParams(size=emb_size, distance=Distance.EUCLID),
            init_from=InitFrom(collection=index_name)
        )
        add_to_index(client, updated_index_name, new_ids, new_texts, tokenizer, model)
        print(f"Index updation Time = {time.time()-start} seconds\n\n")
    else:
        print(f"Index {updated_index_name} already present")

    for query_text in new_queries:
        start = time.time()
        print(f"[Query]: {query_text}")
        similar_ids, similar_texts, distances = search_similar_texts(query_text, client, updated_index_name, tokenizer, model, top_k=5)
        print(f"[Results]:")
        for id, text, distance in zip(similar_ids, similar_texts, distances):
            print(f"\tID: {id},  Distance: {distance}, Text: {text.strip()}")
        print(f"Search Time = {time.time()-start} seconds")
        print('\n'*2)

    if index_v3_name not in existing_indexes:
        print(updated_index_name, "INSERTION new DB")
        start = time.time()
        client.create_collection(
            collection_name=index_v3_name,
            vectors_config=VectorParams(size=emb_size, distance=Distance.EUCLID)
        )
        add_to_index(client, index_v3_name, ids_3, texts_3, tokenizer, model)
        print(f"Index creation Time = {time.time()-start} seconds")
    else:
        print(f"Index {index_v3_name} already present")

    for query_text in queries_3:
        start = time.time()
        print(f"[Query]: {query_text}")
        similar_ids, similar_texts, distances = search_similar_texts(query_text, client, index_v3_name, tokenizer, model, top_k=5)
        print(f"[Results]:")
        for id, text, distance in zip(similar_ids, similar_texts, distances):
            print(f"\tID: {id},  Distance: {distance}, Text: {text.strip()}")
        print(f"Search Time = {time.time()-start} seconds")
        print('\n'*2)
    print("DBG",client.get_collections())


if __name__=="__main__":
    start = time.time()
    main()
    print(f"Total Time in running script = {time.time()-start} seconds")
    print("Vector DB used: Qdrant")
    print("Methodology: Vectorize using `sentence-transformers/all-MiniLM-L6-v2` model then use L2 distance to search closest vectors in DB")
