from fastapi import FastAPI, HTTPException, Query
import insert
import search
import re
import QdrantCloud
from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue


app = FastAPI()


@app.get("/insert.py")
async def insert_data(
        id: int = Query(..., title="ID"), 
        user_id: int = Query(..., title="User ID"), 
        text: str = Query(..., title="Text"),
        site: str = Query(..., title="Site"),
        lang: str = Query(..., title="Lang"),
    ):
    print("Received data:", {"id": id, "user_id": user_id, "text": text})
    try:
        insert.add_to_index(
            client=QdrantCloud.client, index_name=QdrantCloud.index_name, text_ids=[id], 
            texts=[text], user_ids=[user_id], sites=[site], langs=[lang], 
            tokenizer=QdrantCloud.tokenizer, model=QdrantCloud.model
        )
        return 1
    except:
        pass
    return 0


@app.get("/search.py")
async def search_data(
        user_id: int = Query(..., title="User ID"), 
        text: str = Query(..., title="Text"), 
        limit: int = Query(..., title="Limit"),
        site: str = Query(None, title="Site"),
        lang: str = Query(None, title="Lang"),
    ):
    print("Received data:", {"user_id": user_id, "text": text, "limit": limit, "site": site, "lang": lang})
    _, _, _, result = search.search_similar_texts(
        query_text=text, user_id=user_id, site=site, lang=lang,
        client=QdrantCloud.client, index_name=QdrantCloud.index_name, tokenizer=QdrantCloud.tokenizer, 
        model=QdrantCloud.model, top_k=limit
    )
    result = [{"id":item.payload["text_id"], "string": item.payload["text"], \
               "score": item.score} for item in result]
    return result



@app.get("/remove_by_user")
async def remove_by_user(
        user_id: int = Query(..., title="User ID"), 
    ):
    try:
        QdrantCloud.client.delete(collection_name=QdrantCloud.index_name,points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="user_id",match=MatchValue(value=user_id))]))
        )
        return 1
    except:
        pass
    return 0


@app.get("/remove_all_by_word.py")
async def remove_all_by_word(
        user_id: int = Query(..., title="User ID"), 
        word: str = Query(..., title="Word"), 
    ):
    try:
        all_points, = QdrantCloud.client.scroll(collection_name=QdrantCloud.index_name, scroll_filter=Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
            limit=100000,
            with_payload=True,
            with_vectors=False
        )
        deletable_ids = list()
        for point in all_points:
            if word in point.payload["text"]: deletable_ids.append(point.id)
        QdrantCloud.client.delete(collection_name=QdrantCloud.index_name,
            points_selector=QdrantCloud.models.PointIdsList(deletable_ids)
        )
        return 1
    except:
        pass
    return 0


@app.get("/remove_all_by_regex.py")
async def remove_all_by_regex(
        user_id: int = Query(..., title="User ID"), 
        regex: str = Query(..., title="Regular Expression"), 
    ):
    try:
        all_points,_ = QdrantCloud.client.scroll(collection_name=QdrantCloud.index_name, scroll_filter=Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
            limit=100000,
            with_payload=True,
            with_vectors=False
        )
        deletable_ids = list()
        for point in all_points:
            if re.search(regex, point.payload["text"]) is not None: deletable_ids.append(point.id)
        QdrantCloud.client.delete(collection_name=QdrantCloud.index_name,
            points_selector=QdrantCloud.models.PointIdsList(deletable_ids)
        )
        return 1
    except:
        pass
    return 0


@app.get("/get_category_for_title")
async def get_category_for_title(
        user_id: int = Query(..., title="User ID"),
        cats: str = Query(..., title="Categories"), 
        title: str = Query(..., title="Title"),
    ):
    import torch, numpy as np
    def get_embeddings(text):
        inputs = QdrantCloud.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = QdrantCloud.model(**inputs)
        embeddings = outputs.pooler_output
        return embeddings[0].numpy()

    cats = cats.split('\\n')
    print(cats)
    print(title)
    vectors = list()
    for cat in cats: vectors.append(get_embeddings(cat))
    vectors = np.array(vectors)
    emb_title = get_embeddings(title)
    return cats[np.argmin(np.linalg.norm(emb_title-vectors, axis=1))]


