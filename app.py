from fastapi import HTTPException, Query
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import insert
import search
import re
import QdrantCloud
from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue
import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %I:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


app = FastAPI()


# Root route
@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	print(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.get("/insert")
async def insert_data(
        id: int = Query(..., title="ID"), 
        user_id: int = Query(..., title="User ID"), 
        text: str = Query(..., title="Text"),
        site: str = Query(..., title="Site"),
        lang: str = Query(..., title="Lang"),
    ):
    logger.info(f"Received data:, \"id\": {id}, \"user_id\": {user_id}, \"text\": {text}")

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


@app.get("/search")
async def search_data(
        user_id: int = Query(..., title="User ID"), 
        text: str = Query(..., title="Text"), 
        limit: int = Query(..., title="Limit"),
        site: str = Query(None, title="Site"),
        lang: str = Query(None, title="Lang"),
    ):
    logger.info(f"Received data:, \"user_id\": {user_id}, \"text\": {text}, \"limit\": {limit}, \"site\": {site}, \"lang\": {lang}")

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
    logger.info(f"remove_by_user:, \"user_id\": {user_id}")

    try:
        QdrantCloud.client.delete(collection_name=QdrantCloud.index_name,points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="user_id",match=MatchValue(value=user_id))]))
        )
        return 1
    except:
        pass
    return 0


@app.get("/remove_all_by_word")
async def remove_all_by_word(
        user_id: int = Query(..., title="User ID"), 
        word: str = Query(..., title="Word"), 
    ):
    logger.info(f"remove_all_by_word:, \"user_id\": {user_id}, \"word\": {word}")

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


@app.get("/remove_all_by_regex")
async def remove_all_by_regex(
        user_id: int = Query(..., title="User ID"), 
        regex: str = Query(..., title="Regular Expression"), 
    ):
    logger.info(f"remove_all_by_regex:, \"user_id\": {user_id}, \"regex\": {regex}")

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
    logger.info(f"remove_all_by_regex:, \"user_id\": {user_id}, \"cats\": {cats}, \"title\": {title}")

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



if __name__ == "__main__":
    import uvicorn
    logger.info("----- start Ever-Growing Database service -----")
    host = "0.0.0.0"
    port = 8000
    uvicorn.run(app, host=host, port=port, reload=True)
    
