from config import config
import logging

from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue



logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %I:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)



qdrant_url=config["QDRANT_URL"]
qdrant_api_key=config["QDRANT_API_KEY"]

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
index_name = "EverGrowingVDB"




