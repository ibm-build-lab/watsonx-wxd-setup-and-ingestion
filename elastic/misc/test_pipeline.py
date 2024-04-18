import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

from config import *
from elastic.utils import get_client

client = get_client()

with open("elastic/elastic_configs/test_pipeline.json") as f:
    pipeline = json.load(f)
    docs = [
        {"_source": {"text": "55.3.244.1 GET /index.html 15824 0.043 test@elastic.co"}}
    ]
    response = client.ingest.simulate(body={"pipeline": pipeline, "docs": docs})
    print(response)
