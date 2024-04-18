import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from llama_index.core import Settings

from configs.config import AppConfig
from elastic import get_async_client, create_query_engine, get_llm, initialize_index

config = None
index = None


@asynccontextmanager
async def lifespan(app):
    # On app startup
    global index
    global config
    config = AppConfig.from_yaml(os.getenv("CONFIG_FILE_PATH"))
    Settings.embed_model = None
    Settings.llm = get_llm(config.query.llm_path, env_path=".env")
    async_es_client = await get_async_client()
    index = await initialize_index(config.ingest, async_es_client)
    yield
    # On app shutdown
    await async_es_client.close()


app = FastAPI(lifespan=lifespan)


@app.get("/query")
async def send_query(q: str = None) -> str:
    global index
    global config
    query_engine = create_query_engine(config.query, index)
    response = query_engine.query(q)
    return str(response)
