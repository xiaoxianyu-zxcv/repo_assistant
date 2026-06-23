import os

from fastapi import FastAPI
from qdrant_client import QdrantClient


app = FastAPI(title="Repo Assistant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "repo-assistant"
    }


@app.get("/health/full")
def health_full():
    client = QdrantClient(url=QDRANT_URL, timeout=3)
    collections = client.get_collections()

    return {
        "status": "ok",
        "service": "repo-assistant",
        "qdrant": "ok",
        "collections_count": len(collections.collections)
    }