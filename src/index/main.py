import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams, SparseIndexParams
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings

from src.common.jsonl import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = "repo_assistant_chunks"

MODEL_NAME = "BAAI/bge-m3"
VECTOR_SIZE = 1024


def create_qdrant_client():
    # 连接本机docker的qdrant
    return QdrantClient(url=QDRANT_URL, timeout=10)

def load_embedding_model():
    #加载文本向量模型
    return SentenceTransformer(MODEL_NAME)


def recreate_collection(client):
    #类似数据库表，每次重建，避免旧数据干扰
    collections = client.get_collections().collections
    exists = False

    for collection in collections:
        if collection.name == COLLECTION_NAME:
            exists = True
            break

    if exists:
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams()
            )
        }
    )

def build_payload(chunk):
    # payload 保存普通业务字段，后面回答问题需要用它展示来源
    return {
        "chunk_id": chunk.get("id"),
        "repo": chunk.get("repo"),
        "source_type": chunk.get("source_type"),
        "source_id": chunk.get("source_id"),
        "chunk_index": chunk.get("chunk_index"),
        "title": chunk.get("title"),
        "url": chunk.get("url"),
        "text": chunk.get("text"),
        "metadata": chunk.get("metadata"),
    }

def build_dummy_vector(chunk):
    # qdrant 每条数据都必须有vector，先用假的占位
    return [0.1] * VECTOR_SIZE

def build_point(chunk, vector):
    # qdrant 的point id需要int或者uuid，把chunk——id转为稳定的uuid
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"]))

    # 类似数据库的row ,展示内容
    payload = {
        "chunk_id": chunk.get("id"),
        "repo": chunk.get("repo"),
        "source_type": chunk.get("source_type"),
        "source_id": chunk.get("source_id"),
        "chunk_index": chunk.get("chunk_index"),
        "title": chunk.get("title"),
        "url": chunk.get("url"),
        "text": chunk.get("text"),
        "metadata": chunk.get("metadata"),
    }

    return PointStruct(
        id=point_id,
        vector=vector,
        payload=build_payload(chunk),
    )

def build_points(chunks, model):
    # 从每个chunk中取出正文，批量生成真实的embedding。
    texts = []

    for chunk in chunks:
        text = chunk.get("text") or ""
        texts.append(text)

    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    points = []

    for chunk, vector in zip(chunks, vectors):
        point = build_point(chunk, vector)
        points.append(point)

    return points

def upsert_chunks(client, chunks, model):
    points = build_points(chunks, model)

    #upset 有则更新，无则插入。
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )



def main():


    chunks = read_jsonl(CHUNK_PATH)
    print(f"loaded chunks: {len(chunks)}")

    model = load_embedding_model()
    print(f"loaded embedding model: {MODEL_NAME}")

    client = create_qdrant_client()

    recreate_collection(client)
    print(f"created collection: {COLLECTION_NAME}")

    upsert_chunks(client, chunks, model)
    print(f"upserted points: {len(chunks)}")


if __name__ == "__main__":
    main()