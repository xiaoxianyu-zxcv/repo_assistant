import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = "repo_assistant_chunks"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def create_qdrant_client():
    return QdrantClient(url=QDRANT_URL, timeout=10)


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def embed_query(model, query):
    # 把用户问题转换成向量；这样才能和库里的 chunk 向量做相似度比较。
    vector = model.encode(query, normalize_embeddings=True)
    return vector.tolist()


def search_chunks(client, query_vector, top_k=5):
    # 在 Qdrant 中查找和 query_vector 最相近的 top_k 个 chunk。
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    return result.points


def format_point(point):
    # 把 Qdrant 返回的对象转换成普通 dict，方便后面给生成模块或 API 使用。
    payload = point.payload or {}

    return {
        "score": point.score,
        "chunk_id": payload.get("chunk_id"),
        "repo": payload.get("repo"),
        "source_type": payload.get("source_type"),
        "source_id": payload.get("source_id"),
        "chunk_index": payload.get("chunk_index"),
        "title": payload.get("title"),
        "url": payload.get("url"),
        "text": payload.get("text"),
        "metadata": payload.get("metadata"),
    }


def retrieve(query, top_k=5):
    # 对外暴露的检索函数：输入问题，返回最相关的 chunks。
    model = load_embedding_model()
    query_vector = embed_query(model, query)

    client = create_qdrant_client()
    points = search_chunks(client, query_vector, top_k=top_k)

    results = []
    for point in points:
        results.append(format_point(point))

    return results


def print_results(query, results):
    print(f"{query=}")

    for index, result in enumerate(results, start=1):
        text = result.get("text") or ""

        print(f"rank {index}")
        print("score:", result.get("score"))
        print("title:", result.get("title"))
        print("url:", result.get("url"))
        print("chunk_id:", result.get("chunk_id"))
        print("text preview:", text[:500].replace("\n", " "))
        print()


def main():
    query = "How does FastAPI handle OpenAPI schema generation?"

    results = retrieve(query, top_k=5)
    print_results(query, results)

if __name__ == "__main__":
    main()
