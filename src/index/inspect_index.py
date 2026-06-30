import os

from qdrant_client import QdrantClient

COLLECTION_NAME = "repo_assistant_chunks"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def create_qdrant_client():
    return QdrantClient(url=QDRANT_URL, timeout=10)


def inspect_collection(client):
    # 检查 collection 是否存在；如果不存在，Qdrant 会直接报错。
    info = client.get_collection(collection_name=COLLECTION_NAME)
    print("collection:", COLLECTION_NAME)
    print("indexed_vectors_count:", info.indexed_vectors_count)
    print("points_count:", info.points_count)


def inspect_sample_point(client):
    # scroll 类似分页查询；这里取第一条 point 看看 payload 是否写入成功。
    points, next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        print("没找到points")
        return

    point = points[0]
    payload = point.payload or {}

    print("point_id", point.id)
    print("chunk_id", payload.get("chunk_id"))
    print("title", payload.get("title"))

    text = payload.get("text") or ""
    print("text长：", len(text))


def main():
    client = create_qdrant_client()

    inspect_collection(client)
    inspect_sample_point(client)

if __name__ == "__main__":
    main()
