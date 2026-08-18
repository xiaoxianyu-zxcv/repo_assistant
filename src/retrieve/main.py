import os
import jieba


from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pathlib import Path
from src.common.jsonl import read_jsonl
from rank_bm25 import BM25Okapi
from collections import defaultdict

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = "repo_assistant_chunks"

MODEL_NAME = "BAAI/bge-m3"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = PROJECT_ROOT / "eval/gold.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"

def create_qdrant_client():
    if not hasattr(create_qdrant_client, "_cache"):
        create_qdrant_client._cache = QdrantClient(url=QDRANT_URL, timeout=10)

    return create_qdrant_client._cache


def load_embedding_model():
    if not hasattr(load_embedding_model, "_cache"):
        load_embedding_model._cache = SentenceTransformer(MODEL_NAME)

    return load_embedding_model._cache


def load_bm25_index():
    if not hasattr(load_bm25_index, "_cache"):
        jsonl = read_jsonl(CHUNKS_PATH)
        index = build_bm25_index(jsonl)
        load_bm25_index._cache = (index, jsonl)

    return load_bm25_index._cache

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

def format_bm25_result(chunk, score):
    print(f"{chunk.get('id')=}")
    return {
        "bm25_score": score,
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

# 将字符串改为列表，列表中每个元素是一个词
def tokenize(text):
    if not text:
        return []
    # 采取默认分词，直接返回列表
    return jieba.lcut(text, cut_all=False)

# 将 chunks 中的每个 chunk 的 text 字段进行分词，返回一个二维列表，每个元素是一个 chunk 的分词结果
def build_bm25_corpus(chunks):
    if not chunks:
        return []
    return [tokenize(chunk.get("text") or "") for chunk in chunks]

# 加入 BM25 索引构建函数，输入 chunks，返回 BM25 索引对象
def build_bm25_index(chunks):
    return BM25Okapi(build_bm25_corpus(chunks))

# 使用 BM25 索引进行检索，输入 BM25 索引对象、chunks、query，返回 top_k 个最相关的 chunks
def search_bm25(query, top_k=5):
    # 1. 分词
    query_list = tokenize(query)
    # 2. 加载 BM25 索引和 chunks
    index = load_bm25_index()
    bm25, chunks = index
    # 3. 打分
    scores = bm25.get_scores(query_list)
    # 4. 配对
    zip_chunks = zip(chunks, scores)
    #5. 排序
    sorted_chunks = sorted(zip_chunks, key=lambda pair: pair[1], reverse=True)
    # 6. 取前 top_k 个
    top_pairs = sorted_chunks[:top_k]
    return [format_bm25_result(chunk, score) for chunk, score in top_pairs]

# RRF（Reciprocal Rank Fusion，倒数排名融合）：把 dense 检索和 BM25 检索这两份榜单融合成一份。
# 为什么不能直接拿分数相加：dense 的余弦相似度和 BM25 的分数量纲完全不同，没有可比性。
# RRF 的做法是不看原始分数，只看每个 chunk 在各自榜单里排第几名——名次越靠前，
# 贡献的融合分数越高（1 / (k + 名次 + 1)）；如果一个 chunk 在两份榜单里都排得靠前，
# 两边的分数会累加，最终排名也会更靠前。k 是平滑常数，越大则名次差异对分数的影响越小。
def rrf_fuse(dense_results, bm25_results, k=60, top_k=5):
    # 1. 建一个字典，存每个 chunk_id 累加的融合分数
    #    （用 defaultdict(float)，这样后面可以直接 += 不用先判断 key 存不存在）

    # 2. 建一个普通字典，存 chunk_id -> 完整 chunk 数据的对应关系
    #    后面排完序只有 chunk_id，要靠这个字典查回 text/title/url 这些内容

    # 3. 遍历 dense_results，同时要拿到"第几名"和"这一条的内容"
    #    （哪个内置函数能一次给你下标和元素？）
    #    对每一条：
    #      a. 取出它的 chunk_id
    #      b. 按 RRF 公式算这个名次的分数：1 / (k + 名次 + 1)
    #      c. 把这个分数累加进第 1 步的字典（不是覆盖，是加）
    #      d. 把这个 chunk_id 对应的完整数据存进第 2 步的字典

    # 4. 对 bm25_results 重复第 3 步一模一样的逻辑
    #    （两段代码结构会长得几乎一样，只是遍历的列表换了）

    # 5. 把第 1 步的分数字典，按分数从高到低排序，只留前 top_k 个
    #    （字典的 .items() 能同时拿到 chunk_id 和分数；sorted() 的 key 参数
    #      决定按哪个排；reverse 控制升序还是降序）

    # 6. 建一个空列表存最终结果
    #    遍历第 5 步排好序的 (chunk_id, 分数) 这些配对：
    #      a. 用第 2 步的字典，把 chunk_id 换回完整 chunk 数据
    #         （建议 copy 一份，别直接改原始字典）
    #      b. 把这一步算出的融合分数存进这份复制出来的数据里
    #      c. 加进第 6 步建的列表

    # 7. 返回这个列表
    pass


def main():
    # query = "3.4.0 版本里，如果通过 initLogPath 自定义了日志根路径，callbackLogPath 会不会跟着变成新路径？"
    #
    # results = retrieve(query, top_k=5)
    # print_results(query, results)
    # query = "内存泄漏"
    # res = search_bm25(query, top_k=5)
    # for chunk in res:
    #     print(chunk.get("title"), "|", chunk.get("text")[:100].replace("\n", " "), "|", chunk.get("url"))

        # 先用模拟数据
    dense_test = [{"chunk_id": "苹果"}, {"chunk_id": "香蕉"}, {"chunk_id": "橙子"}]
    bm25_test = [{"chunk_id": "橙子"}, {"chunk_id": "西瓜"}, {"chunk_id": "苹果"}]

    res = rrf_fuse(dense_test, bm25_test, k=1)
    print(f"{res=}")




if __name__ == "__main__":
    main()
