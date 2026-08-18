import jieba
from pathlib import Path

from src.common.jsonl import read_jsonl
from rank_bm25 import BM25Okapi

# 表示往上找两层
PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = PROJECT_ROOT / "eval/gold.jsonl"

CHUNKS_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"


# 需要知道一个词出现几次
def tokenize(text):
    # 没有就转换为空串
    if not text:
        return []
    # 用jieba把词进行拆分
    # cut = jieba.cut(text, cut_all=False)
    # print(f"类型是：{type(cut)}")
    # 这里的cut是一个迭代器类型

    # 这里的lcut就是一个列表类型，直接返回就可以了
    lcut = jieba.lcut(text, cut_all=False)
    # print(f"类型是：{type(lcut)}")
    return lcut


def build_bm25_corpus(chunks):
    if not chunks:
        return []
    corpus = []
    for chunk in chunks:
        text = chunk.get("text") or ""
        tokens = tokenize(text)
        corpus.append(tokens)
    return corpus


def build_bm25_index(chunks):
    corpus = build_bm25_corpus(chunks)
    bm25 = BM25Okapi(corpus)
    return bm25


def search_bm25(bm25_index, chunks, query, top_k=5):
    # 1. 分词
    query_tokens = tokenize(query)
    # 2. 打分
    scores = bm25_index.get_scores(query_tokens)
    # 3. 配对
    zip_chunks = zip(chunks, scores)
    # 4. 排序
    sorted_chunks = sorted(zip_chunks, key=lambda pair: pair[1], reverse=True)
    # 5. 取前 top_k 个
    top_pairs = sorted_chunks[:top_k]
    # 6. 返回结果,去掉分数，只返回 chunk
    top_chunks = [pair[0] for pair in top_pairs]
    return top_chunks


def main():
    # t = "我来到北京清华大学"
    # cut = tokenize(t)
    # print(" | ".join(cut))
    # 先拿一个进行测试
    jsonl = read_jsonl(JSON_PATH)[0]
    # print(jsonl)
    question = jsonl.get("question")
    answer = jsonl.get("answer_points")
    # print(f"问题是：{question}")
    # print(f"答案是：{answer}")
    # print(f"{type(question)}") # str
    # print(f"{type(answer)}") # list

    str_answer = "\n".join(answer) if isinstance(answer, list) else str(answer)
    # print(f"{type(str_answer)}")
    l_question = tokenize(question)
    l_answer = tokenize(str_answer)
    # print(f"问题的分词结果是：{' | '.join(l_question)}")
    # print(f"答案的分词结果是：{' | '.join(l_answer)}")

    chunks_jsonl = read_jsonl(CHUNKS_PATH)
    # print(f"chunks_jsonl的长度是：{len(chunks_jsonl)}")
    # corpus = build_bm25_corpus(chunks_jsonl)
    # print(f"语料库的长度是：{len(corpus)}")
    # print(f"语料库的前5条是：{corpus[:5]}")
    index = build_bm25_index(chunks_jsonl)
    # scores = index.get_scores(l_question)
    # print(scores)
    # print(type(index))
    # print(index)

    query = "内存泄漏"
    query_tokens = tokenize(query)
    scores = index.get_scores(query_tokens)

    print(scores)
    print(len(scores))
    print(len(chunks_jsonl))  # 对比一下，看看是不是跟 chunks 数量一样


if __name__ == '__main__':
    main()
