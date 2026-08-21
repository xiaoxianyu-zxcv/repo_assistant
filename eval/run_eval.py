import os
import time

from src.common.jsonl import read_jsonl
from pathlib import Path
from src.generate.main import answer_query
from src.retrieve.main import retrieve

GOLD_PATH = Path(__file__).parent / "gold.jsonl"
# 新的有难度的语料集
# GOLD_PATH = Path(__file__).parent / "gold_hard.jsonl"
TOP_K = 5


def unique_source_ids(chunks):
    return 0


def evaluate_one(gold_item, top_k):
    return 0


def is_evidence_hit(text, expected_evidence):
    # 判断这一段里有没有出现答案原文。
    # v1（gold.jsonl）没有 expected_evidence 字段，走的是 issue 级口径：
    # 只要 source_id 对上就算命中，不再往下判段落，所以这里直接放行。
    # 注意用 not 而不是 is None——None（没这个字段）和 []（空列表）都表示"没有引文"，
    # 两种都该放行。这跟 rank 那里必须用 is not None 不一样：那里的 0 是有意义的值。
    if not expected_evidence:
        return True

    # v2（gold_hard.jsonl）是引文级口径：任意一条引文出现在这段里就算命中。
    return any(ev in text for ev in expected_evidence)


def main():
    time1 = time.time()

    # 先读取文件,这是标准答案。
    jsonl = read_jsonl(GOLD_PATH)

    # 防呆：把这次跑的是哪份题、用的哪种判定口径打出来。
    # 两种口径的数字不能横向比高低（衡量的不是同一件事），所以每次都要看清楚。
    mode = "引文级(严格)" if any(item.get("expected_evidence") for item in jsonl) else "issue级(宽松)"
    print(f"题目文件：{GOLD_PATH.name}  共 {len(jsonl)} 条  判定口径：{mode}\n")
    # 是这样的dict：{question,expected_sources,answer_points}
    # print(jsonl)

    answer_map = {}
    count = 0.0
    rrm = 0.0
    for item in jsonl:
        # 现在需要三个内容，直接循环jsonl
        question = item["question"]
        expected_sources = item["expected_sources"]
        # 用 .get() 不用 []：v1 没有这个字段，[] 会直接 KeyError，.get() 取不到返回 None。
        # 千万不要退回用 answer_points——那是人写的答案概括，不是语料原文，逐字匹配几乎全不中。
        expected_evidence = item.get("expected_evidence")

        list_res = retrieve(question, top_k=TOP_K)
        # print(f"问题：{expected_evidence}")
        # list_source_ids = [source.get("source_id") for source in list_res]
        # print(f"模型返回的sources：{list_source_ids}")

        rank = None
        for i, chunk in enumerate(list_res):
            source_id = chunk.get("source_id")
            text = chunk.get("text")

            if is_evidence_hit(text, expected_evidence) and (source_id in expected_sources):
                rank = i
                break

        # 尝试一行写出来
        # expected_evidence_str = ",".join(expected_evidence)
        # rank =  next((i for i,chunk in enumerate(list_source_text) if any(expected_evidence_str in chunk)), None)

        # rank = next((i for i, sid in enumerate(list_source_ids) if sid in source_id), None)
        # 换成复杂的问题，需要看看是不是对应的切片，而不是直接看source_id

        if rank is not None:
            count += 1
            print(f"这个问题的答案是正确的：{question}")
            rrm += 1.0 / (rank + 1)
        else:
            print(f"这个问题的答案是错误的：{question}")
            rrm += 0.0

    print(f"正确率是：{count / len(jsonl)}")
    print(f"MRR是：{rrm / len(jsonl)}")
    time2 = time.time()
    print(f"开始运行时间：{time1}")
    print(f"结束运行时间：{time2}")
    print(f"总共耗时：{time2 - time1}秒")


if __name__ == "__main__":
    main()
