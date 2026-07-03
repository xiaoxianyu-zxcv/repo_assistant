import os
import time

from src.common.jsonl import read_jsonl
from pathlib import Path
from src.generate.main import answer_query
from src.retrieve.main import retrieve


GOLD_PATH = Path(__file__).parent / "gold.jsonl"
TOP_K = 5


def unique_source_ids(chunks):
    return 0

def evaluate_one(gold_item, top_k):
    return 0

def main():
    time1 = time.time()

    # 先读取文件,这是标准答案。
    jsonl = read_jsonl(GOLD_PATH)
    # 是这样的dict：{question,expected_sources,answer_points}
    # print(jsonl)
    # 逐一去问问模型
    question_source_map = {json["question"]: json["expected_sources"] for json in jsonl}

    answer_map = {}
    count = 0.0
    rrm = 0.0
    for question, source_id in question_source_map.items():
        list_res = retrieve(question, top_k=TOP_K)
        print(f"问题：{source_id}")
        # print(f"模型返回的sources：{list_res.get('sources')}")
        # print(f"模型的输出：{list_res.get("sources")}")
        list_source_ids = [source.get("source_id") for source in list_res]
        print(f"模型返回的sources：{list_source_ids}")
        # if any(item in list_source_ids for item in source_id):
        #     count += 1
        #     print(f"这个问题的答案是正确的：{question}")
        rank = next((i for i, sid in enumerate(list_source_ids) if sid in source_id), None)
        if rank is not None:
            count += 1
            print(f"这个问题的答案是正确的：{question}")
            rrm += 1.0 / (rank + 1)
        else:
            print(f"这个问题的答案是错误的：{question}")
            rrm += 0.0

    print(f"正确率是：{count/len(question_source_map)}")
    print(f"MRR是：{rrm/len(question_source_map)}")
    # print(123)
    # print(question_source_map)
    time2 = time.time()
    print(f"开始运行时间：{time1}")
    print(f"结束运行时间：{time2}")
    print(f"总共耗时：{time2-time1}秒")

if __name__ == "__main__":
    main()

