from pathlib import Path
import os
import requests

from src.common.jsonl import write_jsonl

REPO_NAME = "xuxueli/xxl-job"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"

# 冻结抓取的时间上界。仓库一直有新 issue 被创建关闭，之前没有这个上界时，GitHub 默认按
# created 倒序返回，每次重跑 ingest 抓到的"最新 50 条"窗口都会往后挪，旧 issue 被挤出语料——
# 2026-07-21 就是这么把 gold.jsonl 里标注过的 3893/3894/3896 挤没的，属于隐藏的可复现性 bug。
# 固定这个时间点之后，不管什么时候重跑，抓到的都是"这个时间点之前创建的那批 issue"，
# 时间点之后新产生的 issue 不会再挤占已经抓过的旧窗口。
FETCH_CUTOFF = "2026-07-21T23:59:59Z"


def fetch_closed_issues(max_issues=60, max_pages=10):
    url = f"https://api.github.com/repos/{REPO_NAME}/issues"

    headers = {
        "Accept": "application/vnd.github+json",
        "User-agent": "repo-assistant"
    }

    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    count = 1
    per_page = 100
    issues = []

    while len(issues) < max_issues and count < max_pages:
        params = {
            "state": "closed",
            "per_page": per_page,
            "page": count,
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        items = response.json()
        if not items:
            break

        for item in items:
            if "pull_request" in item:
                continue

            if item.get("created_at", "") > FETCH_CUTOFF:
                continue

            body = item.get("body") or ""

            if not body.strip():
                continue

            item["type"] = "issue"
            issues.append(item)

            if len(issues) >= max_issues:
                break

        count += 1

    return issues


def fetch_issue_comments(issue_number):
    # 参考 fetch_closed_issues() 的写法：
    # - url 换成 https://api.github.com/repos/{REPO_NAME}/issues/{issue_number}/comments
    url = f"https://api.github.com/repos/{REPO_NAME}/issues/{issue_number}/comments"
    # - headers 跟 fetch_closed_issues 里一样（Accept + User-agent + 可选的 GITHUB_TOKEN）
    #   注意：没有 token 时不能把 Authorization 设成 None——requests 会直接报
    #   InvalidHeader 崩掉，必须像 fetch_closed_issues 那样，没有 token 就干脆不设这个 key
    headers = {
        "Accept": "application/vnd.github+json",
        "User-agent": "repo-assistant"
    }

    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    # - 用 requests.get 发一次请求就够，返回的是评论列表，直接 response.json() 返回
    #   评论接口没有 state 参数（那是 issue 列表接口才有的），这里不用传
    params = {
        "per_page": 100,
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    items = response.json()
    #   （不用像 issue 列表那样分页，一个 issue 下的评论数一般远不到 100 条）
    return items


def build_chunk(raw_item):
    source_type = raw_item.get("type")
    source_id = raw_item.get("number")
    body = raw_item.get("body") or ""

    return {
        "id": f"{REPO_NAME}_{source_type}_{source_id}_chunk_0",
        "repo": REPO_NAME,
        "source_type": source_type,
        "source_id": source_id,
        "title": raw_item.get("title"),
        "url": raw_item.get("html_url"),
        "text": body.strip(),
        "metadata": {
            "state": raw_item.get("state"),
            "created_at": raw_item.get("created_at"),
            "updated_at": raw_item.get("updated_at"),
        },
    }


# 长字符床转短字符串
def split_text(text, chunk_size=1000, overlap=150):
    # 去除首尾空格
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start: end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        start = start + chunk_size - overlap

    return chunks


def build_chunks(raw_item):
    source_type = raw_item.get("type")
    source_id = raw_item.get("number")
    body = raw_item.get("body") or ""

    text_parts = split_text(body, chunk_size=1000, overlap=100)

    chunks = []
    for chunk_index, text_part in enumerate(text_parts):
        chunk = {
            "id": f"{REPO_NAME}_{source_type}_{source_id}_chunk_{chunk_index}",
            "repo": REPO_NAME,
            "source_type": source_type,
            "source_id": source_id,
            "chunk_index": chunk_index,
            "title": raw_item.get("title"),
            "url": raw_item.get("html_url"),
            "text": text_part,
            "metadata": {
                "state": raw_item.get("state"),
                "created_at": raw_item.get("created_at"),
                "updated_at": raw_item.get("updated_at"),
            }
        }

        chunks.append(chunk)

    return chunks


def build_comment_chunks(issue_number, issue_title, comment):
    # 把一条评论切成 chunk，返回的字典结构要跟 build_chunks() 对齐：
    # id / repo / source_type / source_id / chunk_index / title / url / text / metadata

    # 坑位 1：source_id 必须还是 issue_number（不是 comment 自己的 id）。
    #   gold.jsonl 里 expected_sources 标的是 issue 号，run_eval.py 判分靠 source_id 是否
    #   出现在 expected_sources 列表里；评论内容算作这个 issue 底下的证据，要跟正文共用同一个来源号。

    # 坑位 2：但是 chunk 的 "id" 字段（index/main.py 里拿去做 uuid5 生成 Qdrant point id）必须全局唯一。
    #   如果 id 只拼 issue_number（照抄 build_chunk 的写法 f"..._{source_id}_chunk_{chunk_index}"），
    #   同一个 issue 下有好几条评论时，每条评论的 chunk_index 都从 0 开始数，会拼出一模一样的 id 字符串——
    #   后写入的评论会在 Qdrant 里静默覆盖先写入的，不会报错，非常隐蔽。
    #   一定要把 comment["id"]（GitHub 给这条评论的全局唯一 id）也拼进 id 字符串里。

    # 切块逻辑直接复用 split_text()，别重新写一遍。
    # source_type 建议写死成 "issue_comment"；url 用 comment["html_url"]（精确指向这条评论，不是整个 issue）。
    body = comment.get("body") or ""
    text_parts = split_text(body, chunk_size=1000, overlap=100)

    chunks = []
    for chunk_index, text_part in enumerate(text_parts):
        chunk = {
            "id": f"{REPO_NAME}_issue_comment_{issue_number}_{comment['id']}_chunk_{chunk_index}",
            "repo": REPO_NAME,
            "source_type": "issue_comment",
            "source_id": issue_number,
            "chunk_index": chunk_index,
            "title": issue_title,
            "url": comment.get("html_url"),
            "text": text_part,
            "metadata": {
                "author": comment.get("user", {}).get("login"),
                "created_at": comment.get("created_at"),
                "updated_at": comment.get("updated_at"),
            }
        }

        chunks.append(chunk)

    return chunks


def main():
    raw_items = fetch_closed_issues()

    chunks = []

    for raw_item in raw_items:
        item_chunks = build_chunks(raw_item)
        chunks.extend(item_chunks)

        # 坑位 3：别对所有 issue 都调 fetch_issue_comments —— raw_item 里已经带了一个
        #   "comments" 字段（GitHub 返回的整数，这个 issue 下有几条评论），等于 0 的直接跳过，
        #   省掉大量没意义的 API 请求（50 个 issue 里大概率有一半以上是 0 评论）。
        # 这里接：判断 raw_item["comments"] > 0 → fetch_issue_comments → 对每条评论调
        #   build_comment_chunks，结果 extend 进 chunks。
        if raw_item.get("comments", 0) > 0:
            comments = fetch_issue_comments(raw_item["number"])
            for comment in comments:
                comment_body = comment.get("body") or ""
                if not comment_body.strip():
                    continue
                comment_chunks = build_comment_chunks(raw_item["number"], raw_item.get("title"), comment)
                print(f"{comment_chunks=}")
                chunks.extend(comment_chunks)

    write_jsonl(chunks, OUTPUT_PATH)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
