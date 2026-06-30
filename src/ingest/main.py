from pathlib import Path
import os
import requests

from src.common.jsonl import write_jsonl

REPO_NAME = "fastapi/fastapi"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"


def fetch_closed_issues(max_issues=50, max_pages=10):
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

            body = item.get("body") or ""

            if not body.strip():
                continue

            item["type"] = "issue"
            issues.append(item)

            if len(issues) >= max_issues:
                break

        count += 1

    return issues


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


def main():
    raw_items = fetch_closed_issues()

    chunks = []

    for raw_item in raw_items:
        item_chunks = build_chunks(raw_item)
        chunks.extend(item_chunks)

    write_jsonl(chunks, OUTPUT_PATH)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
