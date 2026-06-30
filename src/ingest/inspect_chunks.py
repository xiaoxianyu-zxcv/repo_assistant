from pathlib import Path

from src.common.jsonl import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"


def inspect_chunks(chunks):
    ids = set()
    duplicate_ids = 0
    empty_text = 0
    missing_chunk_index = 0
    source_counts = {}

    for chunk in chunks:
        chunk_id = chunk.get("id")

        if chunk_id in ids:
            duplicate_ids += 1
        else:
            ids.add(chunk_id)

        text = chunk.get("text") or ""
        if not text.strip():
            empty_text += 1

        if "chunk_index" not in chunk:
            missing_chunk_index += 1

        source_id = chunk.get("source_id")
        if source_id not in source_counts:
            source_counts[source_id] = 0

        source_counts[source_id] += 1

    unique_sources = len(source_counts)

    if source_counts:
        max_chunks_per_source = max(source_counts.values())
    else:
        max_chunks_per_source = 0

    return {
        "chunks": len(chunks),
        "empty_text": empty_text,
        "duplicate_ids": duplicate_ids,
        "missing_chunk_index": missing_chunk_index,
        "unique_sources": unique_sources,
        "max_chunks_per_source": max_chunks_per_source,
    }


def main():
    chunks = read_jsonl(CHUNK_PATH)
    result = inspect_chunks(chunks)

    print("chunks:", result["chunks"])
    print("empty_text:", result["empty_text"])
    print("duplicate_ids:", result["duplicate_ids"])
    print("missing_chunk_index:", result["missing_chunk_index"])
    print("unique_sources:", result["unique_sources"])
    print("max_chunks_per_source:", result["max_chunks_per_source"])


if __name__ == "__main__":
    main()
