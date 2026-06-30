import json


def read_jsonl(path):
    items = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)
            items.append(item)

    return items



def write_jsonl(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in items:
            line = json.dumps(item, ensure_ascii=False)
            file.write(line + "\n")
