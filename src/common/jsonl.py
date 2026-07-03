import json


def read_jsonl(path):
    # 兼容两种格式：一行一条的标准 jsonl，以及手工美化后每个对象跨多行缩进的格式
    # （gold.jsonl / chunks.jsonl 有时会被手动格式化方便查看）。
    items = []
    decoder = json.JSONDecoder()

    text = path.read_text(encoding="utf-8")
    pos = 0
    length = len(text)

    while pos < length:
        while pos < length and text[pos].isspace():
            pos += 1

        if pos >= length:
            break

        item, end = decoder.raw_decode(text, pos)
        items.append(item)
        pos = end

    return items



def write_jsonl(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in items:
            line = json.dumps(item, ensure_ascii=False)
            file.write(line + "\n")
