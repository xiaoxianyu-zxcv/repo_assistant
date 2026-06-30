from src.retrieve.main import retrieve
from src.common.llm import call_llm

def build_context_block(contexts):
    # 把检索到的 chunks 整理成给 LLM 阅读的上下文材料。
    blocks = []

    for index, context in enumerate(contexts, start=1):
        title = context.get("title") or ""
        url = context.get("url") or ""
        text = context.get("text") or ""

        block = f"""
        Title: {title}
        URL: {url}
        Context: {text}
        """
        blocks.append(block)

    return "\n".join(blocks)

def build_prompt(query, contexts):
    # 这是 RAG 的关键：要求模型只能基于检索上下文回答，不能自己编。
    context_block = build_context_block(contexts)

    prompt = f"""You are a repository assistant.

    Answer the user's question using only the context below.
    If the context does not contain enough information, say:
    "I don't know based on the provided repository context."

    When possible, cite the source number like [Source 1].

    Question:
    {query}

    Context:
    {context_block}

    Answer:
    """

    return prompt

def build_sources(contexts):
    sources = []

    for index, context in enumerate(contexts, start=1):
        source = {
            "source_number": index,
            "score": context.get("score"),
            "chunk_id": context.get("chunk_id"),
            "title": context.get("title"),
            "url": context.get("url"),
            "source_type": context.get("source_type"),
            "source_id": context.get("source_id"),
            "chunk_index": context.get("chunk_index"),
        }
        sources.append(source)

    return sources

def answer_query(query, top_k=5):

    contexts = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, contexts)

    answer = call_llm(prompt)
    sources = build_sources(contexts)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
    }


def main():
    query = "最新的三条记录是哪三条？"

    result = answer_query(query, top_k=5)

    print("Question:")
    print(result["query"])
    print()
    print("Answer:")
    print(result["answer"])
    print()
    print("Sources:")
    for source in result["sources"]:
        print(source)

if __name__ == "__main__":
    main()