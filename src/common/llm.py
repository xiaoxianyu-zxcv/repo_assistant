import os

from openai import OpenAI


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def create_llm_client():
    # 创建 DeepSeek 客户端。DeepSeek 兼容 OpenAI SDK，只需要改 base_url。
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("deepseek密钥不存在")

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def call_llm(prompt):
    # 把 RAG prompt 发给大模型，返回模型生成的答案。
    client = create_llm_client()

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a careful repository assistant. Answer only from the provided context."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        stream=False,
    )

    return response.choices[0].message.content
