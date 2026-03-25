from openai import OpenAI

OLLAMA_BASE_URL = "http://192.168.1.11:11434/v1"
MODEL = "qwen3.5:9b"


def main():
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "你好，告诉我大模型部署方面的知识"}],
        temperature=0.6,
        max_tokens=4096,
        stream=True,
    )

    in_reasoning = False
    in_content = False
    for chunk in stream:
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            if not in_reasoning:
                print("【思考过程】")
                in_reasoning = True
            print(reasoning, end="", flush=True)
        if delta.content:
            if not in_content:
                if in_reasoning:
                    print("\n")
                print("【回答】")
                in_content = True
            print(delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
