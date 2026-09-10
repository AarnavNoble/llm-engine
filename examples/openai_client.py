#!/usr/bin/env python3
"""Any OpenAI client works against the engine. pip install openai"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="engine",
    messages=[{"role": "user", "content": "Explain paged attention in two sentences."}],
    max_tokens=96, temperature=0.0, stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
