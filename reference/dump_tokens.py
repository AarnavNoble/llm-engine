#!/usr/bin/env python3
"""Write tests/data/tokens.jsonl: {text, ids} pairs from the HF tokenizer, the ground truth
for the C++ tokenizer test. Includes chat-template rendering cases."""
import json, sys, pathlib
from transformers import AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-0.5B-Instruct"
out = pathlib.Path("tests/data/tokens.jsonl")
tok = AutoTokenizer.from_pretrained(model_dir)

texts = [
    "Hello world", "hello world", " Hello  world ", "Hello, world!", "I'm here. You're not. We'll see; they'd've known.",
    "It's 2024 and the price is $1,234.56 (roughly 3.14%).", "1234567890", "a1b2c3", "  leading spaces", "trailing spaces  ",
    "tabs\tand\ttabs", "line1\nline2\n\nline3\r\nline4", "\n\n\n", "   \n  x", "x   \n", "", " ", "    ", "!!!", "...and so on...",
    "def f(x):\n    return x ** 2  # square\n", "int main() { return 0; }", "SELECT * FROM users WHERE id = 42;",
    "The quick brown fox jumps over the lazy dog.", "THE QUICK BROWN FOX", "CamelCaseIdentifier snake_case_identifier kebab-case",
    "email@example.com https://example.com/path?q=1&r=2", "naïve café résumé", "日本語のテキスト", "你好，世界！", "Привет, мир!",
    "مرحبا بالعالم", "emoji 😀🚀 test", "mixed 日本語 and English 123 text", "<|im_start|>user\nhi<|im_end|>\n",
    "<|endoftext|>", "text<|endoftext|>more", "don't DON'T Don't", "'s 's 'S", "x's y't z're", "π ≈ 3.14159", "a—b–c-d",
    "quotes \"double\" and 'single'", "unicode spaces nbsp em", "ends with newline\n", "\nstarts with newline",
    "multiple    spaces    between", "12 34 567 8901", "1.5e-10", "-42", "+1", "#hashtag @mention", "C++ and C# and F#",
    "ok\r\n", "\r\n\r\n", "a\n b\n  c", "The year 2025—what a time.", "  x  y  ", "Ünïcödé ÀÉÎÕÜ", "ß straße", "ΑΒΓ αβγ",
    "한국어 텍스트", "हिन्दी पाठ", "עברית", "ไทย", "🇺🇸 flags 🏳️‍🌈", "﷽", "𝕳𝖊𝖑𝖑𝖔", "①②③", "½ ¾", "Ⅻ",
]
# Long paragraph.
texts.append(("Paged attention partitions the KV cache of each sequence into fixed-size blocks, "
              "which may be stored in non-contiguous physical memory. " * 5).strip())

rows = []
for t in texts:
    rows.append({"text": t, "ids": tok.encode(t, add_special_tokens=False)})
# Chat template cases (no tools).
convs = [
    [{"role": "user", "content": "What is paged attention?"}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say hi."}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}],
]
for c in convs:
    rendered = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
    rows.append({"chat": c, "text": rendered, "ids": tok.encode(rendered, add_special_tokens=False)})

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), "rows to", out, "| eos:", tok.eos_token, tok.eos_token_id)
