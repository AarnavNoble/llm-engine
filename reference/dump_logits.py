#!/usr/bin/env python3
"""Ground truth for the numerical tests.

Runs the HF model in fp32 on a fixed prompt set and writes
  tests/data/prompts.json            (committed)  ids, per-position argmax, greedy continuation
  tests/data/ref/<i>.bin             (gitignored) residual stream after embedding and after every
                                                  layer, plus the last-token logits, all fp32
Layout of each .bin: int32 n_tokens, n_layers, hidden, vocab; then
  float32 resid[n_layers+1][n_tokens][hidden]; float32 logits[vocab].
"""
import json, pathlib, struct, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-0.5B-Instruct"
out_dir = pathlib.Path("tests/data"); (out_dir / "ref").mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
cfg = model.config

texts = [
    "The capital of France is", "def fibonacci(n):", "1 + 1 =", "Once upon a time,", "Paged attention",
    "The quick brown fox jumps over the lazy dog. The quick brown fox", "In 1969, humans first",
    "SELECT name FROM users WHERE", "こんにちは、私の名前は", "Explain continuous batching in one sentence:",
    "x", "To be, or not to be, that is the", "The GPU has 24 GB of memory and", "import numpy as np\nimport torch\n",
    "Q: What is 17 * 23?\nA:", "Roses are red, violets are blue,", "The mitochondria is the",
    "CUDA kernels are launched with a grid of", "She opened the door and saw", "In conclusion,",
]
chats = [[{"role": "user", "content": "What is paged attention? Answer in one sentence."}],
         [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name three GPU vendors."}]]
prompts = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in texts]
for c in chats:
    r = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
    prompts.append({"text": r, "ids": tok.encode(r, add_special_tokens=False), "chat": True})

resid = []
hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: resid.append(o.detach()[0].clone()))]
for layer in model.model.layers:
    hooks.append(layer.register_forward_hook(lambda m, i, o: resid.append((o[0] if isinstance(o, tuple) else o).detach()[0].clone())))

with torch.no_grad():
    for i, p in enumerate(prompts):
        resid.clear()
        ids = torch.tensor([p["ids"]])
        logits = model(ids).logits[0]
        assert len(resid) == cfg.num_hidden_layers + 1, len(resid)
        saved = torch.stack(resid)          # snapshot: generate() below fires the hooks again
        p["argmax"] = logits.argmax(-1).tolist()
        # generation_config.json sets repetition_penalty=1.1 (override it), and without an
        # explicit mask generate() masks positions equal to pad_token_id — chat prompts contain
        # <|im_end|>, so always pass an all-ones mask.
        gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=16, do_sample=False,
                             eos_token_id=None, pad_token_id=tok.eos_token_id,
                             repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
        p["greedy_16"] = gen[0, ids.shape[1]:].tolist()
        n = ids.shape[1]
        with open(out_dir / "ref" / f"{i}.bin", "wb") as f:
            f.write(struct.pack("<iiii", n, cfg.num_hidden_layers, cfg.hidden_size, logits.shape[-1]))
            f.write(saved.numpy().astype(np.float32).tobytes())
            f.write(logits[-1].numpy().astype(np.float32).tobytes())
        print(f"[{i}] n={n} argmax_last={p['argmax'][-1]!r} greedy={tok.decode(p['greedy_16'])!r}")

json.dump({"model": model_dir, "prompts": prompts}, open(out_dir / "prompts.json", "w"), ensure_ascii=False, indent=0)
print("wrote", len(prompts), "prompts")
