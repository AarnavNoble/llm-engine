# engine

An LLM inference server in C++ and CUDA, written from scratch.

I want to understand how vLLM, SGLang and TensorRT-LLM actually work, so I'm building the pieces myself: a paged KV cache with block tables, continuous batching, prefix caching, hand-written CUDA kernels for the decode path, and an OpenAI-compatible streaming API in front of it. No PyTorch at runtime — weights come straight from safetensors, the tokenizer parses `tokenizer.json` directly, and the forward pass is my kernels plus cuBLAS.

Target model: Qwen2.5-0.5B-Instruct (fp16), later TinyLlama to prove the loader is generic. Target GPU: a single RTX 4090.

The plan, roughly in order:

- [ ] safetensors loader, config, tokenizer
- [ ] CPU reference forward pass, checked layer by layer against PyTorch
- [ ] paged KV cache and block allocator
- [ ] continuous batching scheduler with chunked prefill and preemption
- [ ] prefix caching
- [ ] CUDA backend: cuBLAS GEMMs + custom rmsnorm / rope / silu·up / paged attention / sampling kernels
- [ ] OpenAI-compatible HTTP API with SSE streaming, Prometheus metrics
- [ ] benchmark table: each optimization measured against the previous one
- [ ] CUDA graphs, kernel fusion, Nsight numbers
- [ ] Dockerfile, Helm chart, KEDA autoscaling on queue depth

Everything is developed CPU-first on a Mac (the CPU path is the numerical oracle) and then ported to the GPU, so the whole thing builds and tests without a GPU.

## Building

```bash
cmake -S . -B build -G Ninja && cmake --build build && ./build/tests/engine_tests
```

MIT licensed.
