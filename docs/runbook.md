# Runbook: finishing the CUDA backend

Follow top to bottom. Every step has a command, an exit criterion, and what to do
when it fails. Design decisions for each kernel are in
[kernel-plan.md](kernel-plan.md); read that before session 2 rather than during
it.

The guiding split: **correctness work does not need a fast GPU, benchmarking
does.** Debug on the cheapest card you can get, and rent the card whose numbers
you intend to publish only to measure.

---

## Session 0 — before paying for anything (free, ~1 hour)

```bash
# On the Mac, confirm the starting state is actually green.
cd ~/Documents/engine
cmake -S . -B build -G Ninja && cmake --build build
./build/tests/engine_tests                  # expect: all tests passed
python3 scripts/check_numbers.py            # expect: every quoted figure matches
```

Then read `docs/kernel-plan.md` §0 and §1–4. You want the host-side plumbing and
the three easy kernels designed before the clock starts.

**Exit:** tests green, and you can state from memory what `slot(pos)` is and why
cuBLAS needs the transpose trick.

---

## Session 1 — bring-up and the easy kernels (~4 hours)

> Follow [session1.md](session1.md) for this one; it is the same plan with every
> command written out, including the cuBLAS wrapper test and the stub files that
> keep the tree building from the first minute.

Rent the cheap GPU. Any NVIDIA card; this session does not care how fast it is.
Pick a CUDA **devel** image so `nvcc` is present.

```bash
git clone https://github.com/AarnavNoble/llm-engine engine && cd engine
scripts/gpu_bringup.sh
```

That installs dependencies, fetches weights, builds the CPU backend, runs the
full suite, generates the PyTorch per-layer dumps and the per-operation golden
tensors, and prints the compute capability to build for. It is idempotent, so
re-run it after any pod restart.

**Exit:** `gpu_bringup.sh` finishes with "ready", and it tells you the CUDA
backend is not implemented, which is the work below.

### Then, in this order

1. `src/model/cuda/cuda_model.cu` — device allocation, weight upload, the KV
   cache, the per-step int arrays, the cuBLAS wrapper. Nothing clever.
2. **Test the GEMM wrapper on its own** before anything depends on it. A wrong
   transpose is indistinguishable from a wrong kernel and will cost you hours.
3. `embedding.cu`, then `rmsnorm.cu`, then `silu_mul.cu`.

Loop on each with:

```bash
scripts/kernel_dev.sh rmsnorm     # build, compare that one operation
scripts/kernel_dev.sh             # all operations
scripts/kernel_dev.sh --watch     # re-run on save
```

**Exit:** `scripts/kernel_dev.sh` green for embedding, rmsnorm and silu_mul.

**If an operation disagrees:** fix the earliest one in the layer first; every
later tensor is computed from its output. The test prints `max |diff|` against
`max |expected|`, so a diff at the magnitude of the values is a wrong formula and
a diff around 1e-3 is a precision problem (almost always fp16 accumulation where
the reference uses fp32).

---

## Session 2 — RoPE, naive attention, first oracle parity (~5 hours)

```bash
# 4. rope.cu, fused with the K/V cache write
scripts/kernel_dev.sh rope

# 5. naive contiguous attention, enough to get end to end
scripts/kernel_dev.sh

# 6. the real test: every layer against PyTorch
scripts/kernel_dev.sh --model
```

**Exit — the milestone of the whole project:** `--model` green means GPU logits
match the CPU oracle after all 24 layers. From here you are optimising a correct
engine rather than debugging an incorrect one.

Also check it generates text:

```bash
./build/engine generate --model models/Qwen2.5-0.5B-Instruct --backend cuda \
  --prompt "The capital of France is" --max-tokens 16
```

**If the per-layer test fails but all operations pass:** the bug is in the
plumbing, not a kernel — wrong stride, wrong residual add, a layer's weights
pointing at the previous layer's buffers, or the logits gather taking the wrong
rows.

Commit here. `git tag v0.1-cuda-naive`.

---

## Session 3 — the paged decode kernel (~5 hours)

Read `docs/kernel-plan.md` §5 first. This is the kernel the project exists for.

```bash
scripts/kernel_dev.sh attention     # against the golden tensor
scripts/kernel_dev.sh --model       # in context, both models
./build/tests/engine_tests "[model]"  # includes the three invariances
```

**Exit:** golden check green, per-layer green, and the chunk and preemption
invariance tests still pass. Those two are what catch a kernel that reads the
block table correctly only when blocks happen to be contiguous.

**Deliberately break the easy case:** the golden manifest's block table is
`[0, 1, 2]`. Allocate a sequence, free a block, allocate again so the table is
non-contiguous, and re-check. A kernel that ignores the block table passes the
default test and fails this one.

Commit. `git tag v0.2-paged-attention`.

---

## Session 4 — prefill, sampling, and the table (~4 hours)

```bash
# 7. attention_prefill.cu (or cuBLAS + masked softmax; say which in the README)
# 8. sampling.cu
scripts/kernel_dev.sh --model
./build/tests/engine_tests            # everything, including the stress test
python3 scripts/e2e_smoke.py --backend cuda
```

Then the numbers. Switch to the 4090 you intend to cite:

```bash
scripts/reproduce.sh                  # fills the serving rows
python3 bench/vllm_baseline.py --tag vllm --runs 3
python3 bench/report.py
python3 scripts/check_numbers.py      # fails if prose and data disagree
```

**Exit:** `docs/results.md` has no "not measured" in the serving table.

Stop the pod.

---

## Session 5 — optimisation rows, optional (~4 hours)

Only after the table is filled, so each change has a before and an after.

```bash
# each of these is one commit with one number
#  - fuse rope + cache write (if not already), measure the saved traffic
#  - 16-byte vectorised loads in decode attention
#  - one block per (sequence, kv_head), reading K/V once instead of 7 times
#  - split-K for contexts over ~2K
ncu --set full -k regex:attention_decode -c 3 \
  ./build/engine generate --model models/Qwen2.5-0.5B-Instruct --backend cuda \
  --prompt hi --max-tokens 8
```

Record `dram__bytes.sum` and the kernel duration, divide, and compare against
1008 GB/s. Put the table in `docs/kernels.md`.

```bash
# CUDA graphs for decode at fixed batch sizes
nsys profile -o docs/prof/decode ./build/engine generate ... --max-tokens 64
```

**Exit:** a per-kernel bandwidth table, and a before/after `nsys` timeline.

---

## Session 6 — the parts that are not code (no GPU)

```bash
python3 bench/report.py               # regenerates docs/results.{md,html}
```

- Host `docs/results.html` — it is standalone, so GitHub Pages works.
- Write-up part 2 with the GPU numbers; part 1 is already in `docs/writeup.md`.
- Two-minute recording: server starting, a streamed completion, the benchmark
  running, Grafana.
- Fill X, Y and Z in the resume bullets at the bottom of
  `experience/inference-server-project-plan.md`.
- `git tag v1.0`.

---

## Things that will cost you money if you forget

1. **Stop the pod.** Closing the tab does not stop billing. A forgotten pod
   overnight costs more than this entire plan.
2. **Keep the volume, destroy the pod.** Storage is pennies a day; rebuilding
   the repo, weights and dumps every session is not.
3. **Use a devel image.** Installing the CUDA toolkit on the clock is twenty
   wasted minutes.
4. **Stop while thinking.** Designing a kernel does not need a GPU attached.
5. **Spot for debugging, on-demand for benchmarks.** A preemption mid-measurement
   wastes the run; mid-debugging it costs nothing, because everything is pushed.

## The three commands worth memorising

```bash
scripts/gpu_bringup.sh        # start of every session
scripts/kernel_dev.sh         # the inner loop
scripts/reproduce.sh          # the table
```
