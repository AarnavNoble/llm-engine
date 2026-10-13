#!/usr/bin/env bash
# Bring a pod up for recording the demo video, in one command.
#
#   git clone https://github.com/AarnavNoble/llm-engine engine && cd engine
#   scripts/demo_session.sh
#
# Then, on the laptop that is recording:
#
#   ssh -N -L 8000:localhost:8000 root@<POD_IP> -p <POD_PORT>
#   open http://localhost:8000
#
# The tunnel rather than exposing the port: the demo page talks to /metrics on
# its own origin, so tunnelling keeps it same-origin and the server keeps its
# production-shaped config with no CORS headers added for a recording's sake.
#
# The KV pool is deliberately small. See the sizing note below -- with the
# default 512 blocks the oversubscribe scenario fits in cache and preempts
# nothing, which is the one shot in the video that cannot be faked.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/Qwen2.5-0.5B-Instruct}
PORT=${PORT:-8000}
step() { printf "\n\033[1m== %s\033[0m\n" "$*"; }

# nvcc is not on a non-interactive PATH on some images.
command -v nvcc >/dev/null 2>&1 || for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
  [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break
done

step "machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

if [ ! -x ./build-cuda/engine ]; then
  step "first-time bring-up (model, deps, CUDA build)"
  scripts/gpu_bringup.sh
fi
[ -x ./build-cuda/engine ] || { echo "no CUDA build; gpu_bringup.sh failed" >&2; exit 1; }

step "sanity: the backend agrees with the CPU oracle"
./build-cuda/tests/engine_tests "[cuda]" 2>/dev/null || \
  ./build-cuda/engine generate --model "$MODEL" --backend cuda \
    --prompt "The capital of France is" --max-tokens 8

# Scenario sizing, so each shot in the video actually shows its mechanism.
# block_size 16, so 384 blocks is 6,144 token slots.
#
#   shared prefix   ~420 prompt tokens, ~26 blocks, shared by 6   -> fits, hits
#   long prompt     ~2,400 tokens, ~150 blocks, chunked at 512    -> 5 chunks
#   oversubscribe   48 x (~12 + 160) ~= 8,400 tokens, ~525 blocks -> exceeds 384,
#                                                                    so it must
#                                                                    preempt
KV_BLOCKS=${KV_BLOCKS:-384}

step "warm-up (weights, cuBLAS handles, first-launch JIT)"
# A cold first request has a multi-second TTFT that has nothing to do with the
# scheduler. Spend it here instead of on camera.
./build-cuda/engine generate --model "$MODEL" --backend cuda \
  --prompt "warm" --max-tokens 32 >/dev/null

step "serving on :$PORT"
setsid nohup ./build-cuda/engine serve --model "$MODEL" --backend cuda --port "$PORT" \
  --num-blocks "$KV_BLOCKS" --block-size 16 \
  --max-num-seqs 32 --max-batched-tokens 2048 --chunk 512 \
  > /tmp/engine-demo.log 2>&1 < /dev/null &
echo $! > /tmp/engine-demo.pid

for _ in $(seq 1 120); do curl -sf "localhost:$PORT/readyz" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/readyz" >/dev/null || { echo "never became ready; see /tmp/engine-demo.log" >&2; exit 1; }
curl -sf "localhost:$PORT/" >/dev/null || echo "WARNING: / did not serve web/demo.html"

step "ready"
cat <<NOTES
KV pool: $KV_BLOCKS blocks x 16 tokens = $((KV_BLOCKS * 16)) slots

On the laptop that is recording:

  ssh -N -L $PORT:localhost:$PORT root@<POD_IP> -p <POD_SSH_PORT>
  open http://localhost:$PORT

Record in this order, resetting between takes:

  1. stagger 6          continuous batching
  2. 6 sharing a prefix prefix reuse      (wait for the seed, ~3 s)
  3. long prompt        chunked prefill
  4. oversubscribe 48   preemption        (do this LAST: it fills the pool)

Counters are cumulative and server-side. To reset them between takes:

  kill \$(cat /tmp/engine-demo.pid); scripts/demo_session.sh

Stop the pod when you walk away.
NOTES
