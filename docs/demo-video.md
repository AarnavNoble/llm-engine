# The two-minute video: shot list

What it is for: **evidence of comprehension, not a feature tour.** "I wrote an
LLM inference server" is a crowded claim and a reader cannot tell this repo from
the hundreds that are a forward pass behind a Flask route. Ninety seconds of a
scheduler behaving correctly, plus three numbers one of which is unflattering,
settles it.

Audience: an infra engineer with ninety seconds, who clicked the link in a
resume header. They already passed the keyword screen. So do not explain what a
KV cache is -- they know, or they are not the viewer.

Record raw and trim after. Do not try for one take.

## Setup

On the pod:

    scripts/demo_session.sh

On the recording laptop:

    ssh -N -L 8000:localhost:8000 root@<POD_IP> -p <POD_SSH_PORT>
    open http://localhost:8000

Browser at 1280x800, zoom 100%, both columns visible without scrolling. Hide
bookmarks. One terminal window, large font, nothing else on screen.

**Record on the pod, not locally.** The CPU backend runs the page at about
47 tok/s; every scenario looks sluggish and the point is lost.

## Shots

### 1. Cold open, 0:00-0:10

Terminal, the server line scrolling past.

> "This is an inference server I wrote in C++ and CUDA. No PyTorch, no
> frameworks past cuBLAS -- the scheduler, the paged KV cache and the attention
> kernels are mine."

No greeting, no name, no title card. Ten seconds of "hi, I'm ..." loses the
viewer you want.

### 2. Continuous batching, 0:10-0:35

Click **stagger 6, one per second**. Let all six land.

> "Six requests, one per second, into a batch that is already decoding. Each one
> joins on the next step -- it does not wait for the others to finish. That is
> continuous batching, and the count going up while other lanes are still
> streaming is the whole of it."

Watch for: `decoding now` climbing while earlier lanes still produce tokens.

### 3. Prefix sharing, 0:35-0:55

Click **6 sharing a long prefix**. The button seeds one request and waits
~2.5 s before sending five more -- do not cut that pause, the jump after it is
the shot.

> "Same system prompt on all six. The blocks holding it are refcounted and
> pointed at, not copied -- so the prefix is in memory once. Twenty-one of
> forty-two blocks reused."

Watch for: `prefix blocks reused` jumping, the cached slice of the KV bar
growing.

### 4. Chunked prefill, 0:55-1:10

Click **inject a 2,000-word prompt** while requests from shot 3 are still
decoding.

> "A long prompt arriving mid-flight. It is prefilled in 512-token chunks, so it
> cannot stall the requests already running -- they keep producing tokens
> through it."

Watch for: the other lanes' token counts still rising.

### 5. Preemption, 1:10-1:25

Click **oversubscribe (48)**. Do this last; it fills the pool.

> "Forty-eight at once, past what the KV pool holds. Rather than thrash or fail,
> the scheduler evicts a live sequence, frees its blocks, and recomputes it when
> there is room. Some requests get shed outright, which is load shedding working."

Watch for: `preemptions` rising and flashing, `recomputed tokens` non-zero.

### 6. The numbers, 1:25-1:55

Cut to https://aarnavnoble.github.io/llm-engine/ -- or just talk over the
results table.

> "Measured on an A40. Continuous batching against static: median time to first
> token, four seconds down to seventy-five milliseconds. The attention kernel is
> 1.62x faster after I made each K/V read serve a whole GQA group instead of one
> query head.
>
> And it was 1.33x *slower* the first time. One block per key/value head instead
> of per query head meant seven times fewer blocks -- sixty-four of them on a
> card with eighty-four SMs. The bandwidth I saved could not pay for a GPU that
> was three quarters idle. Five hundred and twelve threads a block fixed it.
>
> vLLM on the pod where I measured both is fifteen times faster than this. That
> number is published on the same page."

Then stop. No outro.

## Why the last thirty seconds are the video

Everything up to 1:25 is a competent person showing a working system; plenty of
candidates can do that. The negative result and the published vLLM gap are the
part nobody embellishing a project volunteers, so they read as someone who
measures rather than asserts -- which is the job. They are also the most
interview-generating sentences available here, because the obvious follow-up is
"so why *was* it slower?" and that answer runs ten minutes.

## Leave out

- Your face, a slide intro, music, the architecture diagram.
- Scrolling the source. Unwatchable, and the repo is one click away.
- Any figure the results page does not already back. The credibility of the
  video is that every number in it is checkable.

## Figures quoted above, and where they come from

| claim | source | note |
|---|---|---|
| TTFT 3,993 ms -> 75 ms | `bench/results/gpu-static.json`, `gpu-continuous.json` | 53.2x, same instance `isx98c0` |
| attention 31.56 -> 19.43 ms/step | `kernel-time.json`, `kernel-time-gqa.json` | 1.62x, 728 steps |
| 1.33x slower at 4 warps | `src/kernels/attention_decode_gqa.cu:50` | the sweep is in the comment |
| 64 blocks, 84 SMs | same | A40 |
| vLLM 15.4x | `vllm.json` vs engine rows on pod `b0flqxf` | **say "on the pod where I measured both"** -- the engine rows on that pod are not in `bench/results/`, see `docs/results.md` |

Do not say "fifteen times faster" without the instance qualifier. The engine
rows currently in `bench/results/` come from the faster of two A40s that
differed by a factor of two for reasons still unknown, and dividing across them
would flatter this engine by about half.
