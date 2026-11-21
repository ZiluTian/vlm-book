# Chapter 9 — Inference and Serving: Where the Money Goes

> 🎯 **The one sentence:** LLM serving is cache management + scheduling: the KV cache
> is your memory problem, continuous batching is your throughput problem, FlashAttention
> is your attention-memory problem, speculative decoding is your latency problem — and
> vision tokens make every one of them 10× bigger.

## 9.1 The two phases, one last time (then we stop repeating)

- **Prefill:** run the whole input (P vision + L text) in parallel. Compute-bound.
  Builds the KV cache. Cost ≈ O(S²·D) attention + O(S·D²) matmuls.
- **Decode:** one token per step. Memory-bandwidth-bound: every step re-reads
  *weights* (N params) + *all KV so far*. Nothing is computed from scratch — the
  KV cache is the reuse.

**Check:** which phase does a *long* image hurt? → *prefill (S² attention + S·D²
matmuls) — and then decode pays a permanent KV-tax on it (§9.10).*

## 9.2 Decode bandwidth: the arithmetic that decides hardware

Per decode token, you read (fp16): weights `2·N` bytes + KV `2·T_total·n_kv_heads·head_dim·2` bytes.
For 7B + 4k context (GQA-8): ~14 GB + ~0.5 GB ≈ **14.5 GB per token**.
On an H100 (~3 TB/s HBM): theoretical max ≈ 3.3k tok/s — *for a single user*.
Real serving: 100–400 tok/s/user → the model is bandwidth-starved, not FLOP-starved.

**Consequence (say this in interviews):** decode speed is limited by memory
bandwidth, so (a) fewer/bigger GQA groups help, (b) quantization helps (fewer bytes
to read), (c) batching helps *throughput* (weights amortized over users) but costs
*latency*, and (d) KV pruning (§8.4) cuts the per-token read.

## 9.3 The KV cache, concretely

```python
class KVCache:
    def __init__(self, n_layers, n_kv_heads, head_dim, max_seq, dtype=fp16):
        self.K = [zeros(max_seq, n_kv_heads*head_dim) for _ in range(n_layers)]
        self.V = [like(self.K)]
    def append(self, layer, k_new, v_new, pos):
        self.K[layer][pos:pos+len(k_new)] = k_new
        self.V[layer][pos:pos+len(v_new)] = v_new
    # attention at position p: q[p] · K[layer][:p]  (causal slice, no recompute)
```

Per-user memory = `2 (K+V) × n_layers × n_kv_heads × head_dim × 2 bytes × T_max`.
7B-GQA @ 32k: 2·32·8·128·2 = 128 KB/token → **4 GB/user**. A 70B (80 layers, 8 kv
heads): 320 KB/token → **10 GB/user**. That single number explains most serving
architecture. (You did this math in §2.14 — now you use it to *design*.)

## 9.4 Paged KV: vLLM's core idea

Problem: if you reserve `max_seq` KV per request up front, you waste ~60–80% of
memory on *potential* sequences that never happen (same story as OS virtual memory).
vLLM's answer: **page the KV cache** — allocate in blocks (e.g. 16 tokens), keep a
block table per request, grow on demand:

```python
# per request: block_table = [block_id, ...];  block = 16 tokens of (K,V) across layers
def allocate_tokens(req, n_new):
    need = ceil((req.cached + n_new) / BLOCK) - len(req.block_table)
    for _ in range(need):
        req.block_table.append(alloc_block())   # from a global free pool
```

Payoffs: ~no reserved waste, **zero copy** between prefill and decode (same blocks),
and — the killer feature — **prefix sharing**: two requests with the same system
prompt (or the same cached image!) literally share physical KV blocks.

**Check:** why is "prefix sharing" a *memory* feature and not a compute feature?
→ *the shared blocks are never re-read for compute savings alone — they stop being
*allocated per request*. One system prompt, 1k users: KV for it lives once.*

## 9.5 Continuous batching: the scheduler that made serving real

Static batching waits for the slowest sequence in the batch to finish before
starting the next batch. Continuous batching **swaps requests in/out every step**:

```python
def serving_loop(model, queue, max_batch_tokens=8192):
    running = []                      # active (req, its KV blocks)
    while True:
        # 1) admit: pull queued requests while the token budget allows
        while queue and can_fit(queue.front(), running, max_batch_tokens):
            running.append(prefill(queue.pop(), model))      # build its KV
        # 2) one decode step over EVERY running request (their tokens are batched)
        out = model.decode_step(batch_of_last_tokens(running))
        # 3) settle: append tokens; free blocks for finished requests
        for req, tok in zip(running, out):
            req.append(tok)
            if tok == EOS or req.done: free(req.blocks)
```

Two numbers to know: (a) **prefill and decode have different bottlenecks** —
naively mixing a huge prefill into a decode step spikes latency (modern engines
schedule them separately, or chunk prefill); (b) throughput vs latency is a *dial*
(bigger batches → more tok/s total, more p99 latency per request). SLOs are set
*on the dial*, not on the model.

**Check:** why does batching raise total throughput but per-request latency?
→ *each step's fixed cost (reading weights) is amortized over the batch — total
work/byte drops; but every request in the batch waits for the step to finish, so
per-request step time grows with batch size.*

## 9.6 FlashAttention: why attention doesn't OOM

Naive attention materializes the `[T, T]` score matrix → O(T²) memory (8k context:
67M entries per head-layer — and you have hundreds of head-layers).
FlashAttention: **tile the computation** — process q/k/v in blocks that fit SRAM,
compute the softmax online (running max + running sum), never store the full
score matrix. Same result, O(T) memory, and it's *faster* (HBM traffic drops).
FA2/FA3 add better scheduling and FP8 paths.

**Check:** why does tiling *speed up* attention, not just save memory?
→ *the bottleneck was HBM round-trips for the score matrix; tiling keeps the hot
blocks in SRAM and fuses the elementwise ops — fewer bytes moved, same FLOPs.*

## 9.7 Prefix caching (the MLLM superpower)

Anything that repeats across requests — system prompt, few-shot examples, **the
image tokens of a long multi-turn chat with the same image** — can be cached:
request 5 of the same image+conversation re-encodes *nothing* visual; it just
attaches to the existing KV. For VLM products (document Q&A, dashboard monitoring,
AV clip review) this is often a **10× prefill cost cut**. Design note: cache keys
must be byte-exact (template! image bytes! resolution bucket!).

**Check:** what breaks prefix caching in a VLM product? → *anything that changes the
byte prefix: adaptive resolution that varies per call, template drift, different
image preprocessing. Stabilize the pipeline, then cache.*

## 9.8 Speculative decoding: latency without retraining

Idea: a cheap **draft** model (or the same model's early layers) proposes k tokens;
the big model verifies them **in one parallel prefill**; accept the longest correct
prefix, resample the first mismatch:

```python
def speculative_decode(big, draft, x, k=5):
    y = draft.generate(x, k)              # k small tokens, cheap
    p_big = big.forward(x + y)            # one pass scores ALL k candidates
    for i, yi in enumerate(y):
        if sample(p_big[:, i]) != yi:     # first disagreement
            return (x + y[:i] + resample) # accept prefix, fix the tail
    return x + y                          # all k accepted (rare: speedup ≈ k)
```

Rejection sampling keeps the distribution *exactly* the big model's — free quality,
2–3× decode latency typically. For MLLMs: draft the *text* part only; the image KV
is shared — spec-decode is even better when the vision prefix dominates.

**Check:** why is the big model's single verification pass valid for k tokens at
once? → *it's just prefill over x+y: causal attention gives p(yi | x, y<i) for
every i in one pass — the same math as training, parallelized.*

## 9.9 Quantization for serving (the menu)

- **Weights only (W4/W8, A16):** AWQ/GPTQ-style — 4-bit weights, dequant on the fly.
  ~2–4× less weight traffic → near-linear decode speedup, <1–2% quality loss (well-tuned).
- **W8A8:** 8-bit activations too → faster matmuls; more calibration care.
- **KV quant (fp8/int8 KV):** cuts the *other* half of the bandwidth bill (§9.2) —
  increasingly standard at long context.
- **The AV note:** on-vehicle serving = int4 weights + int8 KV + small GQA + short
  context; cloud = the full menu plus spec-decode and paged KV.

**Check:** which quantization helps *prefill* and which helps *decode*?
→ *weights-only helps decode (the bandwidth-bound phase); activation quant helps
prefill (compute-bound, fewer FLOPs via int8 tensor cores).*

## 9.10 MLLM serving: vision token economics

Per-image costs, made explicit (336 px, 576 tokens, 7B-GQA):
- prefill: +576 into S → attention +576·D·2·… and S² growth;
- decode: +576 × 128 KB = **+72 MB of KV** for the *entire conversation*;
- tiled hi-res (2880 tokens): +360 MB. Now you *see* why token budgets are product
  decisions.
Standard levers, in order: (1) don't over-tile; (2) prefix-cache the image across
turns; (3) prune vision KV post-prefill (§8.4); (4) quantize KV; (5) smaller GQA.

**Check:** a user sends 20 images in one document Q&A (each 1k tokens): what's your
first architectural response? → *reject/summarize: run a cheap pass (or a small
VLM) per image to produce text summaries, then one long-context query over the
summaries. Rerouting beats raw context.*

## 9.11 The cost worksheet (do this out loud in interviews)

Given: model N params, context T, batch b, tok/s target.
1. **Memory:** weights (N×bytes) + KV (b×T×KV/token) + activations ≈ fits where?
2. **Throughput:** bandwidth ÷ bytes-per-step ÷ b → tok/s/user; ×b → tok/s total.
3. **Cost:** $/GPU-hour ÷ (tok/s × 3600) → $/1M tokens. Compare to API price.
Worked example: 7B-int4 (~4 GB) on 1×H100, 1k users × 8k ctx, KV 128 KB/token
→ KV alone ≈ 10 GB ×… wait: 1k×8k×128KB = **1 TB** — doesn't fit; answer: GQA is
already small, so *quantize KV to int8 (halve), cap context at 4k (halve), and
you're at ~250 GB → 2–4 H100s + vLLM paging*. The *structure* of that reasoning is
what they grade, not the exact factor of two.

**Check:** what single change usually has the biggest cost impact in a VLM service?
→ *the vision token budget (how many image tokens per query) — it moves both
prefill FLOPs and the permanent KV tax, for every request.*

## 9.12 ❓ Interview

**Q: How much KV memory for X?** — formula + numbers, shown (§9.3). Then: "so we
quantize the KV / cap context / share prefixes."

**Q: Explain vLLM in one paragraph.** — Paged KV (blocks, block tables, no
pre-allocation waste, prefix sharing) + continuous batching (per-step admission)
→ 2–3× throughput at equal quality vs static batching; the de-facto open engine.

**Q: FlashAttention, why and how?** — OOMs from the [T,T] score matrix; tile q/k/v
in SRAM, online softmax, O(T) memory, fewer HBM bytes → also faster.

**Q: Why is decoding bandwidth-bound?** — per token: read all weights + all KV;
HBM/s sets the ceiling; batching amortizes weight reads; quantization cuts bytes;
pruning cuts KV. (Four levers, one cause.)

**Q: Design a VLM endpoint for 100k document-questions/day, p95 < 5s.**
— (1) budget: avg ~2k vision + 512 text tokens, 300-token answers; (2) int4 7–8B +
GQA + int8 KV on 4–8 H100s with vLLM; (3) prefix-cache stable system+image blocks;
(4) chunk/limit images per query (§9.10) — route "many images" to a summarize-then-ask
pipeline; (5) spec-decode for the text tail; (6) load-test the *dial*: batch size
vs p95. Show you'd measure, not guess.
