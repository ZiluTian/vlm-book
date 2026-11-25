# Chapter 8 — The Frontier: What's Actually Moving

> 🎯 **The one sentence:** the field is (a) going **native** — pixels tokenized, no
> bolt-on encoder; (b) getting **video/long** — time becomes a position axis; (c)
> spending its engineering budget on **token efficiency**, because vision tokens are
> the new context-length problem; and (d) learning to **reason and act** — R1-style
> RL, tools, and box outputs.

## 8.1 Native multimodal: delete the encoder and the connector

Family C (§1.4) pretrains one transformer on interleaved data with a shared
tokenizer:

```
... text ... [img_patch_tokens] ... text ... [img_patch_tokens] ... text ...
                       └── same embedding space, same positions, one model
```

**What it buys:**
- Interleaved chains are *native*: text → image → text needs no boundary crossing.
- No projector drift: vision and language co-adapt for the full pretraining run.
- Cleaner long-context: images are just more tokens, cacheable like text.

**What it costs:** you can't freeze a big LLM and bolt vision on; vision quality is
only as good as the pretraining mix; tokenizers for pixels (VQ, ~256–32k codebook)
add a lossy compression step *before* the model even sees the image.

**Check:** why does "native" still often keep a *separate early vision stage*
(like a patch encoder feeding the transformer)? → *raw pixels are 3×H×W values —
far too many "tokens" to feed a transformer directly; you still need a cheap
spatial compression. 'Native' means no *alignment* boundary, not no *compression*.*

## 8.2 Video: time becomes a position axis

One frame = a grid of tokens. A clip = many grids. Two problems:

1. **Time in positions:** Qwen2-VL's **M-RoPE** uses three axes `(t, h, w)`.
   Text uses the same position index on all three axes; visual tokens use
   time, row, and column indices. Position encoding tells the model where a
   retained token came from; it cannot restore a frame that was never sampled.
2. **Token explosion:** 16 frames × 576 = 9216 tokens before the question.
   Fixes, in increasing order of aggression:
   - **temporal pooling:** pair consecutive frames, mean-pool → half the tokens
     (each token = 2 frames);
   - **hierarchical summary:** coarse tokens for old segments, full tokens for
     recent ones (like your BEV: full detail now, summary far back);
   - **keyframe selection + text narration:** pick frames, caption the gaps.

**Check:** temporal pooling 2:1 on a 30-second 24fps clip at 576 tokens/frame —
how many tokens? → *720 frames → pool → 360 × 576 = 207k. You feel why
keyframe selection exists.*

🔗 **BEV bridge:** "full detail now, summary far back" is literally your temporal
BEV with decimated history. The VLM community is rebuilding your stack, one
attention layer at a time.

### Worked example: the warning light between sampled frames

**Why:** video compression can erase a short event even when each individual
frame has enough spatial detail. A dashboard warning visible for 200 ms can
fall entirely between samples taken every 500 ms.

**What:** a four-second clip sampled at 8 fps gives 32 frames. If each frame
contributes 144 decoder tokens (the 2×2 pooling example in [Chapter 5](ch05-bridging-vision-to-language.md)),
that's 4,608 tokens. At 2 fps you get eight frames and 1,152 tokens. Spatial
compression changes what survives *inside* a frame; sampling changes which
moments the model can see at all.

**How:** compare uniform sampling against a hybrid that keeps eight overview
frames plus up to eight distinct frames around an independently flagged event
(at most 2,304 tokens). First test a window centered on the annotated event to
see what denser sampling could recover, then use the real trigger and report
its miss rate. Measure warning-light recall by event duration, false alarms on
no-event clips, and latency. A higher score on long events alone does not show
that the short-event failure was fixed.

## 8.3 The token-efficiency war (the engineering story of 2024–26)

Facts:
- A single 336 px image ≈ 576 tokens; a hi-res tiled one ≈ 3–5k; a video ≈ 10⁴–10⁵.
- Vision tokens often **outnumber text tokens 10:1** in a VLM query.
- They're paid for twice: prefill FLOPs *and* KV memory for every generated token.
- And most of them, most of the time, carry little of what the answer needs
  ("what color is the car?" doesn't need the sky).

So the field builds **token diet**: what to drop, where to drop it, and how to know
which ones matter. Three mechanisms:

## 8.4 FastV: prune by attention, mid-network

Observation: by layer ~ℓ (mid-network), most vision tokens have near-zero
attention from the *text* query positions. Keep the top-k, drop the rest:

```python
# inside the LLM, at layer ℓ (e.g. 30% of depth), during inference:
k_vis = kv_cache[vision_positions].K      # [N_vis, D_kv]
q_now = current_q[answer_query_positions] # text-side queries
score = (q_now @ k_vis.T).mean(dim=heads) # [N_vis]  "how much does the image matter here"
keep  = topk(score, k)                    # e.g. keep 30% of vision tokens
# rebuild KV cache: remove dropped vision tokens' K,V from every layer
# continue decoding: attention now skips them, forever
```

30–70% fewer vision tokens, ~<1% quality drop on standard benchmarks, 2–3× faster
decode (less KV to read per token — pure bandwidth win, Ch 9). Variants: prune
per-layer (pyramidal: drop more at each deeper layer — the network itself decides
how much image it still needs), prune by per-head scores, prune at prefill only.

**Check:** why prune at layer ℓ, not layer 0? → *early layers don't know which
pixels matter — that's what the model *computes*. By mid-depth the attention maps
are the model's own salience signal. Layer 0 pruning = guessing; layer ℓ = reading
its mind.*

**Check:** when would FastV *hurt*? → *tasks where "every" pixel matters: dense
OCR, counting, fine spatial reasoning. The salience signal is answer-conditioned —
a counting question spreads attention thinly, so the pruner keeps too little.*

## 8.5 Token diet, other tools (name-drop list)

- **Pyramidal / progressive drop:** shrink the vision-token set layer by layer.
- **Dynamic resolution:** spend tokens *up front* only where the image is busy
  (entropy-based tiling — LLaVA-NeXT's adaptive grid is the simple version).
- **KV compression:** quantize or summarize the vision KV after prefill.
- **Early-exit vision:** tiny models run, big tokens only for hard queries.
- **The honest baseline:** just *don't tile* — cap resolution. Boring, and often
  within noise on coarse tasks.

**Check:** pruning (runtime) vs pooling (train-time) — same goal, different knob.
What does each sacrifice? → *pooling: fixed, known loss, cheaper at training too;
pruning: adaptive but answer-dependent, needs the mid-network pass, harder to verify.*

## 8.6 Multimodal reasoning: R1, but with eyes

The DeepSeek-R1 recipe (Ch 7 §7.9) transfers to vision:

1. **Data:** (image, problem, *interleaved* thinking) — think in text, call a tool
   (crop/zoom/OCR on a region), think more, answer.
2. **Reward:** verifiable — the final answer checked programmatically; format
   bonuses for tool calls and thinking blocks.
3. **GRPO:** group-sample answers, rank by reward, train.
4. **Result class:** the model learns to *inspect its own image* — a model that
   crops a chart before reading numbers beats one that squints.

This is the strongest 2024–26 trend in VLM capability: **test-time compute for
vision** (more thinking + more tool calls → better accuracy on hard visual
perception, at the price of latency and tokens).

**Check:** why does tool-augmented reasoning beat just "think harder" for a
perception error? → *the bottleneck is often *information*, not *inference*: the
right pixels were low-res in the token stream. A crop re-encodes them at full
resolution — new information, not just better reasoning over old tokens.*

## 8.7 Boxes and coordinates: grounding as output

Three eras: (1) boxes as text ("[32, 118, 512, 431]") — works, imprecise;
(2) **special box tokens** — a small learned vocabulary of coordinates, one token
per corner (Qwen-VL 2023, CogVLM); (3) native grounding heads (frontier, rare).
Interview one-liner: *"Coordinates-as-tokens are a token-level quantization of a
continuous output; box tokens reduce the quantization error; a head removes it —
each step trades generality for precision."*

**Check:** why not just attach a detection head to the LLM? → *you can, and some
models do — but you lose the "text is the interface" property (tools, reasoning,
agents all expect text), and you re-invoke the two-tower problem the connector
existed to solve. Frontier models are circling this trade-off.*

## 8.8 The generation side: discrete vs continuous image tokens

- **VQ-AR:** image → VQGAN codebook → discrete tokens → one AR transformer for
  text+image (Coca, Emu3). Elegant, unified; but the codebook is lossy and AR over
  ~1–4k image tokens is slow + error-cascade-y.
- **Diffusion head:** image pixels/latents from a diffusion process — parallel,
  high quality, but a second objective (no shared next-token loss).
- **Hybrid (Transfusion):** AR loss for text tokens, diffusion loss for image
  patch tokens, *one* transformer. Best of both: unified context, parallel image
  synthesis. This is the current frontier shape for "one model, both directions."

**Check:** why is a diffusion objective a good fit for image *output* specifically?
→ *images are dense, regular, continuous, fixed-size: parallel denoising fits the
structure; a token stream imposes an arbitrary raster order and sequential latency
on something with no intrinsic order.*

## 8.9 Unified understanding+generation (Janus, Show-o, Transfusion family)

One model that can *describe* an image and *make* one. Two design camps:
- **Shared backbone, split output heads** (Janus: separate AR text decoder and
  diffusion image head on a shared encoder) — clean, practical.
- **Truly unified tokens** (Transfusion) — one sequence, two losses.
The open problem: the two capabilities *fight* (generation pulls features toward
synthesis; understanding needs discriminative features) — mixture-of-experts
routing and careful data mixing are the current mitigations.

**Check:** why not just one AR model for both (VQ tokens both ways)? → *quality:
VQ-AR images still trail diffusion at the same compute; and inference cost:
thousands of sequential image tokens. Splitting the output keeps each side's best
tool.*

## 8.10 On-device and small VLMs

7B-class (or smaller) VLMs with strong encoders are shipping on phones/edge
(Phi-3.5-vision, Llama-3.2-vision, MiniCPM-V, Gemma-3v). The recipe is exactly
Ch 5–9 applied at the margin: big-ish ViT, no heavy pooling, GQA, quantized
weights (int4), pruned vision tokens, short context. For AV: on-vehicle 3–7B VLMs
are the realistic "second brain" (Ch 10) — cloud gets the 70B+ frontier models.

**Check:** what's the *first* thing a 7B on-device VLM gives up vs a frontier
70B+? → *reasoning depth (fewer layers, smaller FFN) — perception quality survives
surprisingly well because it lives in the ViT + early layers.*

## 8.11 What to watch (trends list, time-invariant)

1. **Native multimodal everywhere** — encoder/connector stacks fading from frontier.
2. **Reasoning with tools** — VLMs that crop, OCR, query, and *verify themselves*.
3. **Token diets industrialize** — adaptive, per-task budgets become standard serving knobs.
4. **Unified models** — understanding + generation in one backbone (Transfusion-style).
5. **Video as first-class** — mRoPE-style 4D positions, hour-scale context.
6. **On-device frontier** — the 7B-class keeps eating edge use-cases.
7. **Agents as the product** — the VLM is the brain; tools are the body (GUI, sim, car).

## 8.12 ❓ Interview

**Q: Where is VLM research heading? (the 60-second answer)**
— Native multimodality (no bolt-on encoders); reasoning with tools (R1-style RL +
crop/OCR loops); token efficiency as a first-class serving feature; unification of
understanding and generation; video/long-context as the new default input. And for
applications: agents and domain systems (driving included) that *use* all of it.

**Q: Would you still build a LLaVA-style stack today? Why/why not?**
— For open-weights control, fine-tuning on your data, and cost: yes — it's still the
best cost/quality/controllability triangle (Qwen2-VL is LLaVA's descendant). For
frontier quality: use the native APIs. Say both, with the trade.

**Q: Explain the token-efficiency trade-off in one design decision you'd make.**
— Example: "I'd cap hi-res tiling at 2k tokens, use adaptive grid by image entropy,
and enable attention-based pruning (FastV-style) at ~layer ℓ keeping 40% — measured
on *my* eval, because the right k is task-dependent." (Numbers + why + "measure it" =
senior answer.)

**Q: Native vs bolt-on, in your product context (AV)?**
— Bolt-on (family B) wins: we have a proven perception backbone, a licensed/frozen
  LLM to protect, and we fine-tune cheaply on fleet data; native's benefits
  (interleaved chains) matter less than our control-and-cost constraints. The
  frontier will converge anyway — keep the architecture swappable.
