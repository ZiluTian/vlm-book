# Chapter 5 — The Connector: Where Vision Meets Language

> 🎯 **The one sentence:** the connector maps P image features into the LLM's
> embedding space at a chosen token budget — linear/MLP keeps them all, Q-Former
> compresses them to a fixed few, pooling trades detail for context room.
> It's the cheapest component and the most consequential design choice.

## 5.1 The mismatch, stated plainly

After the ViT you have `P` vectors in *its* space, width `D_v` (e.g. 1024).
The LLM expects tokens in *its* space, width `D` (e.g. 3584), and it has a limited
context. Three sub-problems in one:

1. **Space mismatch** — `D_v ≠ D`, and the two spaces have no shared geometry yet.
2. **Budget** — `P` may be 576–4096; the LLM's context is expensive (KV memory!).
3. **Timing** — the ViT and the LLM were trained separately; something has to learn
   the translation *without breaking the LLM*.

Every connector design is an answer to these three. That's the whole chapter.

**Check:** which sub-problem makes the connector trainable-small while the LLM
stays frozen? → *#3 — the bridge is the only newly-trained part (usually).*

## 5.2 Option 1 — one linear layer

```python
v = feat @ W.T + b          # feat: [P, D_v], W: [D, D_v] → v: [P, D]
```

Original LLaVA used a learned linear projection. It works because **most of the
job is done by the ViT and the LLM; the bridge is thin.** A fixed multiplier is
not part of the recipe. If you suspect a scale problem, compare projected feature
and text embedding norms and test whether it actually changes grounding quality.

**Check:** does matching the LLM's width make the projected tokens meaningful to
it? → *no; width makes concatenation legal. Training makes the vectors useful.*

## 5.3 Option 2 — MLP projector (LLaVA-1.5, most open models)

```python
v = gelu(feat @ W1.T + b1) @ W2.T + b2   # [P, D_v] → [P, D_hidden] → [P, D]
```

A nonlinearity in the bridge. Marginally better alignment, still token-preserving
(`P` in, `P` out). This avoids a spatial token-count bottleneck, but preserving
the count does not guarantee that the projection preserves every feature.

**Check:** P in → P out. What does that imply for the token budget? → *nothing is
reclaimed; budget management must happen upstream (resolution, pooling) or
downstream (pruning — Ch 8).*

## 5.4 Option 3 — Q-Former: learned queries do the projection (BLIP-2)

Instead of projecting *every* image feature, learn a fixed set of `Q` query vectors
(= 32) that **attend to** the image features and summarize them:

```python
class QFormer:
    def __init__(self, Q=32, D=768):
        self.queries = param([Q, D])        # learned, image-agnostic

    def forward(self, feat):                # feat: [P, D_v] from ViT
        q = self.queries
        for blk in cross_attn_blocks:       # ~2 layers
            q = self_attn(q) + cross_attn(q, K=feat, V=feat)   # queries read the image
            q = q + ffn(q)
        return q                            # [32, D]  ← always 32 tokens
```

This is BEVFormer in disguise: a fixed set of *queries* distilled from a variable
feature map. (🔗 BEV bridge: you literally built this — BEV queries attending to
camera features, except here the "BEV grid" is a flat bag of 32.)

## 5.5 What Q-Former buys, what it costs

**Buys:**
- **Fixed, tiny token count** (32) regardless of image size → LLM context is cheap
  and predictable. Great when the LLM is small or context is precious.
- **A training scaffold:** BLIP-2 trains the Q-Former against a frozen vision
  encoder with image-text contrastive, matching, and generation objectives. In
  stage 2 it connects the queries to a frozen LLM and trains the bridge with a
  language generation objective.

**Costs:**
- **Bottleneck:** 576 → 32 is a 18× information squeeze. Fine detail (text in
  images, small objects, counting) gets lost — the queries can't keep it.
- The fixed count ignores how much detail an image actually needs.

**Check:** for "what's the license plate number," which connector wins? →
*token-preserving (linear/MLP at high res) — 32 queries cannot reliably carry
a 7-character string. For "describe the scene," Q-Former's summary is plenty.*

## 5.6 Option 4 — Perceiver resampler (Flamingo)

Same query-attention idea as Q-Former, different layout: the resampler is a
Perceiver (cross-attention over a large feature set → fixed latent size), and the
LLM itself is frozen with **cross-attention layers injected** between its blocks
(visual tokens attend into the LLM at multiple depths, not just at the front).

One-liner distinction: *Q-Former compresses once, up front; Flamingo injects
compressed visual latents repeatedly through the LLM's depth.* Both answer "budget"
with "fewer, deeper-read tokens."

**Check:** why inject at multiple depths instead of once at the input? → *early
LLM layers are doing token-local/linguistic work; mid-late layers do global
composition — feeding vision at several depths gives each stage a fresh read of the
image latents.*

## 5.7 Option 5 — spatial pooling

Keep all tokens through the ViT, then mean-pool each 2×2 block of the feature
grid into one token:

```python
# feat: [24*24, D_v] from 336px
grid = feat.view(24, 24, D_v)
pooled = grid.view(12, 2, 12, 2, D_v).mean(dim=(1, 3))   # [144, D_v]
```

576 → 144 tokens for the same image. This is a generic spatial downsampler, not
the original LLaVA-NeXT any-resolution recipe (see [Chapter 6](ch06-llava-the-blueprint.md)).
It cuts decoder cost but not the cost of the already-computed ViT features.

**Check:** 2×2 pooling — what dies first? → *thin structure: edges of text, thin
gauge, fine texture. If your task is OCR-ish, pool less (or not at all).*

## 5.8 Learned spatial merging (Qwen2-VL)

Qwen2-VL uses dynamic resolution but still merges each spatial 2×2 group with
a learned MLP. At 448×448 with 14-pixel patches, that is a 32×32 patch grid and
256 visual embeddings sent to the decoder, excluding boundary tokens. Unlike
mean pooling, the merger sees features in a fixed order within each group, so
it can learn that a stroke in the top-left patch differs from one in the
bottom-right. It still compresses those four positions into one decoder token.

**Check:** pool-upfront vs no-pooling — same ViT, same LLM: which is higher quality
on fine-detail tasks, and what's the price? → *no-pooling wins quality (more
tokens in), pays prefill FLOPs + KV memory; pooling wins cost.*

## 5.9 The token-budget decision table

| Connector | Tokens out | Detail | Context cost | Typical user |
|---|---|---|---|---|
| Q-Former | 32 (fixed) | low | tiny | small LLMs, summarization |
| 2×2 pool / learned merge | P/4 | task-dependent | low | Qwen2-VL (learned merge) |
| Linear/MLP, no pool | P (576…) | preserves spatial slots | high | LLaVA-1.5 |
| Dynamic tiling (Ch 6) | P·(1+hw) | very high | very high | InternVL, LLaVA-NeXT hi-res |

Interview framing: *"The connector is where you decide what the image is allowed to
cost. Every other part of the system just pays that invoice."*

## 5.10 Why the connector gets its own training stage

If you train ViT + connector + LLM together from scratch, the LLM's embedding space
drifts while the bridge chases it — unstable. So the standard recipe (BLIP-2, LLaVA,
Qwen-VL): **stage 1 = train the connector alone** (ViT frozen, LLM frozen), aligning
the two existing spaces; **stage 2 = unfreeze and tune together** on instruction data.
Ch 6 shows the full loop.

**Check:** what breaks if you skip stage 1 and go straight to joint SFT? → *the
LLM gets gradient noise from a misaligned image stream → capability regressions
(catastrophic forgetting of pure-text behavior) and a much less stable start.*

### Worked example: why did OCR get worse after compression?

**Why:** a connector that scores well on scene summaries can still drop the
single character that makes an OCR answer correct. An average score can hide it.

**What:** suppose the ViT returns 576 patches. Compare a projector (576 decoder
tokens), 2×2 pooling (144), and a 32-query resampler. At 128 KiB of KV per token
([Chapter 9](ch09-inference-and-serving.md)), each image needs **72, 18, or 4 MiB**
respectively. These are decoder KV costs, not total GPU memory. The ViT did the
same work in all three cases; the resampler adds computation of its own.

**How:** keep the encoder, image size, decoder, data and training budget fixed;
train each bridge before comparing. Report exact-match OCR by character size,
scene QA accuracy, and time to first token. Include images where changing a
single digit changes the answer. If even the unpooled model fails, probe a
higher-resolution crop with the same model as a diagnostic for upstream detail
loss. If only the compressed model fails on the same features, test whether a
larger token budget restores accuracy. Neither comparison alone proves the
cause, but together they narrow down where to spend the next experiment.

## 5.11 ❓ Interview

**Q: Why not just resize the image smaller and skip the connector's budget job?**
— You can (lower native res), but resolution is a *global* knob: it also kills the
coarse semantics. The connector gives you *selective* compression (pool 2×2 in the
grid, keep a global token, or keep all and prune at inference) without lowering the
input.

**Q: Q-Former vs linear projector — how would you choose?**
— Task first: need fine detail / OCR / grounding → token-preserving; need cheap
context on a small LLM → Q-Former. Second: training compute — Q-Former's stage-1
alignment is a nice scaffold if the LLM must stay frozen (licensing, cost). Third:
measure — a 2×2-pool A/B against no-pool on *your* eval is cheaper than any theory.

**Q: BLIP-2's two stages — why freeze the LLM in stage 1?**
— Stage 1's job is purely to align spaces; letting a 175B LLM move while a 32-query
bridge learns is a waste of compute and risks destroying text capability. Freeze
what's already good; train what's new.

**Q: Where does a connector fail that scaling the LLM can't fix?**
— Information destroyed upstream: 32 queries, heavy pooling, low native resolution.
The LLM can't reconstruct detail its tokens never carried — the connector sets the
ceiling, the LLM only sets how well it's used.

**Q: What would you change about a LLaVA-style stack, given 2025-era tricks?**
— Drop heavy pooling (or make it adaptive), use 2D/3D RoPE for layout, prune
low-attention vision tokens at inference (§8.3), and consider a second dense
encoder branch (DINOv2/SAM) for spatial precision — without moving the LLM at all.
