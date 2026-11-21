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
v = W @ feat + b            # feat: [P, D_v]  →  v: [P, D]
```

LLaVA (v1) used exactly this, with a scale ×100 on `v` (image-token embeddings are
much larger in norm than text embeddings; the scale calibrates them into the same
regime). One matrix. It works — which is the point: **most of the job is done by
the ViT and the LLM; the bridge is thin.**

**Check:** why the ×100 in LLaVA? → *embedding-norm mismatch: raw projected image
vectors are small vs text embeddings, so attention under-weights them; the scale
equalizes their "loudness."*

## 5.3 Option 2 — MLP projector (LLaVA-1.5, most open models)

```python
v = W2( GELU( W1 @ feat + b1 ) ) + b2      # [P, D_v] → [P, D], D_ff ~ 2D
```

A nonlinearity in the bridge. Marginally better alignment, still token-preserving
(`P` in, `P` out). This is the workhorse: cheap to train, no information loss in
the projection, no bottleneck.

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
- **A training scaffold:** BLIP-2's stage 1 trains *only* the Q-Former (ViT and LLM
  frozen) on image-text matching/grounding; stage 2 fine-tunes LLM lightly. The
  queries are the *alignment surface*.

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

## 5.7 Option 5 — spatial pooling (LLaVA-NeXT's 2×2)

Keep all tokens through the ViT, then mean-pool each 2×2 block of the feature
grid into one token:

```python
# feat: [24*24, D_v] from 336px
grid = feat.view(24, 24, D_v)
pooled = grid.view(12, 2, 12, 2, D_v).mean(dim=(1, 3))   # [144, D_v]
```

576 → 144 tokens for the same image. LLaVA-NeXT does this *plus* a full-image
"global" token set, so the LLM sees a coarse overview and a pooled detail grid.
Pooling is the simplest token-budget lever: it's lossy exactly where you're
downsampling.

**Check:** 2×2 pooling — what dies first? → *thin structure: edges of text, thin
gauge, fine texture. If your task is OCR-ish, pool less (or not at all).*

## 5.8 The no-pooling camp (Qwen2-VL)

Qwen2-VL keeps **all** tokens at native (dynamic) resolution — 448×448 might be
32×32 = 1024 tokens — and pays the context cost. Its answers to "how do I afford
this?": 2D-RoPE (cheap positions), GQA (cheap KV), long-context LLM, and later
token pruning at inference (Ch 8). The philosophy: *don't lose detail up front;
spend at inference if you need to.*

**Check:** pool-upfront vs no-pooling — same ViT, same LLM: which is higher quality
on fine-detail tasks, and what's the price? → *no-pooling wins quality (more
tokens in), pays prefill FLOPs + KV memory; pooling wins cost.*

## 5.9 The token-budget decision table

| Connector | Tokens out | Detail | Context cost | Typical user |
|---|---|---|---|---|
| Q-Former | 32 (fixed) | low | tiny | small LLMs, summarization |
| 2×2 pool | P/4 | medium | low | LLaVA-NeXT |
| Linear/MLP, no pool | P (576…) | high | high | LLaVA-1.5, Qwen2-VL |
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
