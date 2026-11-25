# Chapter 4 — Vision Encoders: ViT, DINOv2, SAM

> 🎯 **The one sentence:** a ViT is a strided conv into "patches" followed by plain
> attention; modern VLMs pick their encoder by *what kind of features* they need —
> CLIP/SigLIP for semantics, DINOv2/SAM for dense geometry — because the LLM half
> can't recover what the encoder threw away.

## 4.1 A ViT is a CNN you didn't need

The only "vision" trick in a ViT: turn pixels into a short list of vectors, then
run attention. Your 300k-param first conv layer does exactly the same job as a
patch embedding — see §4.2. After that, no convs at all: attention is the only
"receptive field," and it's global from layer one.

**Check:** what did you give up vs a CNN? → *the inductive bias — translation
equivariance and local neighborhoods. You buy it back with more data and attention.*

## 4.2 Patchify: the whole "vision" part

```python
def vit_forward(x):
    # x: [B, 3, H, W]
    p = conv2d(x, kernel=14, stride=14)      # [B, (H/14)*(W/14), D]  == patchify
    n = p.shape[1]                           # e.g. 224/14=16 → 256; 336/14=24 → 576
    p = p + pos_embed[:n]                    # learnable or 2D-RoPE (§4.6)
    for blk in blocks:                       # N standard pre-LN attention blocks
        p = blk(p)
    return final_ln(p)                       # [B, n, D]  ← ALL n tokens returned
```

`conv2d(kernel=14, stride=14)` *is* patch embedding — you already knew this; the
ViT just keeps every patch as a token instead of pooling them down.

**Check:** 336×336 input, patch 14 → how many tokens? → *24×24 = 576. (That's the
number LLaVA lives with.)*

## 4.3 All tokens, or just one?

Two consumption styles:
- **Pool to 1 vector** (CLS/mean) → CLIP-style alignment (Ch 3).
- **Keep all n tokens** → feed them to a connector (Ch 5). VLMs use this: each
  patch token becomes one "vision token" in the LLM's sequence.

The LLM path is what makes a ViT a *VLM* encoder — and it's why **spatial detail
survives at all** (CLIP's pooled vector couldn't say "where").

## 4.4 Register tokens (small, but an interview signal)

DINOv2 adds a few learned **register** tokens: junk slots that absorb attention
artifacts (the model's "where do I dump my weirdness") so they don't pollute the
real patch tokens. Cheap, measurable win on dense feature quality. One-liner version:
*registers = scratch space in the token stream.*

## 4.5 Resolution: pixels, patches, and cost

```
224 px  → 16×16 = 256 tokens
336 px  → 24×24 = 576 tokens
448 px  → 32×32 = 1024 tokens
```

More pixels → more patches. With dense ViT attention, pairwise work grows
quadratically in patch count. The decoder pays separately, based on how many
tokens actually leave the connector ([Chapter 5](ch05-bridging-vision-to-language.md)).

**Check:** doubling resolution multiplies token count by how much? → *4. Attention
pair count goes up 16× with full attention, but projections and FFNs grow only
4×, so total encoder runtime need not grow 16×.*

### Worked example: resize, tile, or crop?

**Why:** a 14-pixel-tall digit in a 1344-pixel-wide image shrinks to about 3.5
pixels when resized to 336 pixels wide. The resized image may no longer contain
enough evidence to read it, no matter how large the decoder is.

**What:** consider a square image, 14-pixel patches, and an encoder accepting
336-pixel views. A global resize yields 576 patches. Four 672×672 tiles, each
resized to 336×336, preserve about 7 pixels of digit height and yield 2,304
patches before any global view. A 336×336 native-pixel crop around the digit
preserves all 14 pixels with 576 patches, but loses context and needs a way to
select the region. Patch size is not a hard readability threshold: a patch
embedding processes the pixels inside it, but downsampling may erase them first.

**How:** measure exact match on small text with global, tiled, and crop inputs.
Try ground-truth crops first to estimate how much correct localization could
help, then test the actual region selector and report misses and additional
latency. Include examples with text near tile boundaries; run an overlap
ablation before increasing resolution across the entire image.

## 4.6 2D positions: why "which patch is where" must be encoded

Unmasked attention without position information is permutation-equivariant: it
doesn't know which patch came first. A 1D index adds order, but the same offset
can mean different spatial relationships at different grid widths. Options:

```python
# (a) learnable 2D grid embedding  (original ViT)
pos = grid_embed[h, w]
# (b) 2D RoPE: allocate rotary channels to height and width separately
# (c) M-RoPE (Qwen2-VL decoder): allocate channels to time, height, and width
```

Rotary positions can be computed for new grid sizes without resizing a learned
table. Learned grids can also support another resolution by interpolating patch
positions (keeping any class token separate). Neither scheme guarantees good
accuracy at grid sizes never seen in training.

**Check:** why prefer explicit row and column coordinates for variable grids? →
*a vertical neighbor stays one row away whether the image grid is 16 or 32
patches wide. A flattened index alone doesn't express that relationship.*

## 4.7 DINOv2: self-distillation, in six lines

No labels. A student sees a **local crop**; an EMA **teacher** (same network,
slower-updated) sees the **global view**. Student's crop features must match the
teacher's features at the same location:

```python
def dino_step(x):
    crops = random_local_crops(x)            # student views
    s = student(crops).local_feats           # [B*K, D]
    t = teacher(global_view(x)).local_feats  # [B*K, D]  (EMA weights, no grad)
    return contrastive_pair_loss(s, t)       # cosine InfoNCE, sharpened
```

No text, no captions, no pairs of *different* images — just "my crop says the same
as your global view at this spot." 14M images (DL-14M), ~4 weeks on 64 GPUs.

**Check:** why EMA teacher instead of a second trained model? → *stability: the
teacher moves slowly, so the student has a fixed target to chase — the same reason
you'd use a momentum reference in any self-supervised setup (MoCo's trick).*

## 4.8 The loss details that make DINOv2 work

```python
def dino_pair_loss(s, t):
    s = center(s)                  # subtract running mean of student embeddings
    t = center(t)
    t = sharpen(t, tau_t=0.07)     # softmax over a *sharpened* teacher
    return mean(-log_softmax(cos_sim(s, t).T, dim=1))   # crop ↔ global alignment
```

- **Centering** both sides: stops one spatial region from dominating (collapse defense).
- **Teacher sharpening**: a soft but peaked target — hard enough to teach fine detail,
  soft enough not to hallucinate.
- **Global→local (dense) direction**: the teacher's *global* context supervises the
  student's *local* features. That asymmetry is the whole point — it's how local
  patches learn "I'm a wheel of a car."

## 4.9 Why DINOv2 features are *dense* where CLIP's are not

CLIP trained its ViT to make **one good pooled vector** (global matching).
DINOv2 trained it so that **every patch token is useful at its location**
(local matching). Same ViT family, opposite supervision — so DINOv2 excels at
segmentation/depth/detection transfers, while CLIP excels at semantic retrieval.

**Check:** which encoder would you use for an open-vocabulary *detection* head?
→ *DINOv2-style dense features (or both: CLIP for the class-embedding side,
DINOv2 for the spatial side).*

## 4.10 SAM: the segmentation encoder

SAM's encoder (a Hiera-style ViT: early downsampling, then up) was trained on
11M images / 1.1B masks. You don't use SAM for the masks in a VLM — you use its
**encoder features**, which are unusually strong for fine geometry (edges, small
objects), and SAM2 adds memory over time for video. In MLLMs, SAM2 features show up
when the task needs "where, precisely, and how does it move."

**Check:** what does SAM2's memory module correspond to in your stack? →
*temporal feature propagation — the video equivalent of your BEV temporal fusion.*

## 4.11 The encoder picker (the table to recite)

| Encoder | Features are | Best for | Cost/quirk |
|---|---|---|---|
| CLIP ViT | global semantic | alignment, zero-shot, retrieval | weak dense detail, 224–336 px |
| SigLIP ViT | global semantic, better @ high-res | high-res semantic VLMs | same family as CLIP |
| DINOv2 | dense, geometric | grounding, small objects, depth | no text alignment built in |
| SAM / SAM2 | fine geometry, (video) | precise "where," moving things | heavy; needs a text bridge for VLM use |

**Rule of thumb:** semantic question → CLIP/SigLIP family; spatial/precise question →
DINOv2/SAM family; many production VLMs run *both* (Qwen-VL pairs a CLIP-style ViT
with strong 2D position handling; InternVL uses a giant ViT tuned for high-res).

## 4.12 What modern MLLMs actually do (orientation)

- **LLaVA(-1.5):** CLIP ViT-L/14, 336 px → 576 tokens. The reference simple case.
- **Qwen2-VL / 2.5-VL:** CLIP-style ViT, **no pooling — all tokens kept**, 2D-RoPE,
  dynamic resolution (any H, W, even video t, h, w via mRoPE).
- **InternVL:** very large ViT (≈1B params), dynamic tiling of high-res images.
- **InternVL2.5+, some 2025+ models:** add a DINOv2/SAM *second* branch for dense cues.

Pattern: the encoder decides the *quality ceiling*; the connector decides the
*token budget*; the LLM decides the *reasoning*. Know which knob is which.

## 4.13 ❓ Interview

**Q: ViT vs CNN, the 20-second answer?** — A strided conv makes patch tokens; global
attention replaces local convs; you trade inductive bias for data efficiency at
scale and a uniform token interface (the same tokens an LLM can consume).

**Q: Why keep all patch tokens instead of pooling?** — Pooling is lossy and global;
the LLM needs *spatially indexed* features so it can ground words to locations.
Cost: more tokens → more prefill/KV — hence the whole token-budget literature (Ch 5, 8).

**Q: DINOv2 in one breath?** — Self-distillation: EMA teacher on global views
supervises student on local crops with centered, sharpened cosine contrastive loss;
produces dense features without labels; registers absorb artifacts.

**Q: Your driving VLM is bad at small objects (plates, cones). What do you touch,
in order?** — (1) resolution: dynamic-res/tiling or a higher native res; (2) encoder:
add/swap a dense encoder (DINOv2/SAM) branch; (3) tokens: don't pool 2×2 if you can
afford it (Ch 5 trade-off); (4) only then, data: targeted long-tail samples.

**Q: What's the role of position encoding for image tokens specifically?**
— It's where *layout* lives: 2D RoPE/mRoPE makes attention sensitive to relative
(h, w[, t]) geometry, supports variable resolutions and video, and replaces fixed
learned grids that can't generalize to new tile shapes.
