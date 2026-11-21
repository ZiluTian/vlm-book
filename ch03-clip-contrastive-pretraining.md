# Chapter 3 — CLIP, Step by Step

> 🎯 **The one sentence:** CLIP aligns an image encoder and a text encoder with
> InfoNCE — a matched pair must beat every other pair *in the batch* — and that
> alignment is what lets a ViT's features get plugged into a language model later.

## 3.1 The data

~400M web image–caption pairs. No labels, no boxes, no captions you wrote.
Just `(image, "some alt text")`. The whole trick of CLIP is that **this noisy,
label-free data is enough** — if you pick the right loss.

**Check:** what's the one thing this dataset has that ImageNet doesn't? →
*open-vocabulary semantics — the model learns "text space," not a fixed label set.*

## 3.2 The two towers

```python
class ClipModel:
    def __init__(self):
        self.vit   = ViT(patch=14, depth=12, D=768)      # image tower
        self.xfmr  = TextTransformer(depth=12, D=512)    # text tower
        self.logit_scale = param(init=log(1/0.07))       # learned temperature

    def img_vec(self, x):   # x: [B, 3, 224, 224]
        return l2_norm(self.vit(x).pool)                 # [B, D]

    def txt_vec(self, t):   # t: [B, L] token ids
        return l2_norm(self.xfmr(t).pool)               # [B, D]
```

Note: the towers **never exchange activations**. They meet exactly once, at the
similarity matrix. That decoupling is what makes inference cheap (cache either side).

## 3.3 Step 1 — build the similarity matrix

```python
def similarities(img, txt):
    # img, txt: [B, D], both l2-normalized
    return img @ txt.T / exp(self.logit_scale)      # [B, B]
```

`sim[i, j]` = how well image *i* matches text *j*. Diagonal = the true pairs.
Everything off-diagonal in the batch = *free* negatives.

## 3.4 Step 2 — the loss: the batch is the negative set

```python
def clip_loss(sim):
    labels = arange(B)                    # row i's positive is column i
    loss_i2t = cross_entropy(sim, labels)      # softmax over columns (text side)
    loss_t2i = cross_entropy(sim.T, labels)    # softmax over rows    (image side)
    return (loss_i2t + loss_t2i) / 2
```

That's InfoNCE with in-batch negatives, written out. Two soft-softmax directions,
symmetric. **Check:** why two directions? → *the pairings are asymmetric in practice
(a photo often describes many captions; a caption matches few photos); both
directions constrain the geometry.*

## 3.5 The temperature: the knob that makes web data usable

Softmax sees `cos / τ`. With τ ≈ 0.07 the scale is ~14× — the model must be
*confident* that the true caption beats 32k near-duplicate web captions. Without
sharp temperature, random web noise flattens the softmax and gradients vanish.

**Check:** temperature is learned, not fixed. What would happen if it learned to
*grow* without bound? → *softmax → uniform, loss → log(B), no signal. The data
implicitly caps it: a sharper τ only helps if the model can actually tell the pair
apart.*

## 3.6 Why batch size is *the* hyperparameter

Batch B gives B−1 negatives per example. B = 32k means 32k negatives — impossible
to mine by hand, free from the dataloader. Consequences:
- CLIP training is **batch-obsessed**: bigger batch → better negatives → better model.
- Each GPU sees a *slice* of the batch; the loss needs an **all-gather of
  embeddings** across GPUs to see all negatives (that's the "global in-batch" trick).
- Bigger batches let you push temperature sharper (§3.5) — the two co-move.

**Check:** 8 GPUs, 4096 batch each — is that 8×4096 negatives per example, or 4095?
→ *8×4096 (with all-gather); only 4095 if you skip the gather (and train worse).*

## 3.7 Where the softmax breaks (the setup for SigLIP)

Push B toward 10⁶:
- `sim` is O(B²) — memory blows up,
- the softmax couples *every* pair to every other pair — one bad pair distorts all,
- `cos/τ` overflows fp precision → softmax goes one-hot → **gradients die**.

So "just use a bigger batch" hits a wall. The wall is in the *softmax*, not the data.

**Check:** what property of softmax causes the coupling? → *its denominator is a
sum over ALL pairs — each example's logit depends on every other example's logit.*

## 3.8 SigLIP: replace the coupling with a sigmoid

Drop the softmax; make each pair an independent binary "match / not match":

```python
def siglip_loss(img, txt, s=48.0, b=-10.0):
    z = img @ txt.T * s + b                  # [B, B]
    d = arange(B)
    pos = -log_sigmoid(z[d, d]).mean()       # push true pairs  -> +inf
    off = z.clone(); off[d, d] = 0
    neg = -log_sigmoid(-off).sum() / off.numel()   # push other pairs -> -inf
    return pos + neg
```

Same purpose (align two towers), different math (sigmoid, per-pair, no coupling).
Results: robust at huge/global batches, no all-gather of logits, and SigLIP-L at
768 px beats CLIP at high resolution — which is why high-res VLMs increasingly
start from SigLIP encoders.

**Check:** one-line CLIP vs SigLIP for the interview. → *"CLIP is softmax contrastive
— in-batch negatives, batch-coupled. SigLIP is sigmoid contrastive — independent
per-pair objectives, scales to giant batches, better at high res."*

## 3.9 Zero-shot classification (the famous result)

CLIP never saw class labels. To classify an image into C classes:

```python
def zero_shot(img, class_names, templates):
    x = l2_norm(vit_pool(img))                          # [D]
    cls_embs = []
    for c in class_names:
        t = [t.format(c) for t in templates]           # "a photo of a {c}", "an image of {c}", ...
        e = mean([l2_norm(txt_embed(t)) for t in t])
        cls_embs.append(l2_norm(e))
    logits = x @ stack(cls_embs).T * tau_val           # [C]
    return argmax(softmax(logits))
```

The model "knows" class `c` only because English text describing `c` exists.
Swap the class list → new task, no retraining.

**Check:** where does the class prior live? → *in the text tower's embeddings —
change the templates, change the classifier.*

## 3.10 Two details that separate the knowers

1. **Prompt ensembles:** average embeddings over ~10–200 template variants
   ("a photo of a {c}", "a blurry image of {c}", ...) → +3–5 points. Templates are
   a (cheap) hyperparameter, not a formality.
2. **Temperature calibration:** the published τ (0.07) is tuned on a web val set;
   on *your* domain, refit τ on a small val split. Often worth a point, costs nothing.

**Check:** a domain where both matter more than on ImageNet? → *anything with
idiosyncratic language: medical, satellite, dashcam ("ODD violation" was never
paired with dashcam frames on the web).*

## 3.11 CLIP's four blind spots (each motivates a later chapter)

1. **Pooled = blind to "where."** One vector per image → no grounding.
   → dense encoders (DINOv2, SAM) + vision *tokens* (Ch 4, Ch 6).
2. **Language priors leak.** Word co-occurrence beats pixels; style and object
   shortcuts. → generative VLMs that *reason* token-by-token (Ch 6+).
3. **Weak compositionality.** Multi-attribute scenes ("red car left of the bike")
   are hard for a single similarity score. → sequence modeling.
4. **Fixed 224 px.** Road signs and plates are sub-pixel. → dynamic resolution,
   tiling (Ch 5 §5.5, Ch 6 §6.6).

Keep this list — it's the "why does the stack exist" cheat sheet.

## 3.12 ❓ Interview

**Q: Why in-batch negatives instead of a mined memory bank?**
— In-batch: free, always fresh, scales with data parallelism; no momentum-updated
  embeddings to staleness. Memory banks add machinery (EMA updates, balancing) for
  marginal gain at CLIP scale.

**Q: How does temperature interact with batch size?**
— Effective sharpness scales with `s·B`: more negatives lets you afford a sharper
  temperature. Scale B without s and the softmax flattens; scale s without B and
  you overfit to in-batch noise. They're tuned together — and this tension is
  exactly what SigLIP dissolves.

**Q: Why L2-normalize the embeddings?**
— Similarity becomes pure cosine (geometry, not scale); logits stay bounded so the
  temperature has a stable meaning; the contrastive softmax stays numerically sane.

**Q: Zero-shot CLIP is bad on driving data. Diagnose.**
— (1) Domain gap: LAION web ≠ dashcam; (2) resolution: 224 px crushes signs;
  (3) vocabulary gap: "cut-in", "ODD" never co-occurred with such images on the web.
  Fixes: domain-adapt the encoder on in-domain pairs, use a high-res SigLIP ViT,
  or keep CLIP for coarse semantics and add a dense encoder (Ch 4).

**Q: In one sentence, CLIP vs LLaVA?**
— CLIP aligns two encoders into a shared space (no generation); LLaVA projects a
  CLIP-style ViT's *patch features* into a frozen LLM and generates text. LLaVA
  literally stands on CLIP's encoder — Ch 4 shows the handoff.
