# Chapter 3 — CLIP, Step by Step

> 🎯 **The one sentence:** CLIP aligns an image encoder and a text encoder with
> InfoNCE — a matched pair must beat every other pair *in the batch* — and that
> alignment is what lets a ViT's features get plugged into a language model later.

## 3.1 The data

~400M web image–caption pairs. No labels, no boxes, no captions you wrote.
Just `(image, "some alt text")`. The whole trick of CLIP is that **this noisy,
label-free data is enough** — if you pick the right loss.

**Running example:** take a batch of three pairs, in this order: a dog photo
with “A dog on grass,” a train photo with “A train at a station,” and a coffee
photo with “A cup of coffee.” The loader supplies those three matches. It
doesn't supply class IDs or any explicit wrong answers.

**Check:** what's the one thing this dataset has that ImageNet doesn't? →
*open-vocabulary semantics — the model learns "text space," not a fixed label set.*

## 3.2 The two towers

**What has to match?** The widths of the **projected embeddings**, not the
widths of the image and text encoders. In this example, the image encoder
outputs a pooled vector of width `768` and the text encoder one of width `512`.
Each tower has its own learned projection into the same `D`-dimensional space:

```text
image: [B, 768] ── image projection ──► [B, D] ── normalize ──► I
text:  [B, 512] ── text projection  ──► [B, D] ── normalize ──► T
cosine similarities: I @ T.T = [B, D] @ [D, B] → [B, B]
```

**Why:** a dot product multiplies corresponding coordinates, so `[768] · [512]`
is undefined. Matching widths makes the calculation possible; contrastive
training makes the scores meaningful. Even if the raw widths happen to match,
the two encoders don't automatically represent information in the same space.

```python
class ClipModel:
    def __init__(self, D_shared):
        self.vit   = ViT(patch=14, depth=12, D=768)      # D here is image hidden width
        self.xfmr  = TextTransformer(depth=12, D=512)   # D here is text hidden width
        self.img_proj = Linear(768, D_shared)
        self.txt_proj = Linear(512, D_shared)
        self.logit_scale = param(init=log(1/0.07))       # learned temperature

    def img_vec(self, x):   # x: [B, 3, 224, 224]
        return l2_norm(self.img_proj(self.vit(x).pool))  # [B, D_shared]

    def txt_vec(self, t):   # t: [B, L] token ids
        return l2_norm(self.txt_proj(self.xfmr(t).pool)) # [B, D_shared]
```

**How to read the numbers:** `depth=12` means 12 Transformer blocks in each
tower, not 12 tokens or 12 coordinates. The blocks don't have to match in
number across towers. The `D` argument *inside* each encoder is its hidden
width: each image token carries 768 numbers, and each text token carries 512.
`D_shared` is a separate setting used only after pooling, called `D` in the
similarity formula. These are illustrative architecture settings, not a
specification for every CLIP checkpoint.

| Stage | Dog photo | “A dog on grass” |
|---|---|---|
| Input tokens | `patch=14` divides a 224×224 image into a 16×16 grid: 256 patch tokens, plus a class token in this ViT | Tokenizer produces `L` positions, including any special tokens and padding |
| After 12 blocks | `[257, 768]`: one 768-number vector per token | `[L, 512]`: one 512-number vector per token |
| Pool one vector | `[768]`: take the final class-token vector | `[512]`: take the final end-of-text token vector in CLIP |
| Project and normalize | `Linear(768, 4)` → `[4]` | `Linear(512, 4)` → `[4]` |

The last row uses `D_shared=4` for the toy example below. Do this for all
three pairs and you get two `[3, 4]` embedding arrays. Multiplying one by the
other's transpose gives a `[3, 3]` matrix: **four coordinates go into each dot
product, but there are three images and three captions to score.**

Note: the towers **never exchange activations**. They meet exactly once, at the
similarity matrix. That decoupling is what makes inference cheap (cache either side).

**What comes out for our batch?** One vector per photo and one per caption.
The toy example uses `D = 4` *after* projection and normalization. It doesn't
assume either encoder is four-dimensional. Here are invented unit vectors
chosen to make the dot products easy:

| Pair | Image embedding | Caption embedding |
|---|---|---|
| Dog | `[1, 0, 0, 0]` | `[0.8, 0, 0.2, √0.32]` |
| Train | `[0, 1, 0, 0]` | `[0.2, 0.7, 0, √0.47]` |
| Coffee | `[0, 0, 1, 0]` | `[0, 0.1, 0.9, √0.18]` |

Both embedding arrays have shape `[3, 4]`. The fourth caption coordinate makes
each caption vector unit length; the image vectors happen to be zero there.

## 3.3 Step 1 — build the similarity matrix

```python
def similarities(img, txt, logit_scale):
    # img, txt: [B, D], both l2-normalized
    return exp(logit_scale) * (img @ txt.T)         # [B, B]; scale = 1 / τ
```

`sim[i, j]` = how well image *i* matches text *j*. The diagonal contains the
original image–caption pairs; the off-diagonal entries are the cross-pairs.

**How to build it:** multiply `[3, 4] @ [4, 3]` to score all nine pairs. For
example, the dog image vector `[1, 0, 0, 0]` selects the first coordinate of
each caption vector, so its dot products are `[0.8, 0.2, 0.0]`. Because the
vectors have unit length, these dot products are also cosine similarities.
Use a toy scale `exp(logit_scale) = 5` (temperature `τ = 0.2`) to turn the
cosines into the logits that go into the loss:

| Image ↓ / Caption → | Dog caption | Train caption | Coffee caption |
|---|---:|---:|---:|
| Dog photo | 0.80 × 5 = **4.0** | 0.20 × 5 = 1.0 | 0.00 × 5 = 0.0 |
| Train photo | 0.00 × 5 = 0.0 | 0.70 × 5 = **3.5** | 0.10 × 5 = 0.5 |
| Coffee photo | 0.20 × 5 = 1.0 | 0.00 × 5 = 0.0 | 0.90 × 5 = **4.5** |

The bold cells are the designated matches **because of the input pairing**, not
because they're the largest scores. Early in training, the largest score can
land in the wrong cell; that's what the loss is there to fix.

## 3.4 Step 2 — the loss: the batch is the negative set

**What:** for each image, CLIP uses the other captions in the same training
batch as wrong-answer choices. A caption is positive for its own image and
treated as negative for the other images. For a batch of three pairs:

| Image | “A dog on grass” | “A train at a station” | “A cup of coffee” |
|---|---|---|---|
| Dog photo | **Positive** | Negative | Negative |
| Train photo | Negative | **Positive** | Negative |
| Coffee photo | Negative | Negative | **Positive** |

**Why:** you don't have to collect wrong captions separately. Each caption you
already have becomes a distractor for the other images. With `B` pairs, each
image gets **1 positive and `B − 1` negatives**.

**How:** for the dog photo, CLIP scores all three captions. Cross-entropy over
that row rewards assigning a higher score to “A dog on grass” than to the other
two captions. CLIP also runs the task in reverse: given “A dog on grass,” pick
the dog photo over the train and coffee photos. That means scoring down the
column, which the code does by transposing the matrix.

```python
def clip_loss(sim):
    B = sim.shape[0]
    labels = arange(B)                    # row i's positive is column i
    loss_i2t = cross_entropy(sim, labels)      # softmax over columns (text side)
    loss_t2i = cross_entropy(sim.T, labels)    # softmax over rows    (image side)
    return (loss_i2t + loss_t2i) / 2
```

### Follow the numbers through the loss

1. **Image → text:** read the dog photo's **row** `[4, 1, 0]`. Softmax gives
   `[0.936, 0.047, 0.017]`. The dog caption is the target, so the loss is
   `−log(exp(4) / (exp(4) + exp(1) + exp(0))) ≈ 0.066`.
2. **Text → image:** read the dog caption's **column** `[4, 0, 1]`. The strongest
   distractor is now the coffee *photo*, not the train *caption*. Here the dog
   still has probability `0.936` because the competing scores just swap places.
3. **Average:** the three row losses are about `[0.066, 0.077, 0.040]` and the
   three column losses are about `[0.066, 0.106, 0.029]`. The two means are
   `0.0611` and `0.0671`, so the final batch loss is **`0.0641`**.

The catch: **negative means assumed not to match, not verified wrong.** If the
batch contains two dog photos, “A dog on grass” might describe both, even though
standard CLIP treats it as positive for only one. The other match is a *false
negative*; see the worked example in §3.8 for what it does to the gradient.

**Check:** why run the task in both directions? → *choosing the right caption for
an image doesn't guarantee that the caption picks the right image. The two
losses train both retrieval directions.*

## 3.5 The temperature: the knob that makes web data usable

Softmax sees `cos / τ`. CLIP initializes `τ` to `0.07`, giving a scale of ~14×. Lower temperature
concentrates the probability mass, so the highest-scoring negatives matter most.
That includes mislabeled negatives; a sharper softmax does not fix noisy labels.

**Same dog photo, different temperature:** halving `τ` from `0.2` to `0.1`
doubles the row logits from `[4, 1, 0]` to `[8, 2, 0]`. The dog caption's
probability rises from `93.6%` to `99.7%`, but the ranking hasn't changed.
If the wrong caption ranked first, lowering `τ` would make that mistake more
confident, not correct it.

**Check:** what happens as `τ` grows without bound? → *the logits approach zero,
softmax approaches uniform, and the loss approaches `log(B)`. The gradient to
the cosine scores shrinks because it is multiplied by `1/τ`.*

## 3.6 Why batch size is *the* hyperparameter

Our three-pair batch gives the dog photo two negative captions. Add a fourth
pair (say, a bicycle and its caption) and the matrix grows from `3 × 3` to
`4 × 4`. The dog now has three negatives, even though we didn't relabel it.
In general, `B` pairs give `B−1` negatives per anchor. Consequences:
- Bigger batches give more chances to see useful hard negatives, but also more
  chances to see false negatives.
- Each GPU sees a *slice* of the batch; the loss needs an **all-gather of
  embeddings** across GPUs to see all negatives (that's the "global in-batch" trick).
- The added negatives affect the softmax according to their scores. A single
  near-match can matter more than many unrelated captions.

**Check:** 8 GPUs, 4096 batch each — is that 8×4096 negatives per example, or 4095?
→ *32,767 with all-gather (the positive takes one slot); 4,095 without it.*

## 3.7 Where the softmax breaks (the setup for SigLIP)

Push B toward 10⁶:
- `sim` is O(B²) — memory blows up,
- in each direction, every candidate for a given anchor shares a denominator;
  a high-scoring false negative steals mass from the positive.

Stable log-sum-exp avoids exponent overflow, and a tiled implementation need not
store the full matrix. Neither removes the cost of scoring all pairs. SigLIP
changes the normalization, not the number of pairs in a full all-pairs loss.

**Check:** what property of softmax causes the coupling? → *its denominator is a
sum over candidates for the same anchor — their probabilities are coupled even
though their raw logits are computed independently.*

## 3.8 SigLIP: replace the coupling with a sigmoid

Drop the softmax; make each pair an independent binary "match / not match":

```python
def siglip_loss(img, txt, logit_scale, bias):
    B = len(img)
    z = exp(logit_scale) * (img @ txt.T) + bias  # [B, B]
    signs = 2 * eye(B, device=z.device) - 1     # +1 diagonal, -1 off-diagonal
    return softplus(-signs * z).sum() / B       # sum pairs, normalize by anchors
```

**Same scores, different loss:** for illustration, use scale `5` again and a
bias of `−2`. The dog photo's logits become `[2, −1, −2]`, and the pair labels
are `[+1, −1, −1]`. The sigmoid match scores are `[0.881, 0.269, 0.119]`.
They don't have to add to one: each caption gets its own match score. Adding
the bicycle caption introduces a fourth loss term but doesn't renormalize
these three scores. CLIP's row softmax would change their probabilities.

Same purpose (align two towers), different math (sigmoid, per-pair, no softmax
denominator). You can sum losses over blocks of exchanged embeddings without
materializing global logits. If you want global negatives, those embeddings still
have to reach the devices that score their pairs.

**Check:** one-line CLIP vs SigLIP for the interview. → *"CLIP is softmax contrastive
— in-batch negatives, coupled per anchor. SigLIP is sigmoid contrastive —
independent per-pair terms that can be computed block by block."*

### Worked example: the hard negative that isn't negative

**Why:** contrastive training can penalize good matches when the dataset assumes
every image has exactly one valid caption.

**What:** replace the train pair with a second dog photo captioned “A dog sitting
on grass.” Suppose the first dog photo's row is now `[4, 3, 0]`. Softmax gives
about `[0.721, 0.265, 0.013]`; subtract the target `[1, 0, 0]` to get the row
loss gradient `[-0.279, +0.265, +0.013]`. This loss pushes the designated
positive score up and the second caption's score down. If both captions describe the image,
the second update is wrong. Sharpening the distribution isn't a label fix, and
switching to SigLIP still pushes a pair apart if its sign is `-1`.

**How:** if both matches are verified, try a normalized soft target with mass `0.5` on
each caption (and zero on genuine negatives), or label both pairs `+1` for
SigLIP. For uncertain duplicates, masking those pairs is safer than assuming
equivalence: two almost-identical product images can show different model numbers.
Evaluate recall@K with **all** verified matches accepted, and keep a separate
slice for visually similar pairs that really are different. Otherwise you cannot
tell whether the fix recovered valid matches or just blurred distinctions.

**Distributed sanity test:** with two ranks and `b` examples per rank, rank 1's
local image row `i` has positive text column `b + i` if the gather is rank-ordered.
Compare the distributed loss **and both towers' gradients** against a single-device
run on the same global batch. A detached gather can preserve the forward values
while silently dropping the gradient contributions from remote anchors unless the
implementation accounts for them separately.

Sources: [CLIP](https://arxiv.org/abs/2103.00020),
[SigLIP](https://arxiv.org/abs/2303.15343).

## 3.9 Zero-shot classification (the famous result)

CLIP never saw class labels. To classify an image into C classes:

```python
def zero_shot(model, img, class_names, templates):
    x = model.img_vec(img[None])[0]                     # [D]
    cls_embs = []
    for c in class_names:
        prompts = [t.format(c) for t in templates]
        e = model.txt_vec(tokenize(prompts)).mean(dim=0)
        cls_embs.append(l2_norm(e))
    logits = exp(model.logit_scale) * (x @ stack(cls_embs).T)  # [C]
    return class_names[argmax(logits)]
```

The model "knows" class `c` only because English text describing `c` exists.
Swap the class list → new task, no retraining.

**Same matrix operation at inference:** treat the three example captions as
one prompt each for the classes `dog`, `train`, and `coffee`. A query with the
toy dog embedding gets logits `[4, 1, 0]`, so `argmax` returns `dog`. There's
no batch of reference photos and no training loss: the three columns now
represent the answer choices you supply at inference. To classify a new set
of classes, embed those prompts and replace the columns.

**Check:** where does the class prior live? → *in the text tower's embeddings —
change the templates, change the classifier.*

## 3.10 Two details that separate the knowers

1. **Prompt ensembles:** average embeddings over ~10–200 template variants
   ("a photo of a {c}", "a blurry image of {c}", ...) → +3–5 points. Templates are
   a (cheap) hyperparameter, not a formality.
2. **Temperature calibration:** if you need calibrated probabilities, fit a
   temperature on held-out target-domain data. It can improve confidence
   estimates, but a positive scale alone cannot change the top-1 prediction.

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
— More negatives add terms to the softmax denominator; lower temperature
   concentrates mass on the largest logits. The effect depends on how closely
   the new negatives score to the positive, not just on the batch size.

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
