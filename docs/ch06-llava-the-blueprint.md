# Chapter 6 — LLaVA: The Blueprint, Built in Stages

> 🎯 **The one sentence:** LLaVA learns to insert projected vision features into a
> decoder's text sequence, first aligning the bridge with frozen backbones, then
> tuning the bridge and LLM to generate assistant replies conditioned on the image.

## 6.1 The three components, and who moves

```
[ViT-L, frozen] ──► [projector, TRAINED] ──► [LLaMA-7B, frozen in stage 1]
```

Both original LLaVA and LLaVA-1.5 use alignment followed by instruction tuning.
The original uses a linear projector; LLaVA-1.5 uses an MLP and a 336-pixel encoder.

**Check:** which component holds the most parameters? → *the LLM. In stage 1
only the much smaller projector trains, although both backbones still run.*

## 6.2 Step 0 — the text-only forward pass (recap)

```python
h = embed(ids)              # [T, D]
h = gpt(h)                  # [T, D]
logits = lm_head(h)         # [T, V]
loss = ce(logits[:-1], ids[1:])
```

You already know this (Ch 2). The VLM adds exactly one thing to it: **more input
tokens that aren't text.** Everything below is about inserting those cleanly.

## 6.3 Step 1 — encode the image

```python
feat = vit(image)           # image: [3, 336, 336] → feat: [576, 1024]
```

576 patch tokens, width 1024, in the ViT's space. (Ch 4: why 336, why all tokens.)

## 6.4 Step 2 — project into LLM space

```python
v = projector(feat)         # [576, 1024] → [576, D]   (MLP in LLaVA-1.5)
```

Now `v` is an ordered sequence of 576 vectors with the same width as text
embeddings. The projector learns to make those vectors useful to the LLM.

## 6.5 Step 3 — assemble the sequence

```python
e_q = embed(question_ids)           # [Lq, D]
seq = cat([v, e_q], dim=0)          # [576 + Lq, D]
```

In this simplified layout the image goes first, and RoPE uses 1D positions
throughout the sequence. In a real chat, insert the features where the template
places the image placeholder; some architectures also emit explicit boundaries.
Qwen2-VL uses separate position axes for time, height, and width.

**Check:** could the image go *after* the question? → *yes, and some templates do;
causal attention means the answer (generated after both) can attend to either.
Front placement is convention + cache-friendly (image KV built once, reused).*

## 6.6 Step 4 — the full forward with the loss

```python
def llava_train(image, question_ids, answer_ids):
    v = projector(vit(image))                     # [P, D]
    h = cat([v, embed(question_ids), embed(answer_ids)])   # [P + Lq + La, D]
    logits = gpt(h)                               # [P+Lq+La, V]

    # targets: what should each position predict?
    tgt = full(P + Lq + La, -100)                 # -100 = "ignore"
    tgt[P + Lq : P + Lq + La] = answer_ids        # only the answer is a target
    loss = ce(logits[:-1], tgt[1:])               # -100 slots are masked out
    return loss
```

Read that `tgt` line slowly — **it is the entire SFT technique** (§2.11). Image
tokens and question tokens are *context*; only the answer is *supervised*.

## 6.7 Why the image tokens are never targets

The language head predicts a distribution over vocabulary IDs, and projected
image vectors have no vocabulary IDs. Ignore them as **targets**, but keep them
visible to later tokens in the **attention mask**. Answer loss can still train
the projector through that attention path.

**Check:** in this assistant-only recipe, which tokens get loss weight 1? → *the
selected assistant response spans (often all assistant turns, plus any stop tokens
the template intentionally supervises); user and system prompts are masked.*

### Worked example: trace the first supervised token

**Why:** after expanding an image placeholder, the input can have the right shape
while the labels are shifted by hundreds of positions. The loss may still fall
even though the model never learns the intended image–answer association.

**What:** replace one placeholder with two visual embeddings. With simplified
role markers and single-token answer `red`, the post-expansion layout is:

| Position | Decoder input | Label at this position | Logit predicts a supervised target? |
|---:|---|---|---|
| 0 | `BOS` | ignore | no |
| 1 | visual embedding 0 | ignore | no |
| 2 | visual embedding 1 | ignore | no |
| 3 | user question token | ignore | no |
| 4 | `<assistant>` | ignore | yes → `red` |
| 5 | `red` | `red` | yes → `EOS` |
| 6 | `EOS` | `EOS` | no |

**How:** expand an ignore label alongside every inserted visual vector and shift
exactly once. If the model API shifts internally when given `labels`, pass the
unshifted column above; if you compute cross-entropy yourself, pair `logits[:-1]`
with `labels[1:]`. Inspect one real collated batch: after padding/truncation,
embedding lengths and label lengths must match, padding and vision labels must
be ignored, and at least one assistant target must remain per example used for
training. In a microbatch test, verify that a frozen LLM accumulates no parameter
gradients but the projector receives a nonzero gradient. Freezing the LLM's
parameters is not the same as wrapping its forward pass in `no_grad()`.

## 6.8 Inference — the loop, with an image

```python
def llava_infer(image, prompt_ids):
    v = projector(vit(image))                    # [P, D]
    h = cat([v, embed(prompt_ids)])              # prefill input
    cache = new_caches()
    logits = gpt(h, cache)                       # one prefill over P + L tokens
    toks = []
    for _ in range(max_new):
        t = sample(logits[:, -1])
        if t == EOS: break
        toks.append(t)
        logits = gpt(embed(t), cache)            # each new token sees the image
    return detokenize(toks)
```

The image is encoded and prefetched into decoder KV once per prefill. Each decode
step still reads the retained visual KV and computes decoder output for the new
token. A long answer can dominate end-to-end latency even with perfect KV reuse.

**Check:** two questions, same image, same process: how many ViT runs? → *two,
unless you cache the vision tokens — a standard serving optimization (Ch 9).*

## 6.9 Training stage 1 — align the spaces without updating the backbones

```python
# data: image-caption pairs (about 558k for LLaVA-1.5)
for image, caption in batch:
    h = cat([projector(vit(image)), embed(caption)])
    # supervise ONLY the caption tokens:
    loss = masked_ce(gpt(h), targets=caption_ids)
    loss.backward()      # grads flow into projector only (ViT & LLM frozen)
```

After stage 1: the projector speaks the LLM's language. Text-only LLM behavior is
*untouched* (it never received a gradient).

**Check:** why not just train the projector on captioning data end-to-end in one
stage with the LLM unfrozen? → *you can, but stage 1 is a controlled alignment:
cheaper, safer (no LLM drift), and the stage-2 SFT converges faster from a
pre-aligned bridge.*

## 6.10 Training stage 2 — instruction tuning

```python
# data: image-grounded instructions, including synthetic and existing task data
# now: projector + LLM (LoRA or full) both update
for (image, turns) in batch:
    h = build_chat_seq(image, turns)             # §6.12 template
    loss = masked_ce(gpt(h), targets=assistant_turns_only)
    loss.backward()
```

Typical hyperparameters (know these are *orders of magnitude*, not gospel):
full-FT LLM @ LR ~1e-5, 1–3 epochs; or LoRA rank 128–1024 @ LR ~1e-4 (Ch 7).

**Check:** why keep epoch count low in stage 2? → *the data is synthetic-ish and
repetitive; 2–3 epochs already risks memorizing answers; more epochs = overfit +
text-capability bleed.*

## 6.11 The data engine (where LLaVA actually won)

The architecture was simple; the *data* was the move:
- The original [LLaVA](https://arxiv.org/abs/2304.08485) used text-only GPT-4 with
  COCO captions and bounding-box descriptions to write conversations, detailed
  descriptions, and reasoning questions; GPT-4 was not shown the source pixels.
- The original release used roughly 158k generated instruction examples.
  [LLaVA-1.5](https://arxiv.org/abs/2310.03744) used roughly 665k examples in an
  expanded mix that also included existing task datasets.
- A synthetic teacher cannot verify details absent from its inputs. Audit labels
  against pixels before treating them as evidence of fine visual grounding.

🔗 **BEV bridge:** this is your auto-labeling pipeline, upgraded: instead of
labeling boxes, you label *behavior and reasoning* — and the labels are in a format
any LLM can consume. Data engine > architecture is the pattern everywhere in 2024–26.

**Check:** what is a likely failure if the teacher only sees captions? → *it may
write a plausible answer about a detail that the captions never established.
Verify the claim against the image, not against the teacher's fluency.*

## 6.12 Chat templates and multi-image

```
<image>
<user>What's on the sign?</user>
<assistant>It says "Main St 5 mph".</assistant>
<image><image>
<user>Compare the two.</user>
<assistant>The left sign is a stop sign; the right is yield.</assistant>
```

- `<image>` is a **placeholder** in the text that gets *expanded* to the P vision
  tokens at sequence build time (§6.5).
- Multiple images = multiple placeholder expansions, in order.
- Roles/turns are what the loss mask follows: every `<assistant>…</assistant>`
  span is supervised; everything else is context.

**Check:** what if the template puts a stray special token *inside* the answer?
→ *it gets supervised too — you're teaching the model to emit template junk.
Templates must be byte-exact between training and inference.*

## 6.13 High resolution: any-resolution tiling (LLaVA-NeXT)

Problem: 336 px crushes detail. LLaVA-NeXT's answer — **tile the image**:

```python
def any_res_tokens(image, vit):
    global_feats = vit(resize(image, 336))           # 576 tokens: the "coarse view"
    (h, w) = pick_grid(image)                        # e.g. 2×2 for a wide image
    tiles = split_into_grid(image, h, w)
    tile_feats = [vit(resize(t, 336)) for t in tiles]  # 576 tokens *each*
    return cat([global_feats] + tile_feats)          # 576*(1 + h*w) tokens
```

2×2 grid → 576×5 = 2880 tokens in this simplified accounting. Actual
LLaVA-NeXT packing can rearrange tile features into a spatial grid, remove padded
regions and add row separators. Count tokens after packing for the checkpoint
you serve. Qwen2-VL instead uses dynamic-resolution preprocessing and a learned
patch merger ([Chapter 5](ch05-bridging-vision-to-language.md)).

**Check:** 2880 vision tokens + 2k text, what do you hit first? → *context window
and KV memory (Ch 9) — the reason token *pruning* at inference exists (Ch 8).*

## 6.14 The "what breaks" list (debug intuition)

| Symptom | First suspect |
|---|---|
| Answers ignore the image ("plausible but generic") | verify image insertion, assistant targets and projector gradients before changing the training recipe |
| Text-only capability regressed | stage 2 LR/epochs too hot; mask leaking onto context |
| Bad at small text/objects | resolution or pooling (§6.13, §5.7) |
| Inconsistent spatial answers ("left/right" flips) | position encoding too weak (plain 1D RoPE on image tokens) |
| Template artifacts in output | train/serve template mismatch (§6.12) |
| High cost per image | token budget: tiling without pooling, no pruning (Ch 8) |

## 6.15 ❓ Interview

**Q: Walk me through LLaVA's forward pass.** — Use §6.3–6.6 in order; end with the
masking line ("only assistant tokens are supervised"). 60 seconds, no gaps.

**Q: Why two training stages?** — Stage 1 aligns two *pre-trained, frozen* spaces
with a tiny trainable bridge (cheap, safe); stage 2 adapts the bridge+LLM to
instruction behavior. Skipping stage 1 → unstable joint start + LLM drift.

**Q: Where does the image live in the sequence, and does it matter?**
— Convention: front. It matters because (a) causal attention lets the answer
attend to it, (b) its KV is built once in prefill and reused for every generated
token, (c) 2D position schemes attach to it. Back-placement also works; it's a
template detail, not physics.

**Q: What's the token cost of a LLaVA-NeXT 2×2 image, and how do you manage it?**
— 576 global + 4×576 tiles = 2880 before any unpadding and separators. Manage: adaptive grid by aspect ratio, cap
total tokens, pool 2×2 within tiles if the LLM is small, or prune low-attention
vision tokens at inference (§8.3).

**Q: If you had to rebuild LLaVA today with one change, what would it be?**
— Strong answer set (pick one, justify): native 2D/3D RoPE instead of 1D for
spatial grounding; or a second dense encoder branch (DINOv2) for geometry; or
inference-time token pruning to cut cost without hurting quality. Name the *trade*
you're making in each case.
