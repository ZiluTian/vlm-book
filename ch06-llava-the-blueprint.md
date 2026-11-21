# Chapter 6 — LLaVA: The Blueprint, Built in Stages

> 🎯 **The one sentence:** LLaVA = frozen ViT + frozen LLM + a tiny trained projector,
> where image features are prepended to the text sequence as "vision tokens" and only
> the assistant's reply tokens are supervised. Build that forward pass below, one
> stage at a time — it's the mental model for *every* encoder-connector VLM.

## 6.1 The three components, and who moves

```
[ViT-L, frozen] ──► [projector, TRAINED] ──► [LLaMA-7B, frozen in stage 1]
```

LLaVA v1: linear projector, 595k image–text pairs, stage 1 only.
LLaVA v1.5: MLP projector, two stages, instruction data → this is the reference.

**Check:** which component holds the most parameters? → *the LLM. The projector is
~0.1% of the model — that's the whole design bet: tiny bridge, frozen giants.*

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
v = projector(feat)         # [576, 1024] → [576, 3584]   (MLP in LLaVA-1.5)
v = v * 100                 # LLaVA's scale: match text-embedding norms (§5.2)
```

Now `v` is a bag of 576 vectors that *look like* the LLM's own embedded tokens.

## 6.5 Step 3 — assemble the sequence

```python
e_q = embed(question_ids)           # [Lq, D]
seq = cat([v, e_q], dim=0)          # [576 + Lq, D]
```

The image goes **first**, then the question. Positions: RoPE assigns 0,1,2,… over
the whole sequence (Qwen2-VL instead gives the image tokens 2D (h,w) positions —
Ch 4 §4.6). The model never sees a boundary marker; it just sees 576 tokens that
"mean something" followed by text.

**Check:** could the image go *after* the question? → *yes, and some templates do;
causal attention means the answer (generated after both) can attend to either.
Front placement is convention + cache-friendly (image KV built once, reused).*

## 6.6 Step 4 — the full forward with the loss

```python
def llava_train(image, question_ids, answer_ids):
    v = scale(projector(vit(image)))              # [P, D]
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

If you supervised them, the model would be trained to *predict image vectors from
the previous image vector* — i.e., to hallucinate images as "text." They carry
information *into* the sequence; they are not the model's output. Mask = 0.
(One-liner for the interview: "we condition on the image, we don't generate it.")

**Check:** in a chat model, which tokens get loss weight 1? → *only the assistant's
reply tokens, after the last user turn (and system prompts are masked too).*

## 6.8 Inference — the loop, with an image

```python
def llava_infer(image, prompt_ids):
    v = scale(projector(vit(image)))             # [P, D]
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

Notice the economics: the image costs **one** expensive prefill; every decoded token
after that just reads the cached image KV. Longer answers are *cheap* relative to
the image — the opposite of a CNN where you'd re-run the backbone per query.

**Check:** two questions, same image, same process: how many ViT runs? → *two,
unless you cache the vision tokens — a standard serving optimization (Ch 9).*

## 6.9 Training stage 1 — align the spaces (train ~0.1% of the model)

```python
# data: 595k (image, caption) pairs, captions from web/GPT-4
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
# data: ~150k multi-turn conversations (GPT-4-generated: QA, complex reasoning,
#       dialogue, grounding) — see §6.11
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
- GPT-4 looked at CC3M images and wrote (a) detailed captions, (b) complex
  Q&A (multi-step, compositional), (c) multi-turn conversations.
- ~665k instruction samples total; quality bar: "would a human ask this?"
- No pixel-level labels, no boxes — the model learns to *describe and reason*,
  grounding emerges from the text.

🔗 **BEV bridge:** this is your auto-labeling pipeline, upgraded: instead of
labeling boxes, you label *behavior and reasoning* — and the labels are in a format
any LLM can consume. Data engine > architecture is the pattern everywhere in 2024–26.

**Check:** why did GPT-4-written data beat more, human-annotated data? → *scale of
diversity: 665k varied instruction types, cheap; human annotation at that breadth
is cost-prohibitive. Quality control via prompt design + filtering.*

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

2×2 grid → 576×5 = 2880 tokens. The LLM sees a low-res overview *plus*
high-res local detail — like a FPN with explicit token budgets per level.
InternVL does the same family of trick with a bigger ViT. (And the no-pooling
dynamic-resolution camp — Qwen2-VL — does it by simply running the ViT at native
size. §5.8.)

**Check:** 2880 vision tokens + 2k text, what do you hit first? → *context window
and KV memory (Ch 9) — the reason token *pruning* at inference exists (Ch 8).*

## 6.14 The "what breaks" list (debug intuition)

| Symptom | First suspect |
|---|---|
| Answers ignore the image ("plausible but generic") | connector under-aligned (stage 1 undertrained, wrong scale §5.2) |
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
— 576 global + 4×576 tiles = 2880. Manage: adaptive grid by aspect ratio, cap
total tokens, pool 2×2 within tiles if the LLM is small, or prune low-attention
vision tokens at inference (§8.3).

**Q: If you had to rebuild LLaVA today with one change, what would it be?**
— Strong answer set (pick one, justify): native 2D/3D RoPE instead of 1D for
spatial grounding; or a second dense encoder branch (DINOv2) for geometry; or
inference-time token pruning to cut cost without hurting quality. Name the *trade*
you're making in each case.
