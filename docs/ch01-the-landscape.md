# Chapter 1 — What a VLM Actually Is

> 🎯 **The one sentence:** a VLM turns pixels into a sequence of tokens, sticks them
> in front of a language model, and lets the LM read, reason, and talk over the image.
> Everything else in this book is a refinement of that one move.

## 1.1 Start from a problem

You have a dashcam clip. A human asks: *"Why did the car brake?"*

Your CV toolkit can answer half of that: detectors, trackers, occupancy. But the
*why* — "braked because a delivery van ignored its blinking turn signal" — needs
**open-vocabulary semantics + reasoning + language output**. That's exactly the gap
VLMs fill: they turn *pixels* into something a *language model* can chew on.

**Check:** what output format do you need for the answer to be useful to a human
(or another model)? → *text. That fact decides a lot of the architecture.*

## 1.2 Family A — contrastive two-tower (CLIP)

Two separate encoders, trained so matching image–text pairs get high cosine
similarity:

```
image ──► [ViT] ──►  img_vec  [D]
text  ──► [Xfmr] ──►  txt_vec  [D]
                    cos(img_vec, txt_vec)  ← high for the right pair
```

- No generation. You get a *score*, not a sentence.
- Image and text paths never meet during inference → you can cache either side.
- **Check:** can CLIP tell you *where* the stop sign is? → *No. One pooled vector
  per image; spatial structure is gone. (That weakness drives half of Ch 4.)*

Full loss details: Chapter 3.

## 1.3 Family B — encoder → connector → LLM (the main course)

A frozen-ish **vision encoder** makes image features; a small **connector** maps them
into the LLM's space; a **decoder LLM** reads *image tokens + text tokens* and
generates the answer.

```
img ──► [ViT] ──► P features ──► [connector] ──► P "vision tokens"
text ──────────────────────────► L text tokens
                                  └──► [LLM decoder] ──► "A van..."
```

LLaVA, Qwen-VL, InternVL, BLIP-2 — all of them are variations of this diagram.
This family is what Chapters 4–7 deconstruct.

**🔗 BEV bridge:** this is your two-stream fusion. Camera backbone = vision encoder,
BEV token grid = vision token sequence, BEVFormer cross-attention = connector.

**Check:** which part is usually *frozen* first? → *the LLM (and often the ViT too);
the connector is what you train.*

## 1.4 Family C — native multimodal

No separate encoder, no connector. Pixels get **tokenized directly** and one
transformer is pre-trained from the start on interleaved image–text data.
Models: Chameleon, Gemini, GPT-4o.

- Strongest on interleaved chains (text → image → text → image), because there is
  no feature-space wall to cross mid-sequence.
- Cost: you pay full pretraining for vision; you can't "bolt on" a big LLM later.

**Check:** what's the architectural price of nativeness? → *no freeze-and-plug;
vision and language must be trained together from (near) scratch.*

## 1.5 Family D — the generation side (know the shape, skip the detail)

Models that *emit* images: autoregressive over discrete VQ tokens (Coca, Emu3),
a diffusion head, or hybrids (Transfusion: AR for text, diffusion for image patches).
Interview relevance: one question — *"discrete tokens vs. diffusion for image
output?"* Answer lives in §8.5.

**Check:** why are image outputs often *not* autoregressive like text? → *images are
regular, continuous, fixed-size — diffusion (parallel) fits better than a token
stream (sequential).*

## 1.6 The anatomy, fully labeled

```
                     ┌──────────────────────────────────────────┐
 image [3,H,W] ────► │  VISION ENCODER  (ViT; often frozen)     │
                     │  patchify → N blocks → final norm        │
                     └─────────────────┬────────────────────────┘
                                       │  P features [P, D_v]
                                       ▼
                     ┌──────────────────────────────────────────┐
                     │  CONNECTOR (linear | MLP | Q-Former | …) │
                     └─────────────────┬────────────────────────┘
                                       │  P vision tokens [P, D]
 question ── BPE ──► L token ids ── embed ──► [L, D]
                                       │
                     concat →  [P+L, D]  + positions (1D/2D RoPE)
                                       ▼
                     ┌──────────────────────────────────────────┐
                     │  LLM DECODER  (N layers, causal attn)    │
                     │  per layer: norm → MHA(+RoPE,KV cache)   │
                     │             → norm → FFN(SwiGLU)         │
                     └─────────────────┬────────────────────────┘
                                       ▼
                             logits [T, V] → sample → next token → loop
```

Memorize this diagram. Every chapter annotates one part of it.

**Check:** after prefill, where does the image "live"? → *nowhere — only its P
tokens, in the sequence/KV cache. That's why token count = detail = cost.*

## 1.7 Translation table (your world → VLM world)

| You know | VLM equivalent |
|---|---|
| camera backbone (ResNet/ViT) | vision encoder |
| BEV token grid | sequence of vision tokens |
| BEVFormer spatial cross-attn | connector / Q-Former queries |
| temporal fusion over BEV | video token pooling / KV cache |
| task heads (det/map/track) | **one LM head** — text is the head |
| extrinsics / BEV voxel coords | position encodings (RoPE, 2D/3D mRoPE) |
| InfoNCE, BCE, F1 losses | next-token cross-entropy (with masking) |
| NMS / class heads | (mostly) gone — detection becomes text |

**Check:** which row is the *shocking* one for a CV engineer? → *the task-head row:
one head replaces all of them, and it outputs text.*

## 1.8 Why the LLM decoder won the multimodal race

1. **Data.** ~10¹⁵ text tokens vs ~10¹¹ image–text pairs. Scaling the LLM first,
   then adding vision, was far cheaper — and bought the *reasoning* that makes VLMs
   more than captioners.
2. **Text is a universal interface.** Detection (coords as text), reasoning, tool
   calls, agents — one output format does it all.
3. **Scaling laws transferred.** Decoders scaled cleanly; vision plugged in for free.
4. **Composition.** AR decoding composes with agent loops (plan → tool → observe).
   That's *why* the frontier is going native (Family C).

**Check:** name the reason that is *not* a data argument. → *#2, #3, #4; only #1 is
pure data.*

## 1.9 What transfers, what resets

**Transfers:** data is the moat; loss/masking discipline; eval hygiene; deployment
math (FLOPs/memory/bandwidth).

**Resets:**
- no pooling pyramid — token count is an explicit budget you choose;
- no NMS/class heads — behavior is emergent from text generation;
- receptive field = context length, metered in KV-cache memory;
- spatial layout must be *encoded in position* (RoPE), not assumed from a grid.

**Check:** your BEV model had a fixed BEV grid. What is the VLM's analog of that
grid, and what's the cost of making it bigger? → *the P vision tokens; cost =
context length, KV memory, and prefill FLOPs.*

## 1.10 Which model is which? (orientation, no details)

| Model | Family | Hook |
|---|---|---|
| CLIP / SigLIP | A | contrastive alignment |
| BLIP-2 | B | Q-Former connector |
| Flamingo | B | Perceiver resampler |
| LLaVA(-NeXT) | B | linear projector → tiling |
| Qwen2/2.5-VL | B | keep *all* tokens, 2D/3D RoPE |
| InternVL | B | huge ViT (1B+), dynamic tiling |
| Chameleon, Gemini, GPT-4o | C | native pretraining |
| EMMA / OpenEMMA | B→VLA | VLM that also emits trajectories (Ch 10) |
| GAIA-1 | world model | diffusion video gen for sim (Ch 10) |

### Worked example: one dashboard photo, three different jobs

**Why:** “understand the dashboard” could mean finding similar photos, reading
the odometer, or locating it. Those jobs require different outputs.

**What:** suppose the photo shows `10482` miles. A two-tower model can rank
candidate captions such as “an illuminated dashboard” above unrelated text, but
its score is not a generated answer or a box. An encoder–connector–decoder model
can take “What mileage is displayed?” and generate `10482`. If the task needs
the dial's position, require a box in an explicit coordinate system rather than
assuming that a correct answer implies correct localization.

**How:** test each output separately. Measure retrieval recall for similar
photos, exact match for mileage, and box IoU for localization. Add a second photo
that differs only in the last digit: if both photos retrieve the same dashboard
caption, that's fine; if both produce `10482`, the reader failed the OCR task.
See [Chapter 5](ch05-bridging-vision-to-language.md) for how the connector can
affect that last digit.

## 1.11 ❓ Interview — "Walk me through an image + question in a VLM"

Ninety seconds, five beats:

1. *Encode:* image → patches → ViT → P feature vectors (576 at 336 px).
2. *Project:* connector maps each to LLM dim D → P "vision tokens."
3. *Concat:* `[v₁…v_P, question tokens]` with positions assigned (1D or 2D RoPE).
4. *Prefill:* one parallel forward pass; KV cache built; model is now "reading" it.
5. *Decode:* each new token attends over the cached KVs — image included — until EOS.

Closing line (this is the differentiator):
*"After prefill, the image is never looked at again — its entire influence on the
answer is those P tokens. Token count is therefore simultaneously a detail knob,
a quality knob, and a cost knob."*
