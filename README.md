# From BEV to Tokens: The VLM Handbook for the CV Engineer

A practical book for experienced computer-vision engineers — CNNs, two-stream fusion,
BEV perception, LSS/BEVFormer-style view transform, temporal fusion, deployment —
who are moving into **Vision–Language Models (VLMs / MLLMs)** and preparing for
technical interviews at teams working on frontier multimodal systems.

Every core concept comes with **pseudocode** (Python/PyTorch-flavored, shapes
annotated), an **"Interview"** box (what to expect, how to answer), and a
**"BEV bridge"** box (the analogy to your existing world).

---

## Who this book is for

You already know:
- ResNet/Swin backbones, FPNs, detection/segmentation heads, NMS, calibration
- BEV perception: camera→BEV lifting (LSS), BEVFormer-style spatial cross-attention,
  temporal self-attention over BEV tokens
- Training loops: mixed precision, DDP, learning-rate schedules, data pipelines,
  ablation discipline, metric tracking

You may *not* yet have:
- written a transformer decoder LLM forward pass
- seen how an image becomes a sequence of tokens inside a language model
- trained or served a VLM, or reasoned about KV-cache memory and token budgets

That gap is exactly what this book closes.

## How this book is written

This book is written in the **bite-size, build-it-up** style: every section is
*one idea, one screen*, with a small piece of code, the reason it exists, and a
quick check. You read a section, run the idea in your head, pass the check, move on.
No section assumes you read the last one beyond what it states. When a chapter builds
a system (e.g. an MLLM forward pass), it does it in **tiny increments** — each step
adds one line of mechanism and explains *why that line is there*.

Think of it like `LLVM Essentials`: you don't read a compiler, you *assemble* one,
one pass at a time. Here you assemble a VLM, one token at a time.

## Conventions

- Pseudocode is Python-like, `torch`-style. Shapes are annotated as `[B, T, D]`,
  where `B` = batch, `T` = sequence length (tokens), `D` = model hidden dim.
- `#` comments inside pseudocode are load-bearing — read them.
- Each section ends with a one-line **Check** (answer it in your head, no lookup).
- **🔗 BEV bridge** = the 1-sentence mapping back to your CNN/BEV intuition.
- **❓ Interview** (at chapter end) = the questions to expect, with answer skeletons.
- **🎯** at a chapter's top = the single sentence you must be able to say out loud.

## Table of contents

| # | Chapter | What you'll be able to do after |
|---|---------|--------------------------------|
| 1 | [The Landscape](ch01-the-landscape.md) | Classify any VLM you meet; explain the 3–4 architectural families |
| 2 | [The LLM Refresher](ch02-the-llm-refresher.md) | Write a GPT-style forward pass, the KV-cache decode loop, and the loss |
| 3 | [CLIP and Contrastive Pretraining](ch03-clip-contrastive-pretraining.md) | Explain InfoNCE with in-batch negatives, zero-shot classification, and SigLIP |
| 4 | [Vision Encoders](ch04-vision-encoders.md) | Explain ViT vs. CNN, DINOv2, SAM, and which encoder to pick and why |
| 5 | [Bridging Vision to Language](ch05-bridging-vision-to-language.md) | Explain linear projectors, Q-Former, resamplers, and token-budget trade-offs |
| 6 | [LLaVA: The Blueprint](ch06-llava-the-blueprint.md) | Walk the *entire* forward pass of a modern MLLM, image to token, step by step |
| 7 | [Fine-tuning and Alignment](ch07-finetuning-and-alignment.md) | Explain SFT masking, LoRA/QLoRA, DPO, and GRPO (R1-style reasoning RL) |
| 8 | [The Frontier](ch08-the-frontier.md) | Discuss native multimodality, video tokens, token pruning, multimodal reasoning, unified generation |
| 9 | [Inference and Serving](ch09-inference-and-serving.md) | Compute KV-cache memory, explain FlashAttention, continuous batching, speculative decoding; do cost math |
| 10 | [VLMs for Autonomous Driving](ch10-vlms-for-autonomous-driving.md) | Design where a VLM lives in an AV stack; discuss VLA models, world models, data engines |
| 11 | [Evaluation](ch11-evaluation.md) | Pick/design benchmarks, fight hallucination, use LLM-as-judge, build your own eval set |
| 12 | [The Interview Playbook](ch12-interview-playbook.md) | 40+ question bank with answer skeletons, a full system-design walkthrough, math cheat-sheet |
| — | [Glossary](glossary.md) | Fast lookup for every term used in this book |

## Suggested paths

- **1-day crash path (before an interview):** Ch 1 → 2 → 6 → 9 → 12. You'll be able to
  talk about how VLMs work, serve them, and answer the core question bank.
- **1-week path:** Add Ch 3 → 4 → 5 → 7 in that order (the "how it's built" spine).
- **Full path (2–3 weeks):** All chapters, then Ch 10 and 11 to connect back to your AV background.

## What this book is *not*

- Not an LLM course from zero: attention, backprop, and basic PyTorch are assumed.
- Not a model zoo: specific numbers (parameter counts, scores) change quarterly;
  what doesn't change is architecture and data flow — that's what this book nails down.
- Not a claim of completeness: pick any model from the last 12 months and this book
  gives you the vocabulary to deconstruct it in ~10 minutes.
