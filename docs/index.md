# From BEV to Tokens

## The VLM Handbook for the CV Engineer

A practical book for experienced computer-vision engineers — CNNs, two-stream
fusion, BEV perception, LSS/BEVFormer-style view transform, temporal fusion,
deployment — who are moving into **Vision–Language Models (VLMs / MLLMs)** and
preparing for technical interviews at teams working on frontier multimodal systems.

Every core concept comes with **pseudocode** (Python/PyTorch-flavored, shapes
annotated), an **"Interview"** box (what to expect, how to answer), and a
**"BEV bridge"** box (the analogy to your existing world).

---

## How this book is written

This book is written in the **bite-size, build-it-up** style: every section is
*one idea, one screen*, with a small piece of code, the reason it exists, and a
quick check. You read a section, run the idea in your head, pass the check, move
on. When a chapter builds a system (e.g. an MLLM forward pass), it does it in
**tiny increments** — each step adds one line of mechanism and explains *why that
line is there*.

Think of it like `LLVM Essentials`: you don't read a compiler, you *assemble*
one, one pass at a time. Here you assemble a VLM, one token at a time.

**Conventions**

- Pseudocode is Python-like, `torch`-style. Shapes are annotated as `[B, T, D]`,
  where `B` = batch, `T` = sequence length (tokens), `D` = model hidden dim.
- `#` comments inside pseudocode are load-bearing — read them.
- Each section ends with a one-line **Check** (answer it in your head, no lookup).
- **🔗 BEV bridge** = the 1-sentence mapping back to your CNN/BEV intuition.
- **❓ Interview** (at chapter end) = the questions to expect, with answer skeletons.
- **🎯** at a chapter's top = the single sentence you must be able to say out loud.

## Suggested paths

- **1-day crash path (before an interview):**
  [ch01](ch01-the-landscape.md) → [ch02](ch02-the-llm-refresher.md) →
  [ch06](ch06-llava-the-blueprint.md) → [ch09](ch09-inference-and-serving.md) →
  [ch12](ch12-interview-playbook.md). You'll be able to talk about how VLMs work,
  serve them, and answer the core question bank.
- **1-week path:** add [ch03](ch03-clip-contrastive-pretraining.md) →
  [ch04](ch04-vision-encoders.md) → [ch05](ch05-bridging-vision-to-language.md) →
  [ch07](ch07-finetuning-and-alignment.md) in that order (the "how it's built" spine).
- **Full path (2–3 weeks):** all chapters, then [ch10](ch10-vlms-for-autonomous-driving.md)
  and [ch11](ch11-evaluation.md) to connect back to your AV background.

## What this book is *not*

- Not an LLM course from zero: attention, backprop, and basic PyTorch are assumed.
- Not a model zoo: specific numbers (parameter counts, scores) change quarterly;
  what doesn't change is architecture and data flow — that's what this book nails down.
- Not a claim of completeness: pick any model from the last 12 months and this book
  gives you the vocabulary to deconstruct it in ~10 minutes.

## Start reading

👉 **[Chapter 1 — What a VLM Actually Is](ch01-the-landscape.md)**
