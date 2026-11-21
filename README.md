# From BEV to Tokens: The VLM Handbook for the CV Engineer

> **Read the book online:** [zilutian.github.io/vlm-book](https://zilutian.github.io/vlm-book/)
> (rendered with MkDocs Material, rebuilt on every push to `main`)

A bite-size, pseudocode-driven book on **Vision-Language Models** for experienced
CV engineers — CNNs, two-stream fusion, BEV perception, deployment — preparing for
technical interviews on frontier multimodal systems.

Every core concept comes with pseudocode (Python/PyTorch-flavored, shapes
annotated), the *why* behind each piece, and a one-line comprehension **Check**.
Each chapter ends with an **Interview** question bank. A chapter's **🎯** one-liner
is the sentence you must be able to say out loud.

## Layout

- `docs/` — the book (12 chapters + glossary)
- `mkdocs.yml` — site configuration (MkDocs Material)
- `.github/workflows/deploy.yml` — CI: build on every push, deploy to GitHub Pages

## Suggested paths

- **1-day crash path (before an interview):** ch01 → ch02 → ch06 → ch09 → ch12
- **1-week path:** + ch03 → ch04 → ch05 → ch07 (the "how it's built" spine)
- **Full path (2–3 weeks):** all chapters, then ch10 + ch11 to connect back to AV
