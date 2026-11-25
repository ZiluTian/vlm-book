---
name: VLM Handbook Editing
description: Revise or review a VLM handbook chapter with a running example, shape-annotated pseudocode, measured tradeoffs, and verifiable engineering checks.
---

## Workflow

1. Read the target chapter and any linked chapter that defines its quantities.
   Identify the decision the reader needs to make, not just the vocabulary to learn.
2. Use **Why → What → How**: establish the failure or constraint, explain what
   changes at the relevant boundary, then show how to reproduce or test it.
3. Carry one concrete input across sections where possible. Trace the actual
   intermediate values: token counts, tensor widths, logits, labels, or bytes.
   Explain every number that appears in the pseudocode. Distinguish illustrative
   inputs from checkpoint settings, and estimates from measurements.
4. Change one variable at a time in a comparison. Name the control, the failure
   slice, the metric, and what outcome would change the engineering decision.
   Don't turn a plausible diagnosis into a proven cause.
5. Preserve chapter conventions (`**Check:** ... → *...*`, interview prompts,
   and stable section anchors). Check arithmetic, run `git diff --check` and
   `mkdocs build --strict`, and verify any changed link destinations. For this
   repository, commit and push the scoped changes as requested; distinguish a
   successful push from a verified GitHub Pages deployment.

## Reference pattern: Chapter 5's connector comparison

- Start with a 336×336 image and 14-pixel patches: `24 × 24 = 576` spatial slots.
  Track slot count separately from width: `[576, D_v] → [576, D]` can change
  the representation without changing the sequence length.
- Compare the same encoder features under a projector (576 tokens), 2×2 pooling
  (144), and a 32-query resampler (32). At 128 KiB of decoder KV per token,
  that is 72, 18, and 4 MiB per image. It is not total GPU memory or a claim
  that the upstream encoder did less work.
- Test what the loss of slots means: hold the encoder, decoder, data, and training
  budget fixed; compare single-digit OCR changes, scene QA, and time to first
  token. A high-resolution crop probes upstream evidence; raising the connector
  budget on the same features probes compression. Neither result proves causation
  alone. Keep generic pooling, learned merging, and checkpoint-specific packing
  separate rather than attributing them all to the same model.
