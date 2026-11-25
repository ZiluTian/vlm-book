---
hide:
  - toc
---

<div class="cover" markdown="1">

<span class="folio">Engineering field guide &nbsp; / &nbsp; Multimodal systems</span>

# From BEV to Tokens

**You can lift a camera feature into a BEV grid. What does it take to turn that feature into a word?**

![An image patch grid unfolds into a sequence of visual tokens followed by text tokens.](assets/cover.svg)

[Start with Chapter 01 →](ch01-the-landscape.md){ .cover-link }

</div>

## Open the book at …

<div class="plates" markdown="1">

- **I · Foundations**  
  [How does a decoder see a patch?](ch01-the-landscape.md)<br>
  Start at Chapter 1 and trace an image from the vision encoder to the decoder. For the projector and compression tradeoffs, jump to [Chapter 5: The connector](ch05-bridging-vision-to-language.md).
- **II · The blueprint**  
  [Build the forward pass](ch06-llava-the-blueprint.md)  
  Start at Chapter 6 and follow one image and one question all the way to the first supervised token.
- **III · Production**  
  [Price a high-resolution image](ch09-inference-and-serving.md)  
  Start at Chapter 9 and count the image's tokens twice: once for prefill, then for the cache that stays resident.
- **IV · Interview studio**  
  [Take the whiteboard round](ch12-interview-playbook.md)  
  Start at Chapter 12: size a cache, defend a connector, and design a fleet-data pipeline.

</div>

## Pick a reading route

**Want to understand the mechanism, not just remember the answer?** Each chapter
has a **Why → What → How** worked example:

- **Foundations:** [one photo, three tasks](ch01-the-landscape.md#worked-example-one-dashboard-photo-three-different-jobs), [the first decode step](ch02-the-llm-refresher.md#worked-example-follow-one-visual-prompt-into-decode), [false negatives in CLIP](ch03-clip-contrastive-pretraining.md#worked-example-the-hard-negative-that-isnt-negative), [resize, tile, or crop](ch04-vision-encoders.md#worked-example-resize-tile-or-crop), [OCR after compression](ch05-bridging-vision-to-language.md#worked-example-why-did-ocr-get-worse-after-compression).
- **The blueprint:** [the first supervised token](ch06-llava-the-blueprint.md#worked-example-trace-the-first-supervised-token), [LoRA training memory](ch07-finetuning-and-alignment.md#worked-example-what-lora-saves-and-what-it-doesnt), [a warning light between frames](ch08-the-frontier.md#worked-example-the-warning-light-between-sampled-frames).
- **Production:** [KV under burst load](ch09-inference-and-serving.md#worked-example-budget-a-burst-of-document-questions), [audit a fleet-data shortlist](ch10-vlms-for-autonomous-driving.md#worked-example-find-unusual-clips-without-hiding-the-missed-ones), [did the model use the image?](ch11-evaluation.md#worked-example-did-the-model-use-the-image-or-guess-from-the-prompt).
- **Interview:** [defend a connector on the whiteboard](ch12-interview-playbook.md#worked-example-defend-a-connector-on-the-whiteboard).

**Interview tomorrow?** Read the [architecture map](ch01-the-landscape.md), [decoder essentials](ch02-the-llm-refresher.md), [LLaVA walkthrough](ch06-llava-the-blueprint.md), and [serving math](ch09-inference-and-serving.md). Then close the book and try the [interview problems](ch12-interview-playbook.md).

**Building a model?** Work through Chapters 1–8 in order. Sketch the shapes beside each pseudocode block. At each **Checkpoint**, give your answer before opening the explanation.

**Coming from a driving team?** Pair [VLMs for autonomous driving](ch10-vlms-for-autonomous-driving.md) with [evaluation](ch11-evaluation.md). Ask which claims can be checked from pixels and which need your geometric stack.
