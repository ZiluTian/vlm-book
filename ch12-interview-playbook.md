# Chapter 12 — The Interview Playbook

> 🎯 **The one sentence:** you will be graded on three things — (1) can you explain
> how a VLM works *mechanistically* (Ch 2, 6), (2) can you reason about *trade-offs
> with numbers* (Ch 5, 8, 9), and (3) can you *design a system* that puts a VLM in
> the right place (Ch 10, §12.5). This chapter is the question bank and the answers.

## 12.1 The narrative (lead with this, it's your edge)

> "I've spent X years building the geometric layer — CNNs, two-stream fusion,
> BEV perception, deployment. VLMs are the *semantic* layer: a tokenized image
> feeding a decoder LLM. I can explain the whole data path from pixels to logits
> (Ch 6), I can do the serving math (Ch 9), and I know exactly where each layer
> belongs in a driving stack (Ch 10): geometry at 10 Hz, semantics at 0.1 Hz,
> and the data loop around both."

Every answer after this should land back on that spine. You're not a "CV person
learning NLP" — you're a *perception person who now owns the interface to language*.

## 12.2 Quick-fire concept questions (answer in 2–3 lines each)

| # | Question | Model answer (skeleton) |
|---|---|---|
| 1 | What is a VLM? | Pixels → ViT features → connector → vision tokens in an LLM sequence → AR text out. (Ch 1 §1.6.) |
| 2 | How does CLIP train? | Dual towers, InfoNCE, in-batch negatives, learned temperature; zero-shot = text-embedding lookup. (Ch 3.) |
| 3 | SigLIP vs CLIP? | Sigmoid per-pair vs softmax-coupled; scales to huge batches; better high-res. (Ch 3 §3.8.) |
| 4 | Why a connector at all? | Space mismatch (D_v≠D, no shared geometry), token budget, and training stability (align before joint). (Ch 5 §5.1, §5.10.) |
| 5 | Q-Former in one breath? | 32 learned queries cross-attend to image features → fixed, tiny, LLM-friendly token set; detail bottleneck. (Ch 5 §5.4.) |
| 6 | LLaVA training, two stages? | s1: train projector only on captions (ViT+LLM frozen); s2: projector+LLM on GPT-4 instruction data, mask everything but assistant tokens. (Ch 6.) |
| 7 | Why mask image tokens in the loss? | They're context, not output — supervising them trains the model to generate image vectors. (Ch 6 §6.7.) |
| 8 | LoRA math? | W + (α/r)BA, r≪d, B init 0, merge at deploy, ~0.1–1% params. (Ch 7.) |
| 9 | DPO in one breath? | RLHF in closed form: raise relative-to-ref log-prob of preferred, lower dispreferred; no RM, no critic. (Ch 7 §7.8.) |
| 10 | GRPO in one breath? | Group-sample per prompt, score with a *verifier*, rank within group as advantage, ratio+KL loss; no critic, no learned RM. (Ch 7 §7.9.) |
| 11 | KV cache: what/why/size? | Reused K,V per position → O(1) new compute per token; size = 2·layers·kv_heads·head_dim·bytes·T. (Ch 9 §9.3.) |
| 12 | Why is decode bandwidth-bound? | Per token you re-read all weights + all KV; HBM/s is the ceiling. (Ch 9 §9.2.) |
| 13 | Continuous batching? | Per-step request admission/exit; amortize weight reads over a token-budgeted batch; dial between throughput and p99. (Ch 9 §9.5.) |
| 14 | FlashAttention? | Tile attention in SRAM, online softmax, O(T) memory, fewer HBM bytes → faster *and* smaller. (Ch 9 §9.6.) |
| 15 | Where do vision tokens hurt serving? | Prefill FLOPs (S²), permanent KV tax, context window; levers: budget, prefix-cache, prune, quantize. (Ch 9 §9.10.) |
| 16 | FastV idea? | Mid-network, prune low-attention vision tokens (the model's own salience) → 30–70% fewer, ~2–3× decode. (Ch 8 §8.4.) |
| 17 | 2D/3D RoPE (mRoPE)? | Positions as (t,h,w) tuples inside RoPE → variable resolution + video with relative geometry. (Ch 4 §4.6.) |
| 18 | Native vs bolt-on multimodal? | Native: shared tokens/pretraining, interleaved-native, no alignment drift; cost: full pretraining price. (Ch 8 §8.1.) |
| 19 | Why AR for text, (often) diffusion for images? | Text: ordered, open-ended, one-token-at-a-time fits. Images: dense, regular, no intrinsic order → parallel denoising fits. (Ch 8 §8.8.) |
| 20 | DINOv2 in one breath? | EMA-teacher self-distillation: global views supervise local crops, centered+sharpened cosine loss → dense, label-free features. (Ch 4 §4.7.) |
| 21 | Why keep all ViT tokens instead of pooling? | Pooling is global+lossy; the LLM needs spatially-indexed features to ground words to places; token budget is managed elsewhere. (Ch 4 §4.3.) |
| 22 | Hallucination: cause/measure/fix? | Weak mid-layer grounding + token priors + no uncertainty training; POPE-No/CHAIR/MMVP; negative-pair DPO, IoU-verifier GRPO, resolution/data. (Ch 11.) |
| 23 | What's a "world model" for AV? | Video/occupancy diffusion conditioned on (state, action); generates long-tail sim data + counterfactuals. (Ch 10 §10.7.) |
| 24 | VLA vs VLM? | VLM = see+read+speak; VLA = VLM + action output (waypoints as tokens/head) trained with driving rewards. (Ch 10 §10.5.) |
| 25 | What would you improve in LLaVA? | (Any of, with trade named:) 2D/3D RoPE for grounding; second dense encoder (DINOv2); inference token pruning; adaptive tiling. (Ch 6 §6.15, Ch 8.) |

## 12.3 Math questions (do these *out loud*, show the steps)

**M1. KV memory for a 70B (80 layers, 8 kv heads, head_dim 128, fp16) at 32k context.**
→ 2·80·8·128·2 = 320 KB/token → ×32k = **10 GB/user**. Then: "so int8 KV halves it,
context cap to 16k halves it, prefix-sharing makes the system-prompt part free."

**M2. Decode tok/s ceiling for 7B-int4 on an H100, 1 user, 8k context (GQA-8).**
→ bytes/token ≈ weights 4 GB + KV 8k×128 KB ≈ 1 GB → ~5 GB; ÷ 3 TB/s ≈ **~600 tok/s
theoretical**, real ~150–250 (engine overhead). Note the KV term *grows with context*.

**M3. Cost: 100k captions/day, 1k vision + 200 text tokens in, 100 out, 7B-GQA int4, 4×H100.**
→ tokens/day ≈ 100k×1300 ≈ 1.3×10⁸; 4 H100 @ ~2k tok/s combined (realistic, batched)
≈ 6.9×10⁸ tok/day capacity → ~44% utilization → $/day ≈ 4×$3×24 ≈ $288 →
**~$2.2 / 1M tokens** (incl. idle). Compare API: 10–50× that. (Structure > precision.)

**M4. Prefill FLOPs for a 576-token image + 128-text prompt on 7B.**
→ ≈ 2·N·(576+128) ≈ 2·7×10⁹·704 ≈ **10 TFLOP** (matmul term; attention adds a small
S² term). "Image is ~80% of the prefill cost — that's the token-budget argument."

**M5. LoRA param count: 7B, r=128, applied to q,k,v,o of 32 layers, D=4096.**
→ 4 matrices × 32 layers × 2·r·D = 128 × 8.4M ≈ **~1.1×10⁹ → ~15% of the model**
at r=128 (hence "big-rank LoRA" when reshaping behavior; r=16 → ~2%).

**M6. Tokens for a 2×2 tiled 448-px image, LLaVA-NeXT-style, ViT-14 @336/tile.**
→ 576 global + 4×576 tiles = **2880**; + 2k text → ~5k context → KV @128 KB/token
≈ 640 MB. "This is why we cap tiles and prefix-cache images."

## 12.4 Deep-dive questions (answer *skeletons*)

**D1. "Why decoder-only for the LM half?"**
Factorized next-token loss gives one training objective for every capability;
symmetric/encoder-LMs don't compose with generation or agents; scaling laws are
cleanest here. Concession: image *output* often wants diffusion (§8.8) — so
"decoder-only" applies to the language stream, not necessarily everything.

**D2. "Why not train ViT+LLM end-to-end from scratch (one model, one loss)?"**
You can — that's Family C. Costs: full pretraining price for vision; the LLM's text
capability must be re-earned; and you lose the freeze-and-plug property that makes
open-weights fine-tuning cheap. Bolt-on wins on cost/control; native wins on
interleaved chains. Name both, then say which one *your product* needs.

**D3. "What actually limits a VLM's fine-detail performance?"**
Three ceilings: (1) encoder resolution/token budget (info never made it into the
tokens — Ch 5); (2) mid-layer visual grounding (tokens carried the info but the LM
stopped reading them — Ch 11 §11.5); (3) training data (thin-evidence cases were
never supervised). Diagnosis order: tokens → attention → data.

**D4. "Explain the train/serve gap in VLMs."**
Teacher forcing vs sampling (the model never saw its own errors in SFT → RL stages);
template drift (byte-exact templates or outputs rot); token-budget drift (training
at 576 tokens, serving at 2880 → OOD positions/lengths); judge≠user (benchmarks
vs product ODDs — Ch 11 §11.9).

**D5. "How would you cut a VLM endpoint's cost 3× without touching the model?"**
(1) prefix-cache stable image+system blocks (§9.7); (2) token diet: cap tiling,
adaptive grid (§8.5); (3) FastV-style pruning at ~30–50% retention (§8.4);
(4) int8 KV + int4 weights if not already; (5) route easy queries to a smaller
model (cascade). Each is a *serving* lever — measure each on your eval.

**D6. "What breaks if you feed a 4k-video to a 32k-context VLM?"**
Token explosion (16×576=9k/frame → hundreds of k), KV OOM, prefill latency, and
*quality*: attention dilutes over 10⁵ tokens, and the language prior takes over
long before context does. Real systems: keyframes + temporal pooling + hierarchical
summary (Ch 8 §8.2) — i.e., your BEV "detail now, summary far back," again.

## 12.5 System design walkthrough (the one to rehearse)

**Prompt: "Design a VLM-based logging and data-curation system for an AV fleet
(10k vehicles, ~1 PB/week of clips). Goals: find long-tail events, explain
disengagements, and feed training data to our E2E policy."**

**Step 1 — constraints (say these before drawing anything):**
- Volume: 1 PB/week; VLM cost per clip is ~10⁴× a BEV frame → *never* VLM-everything.
- Latency classes: on-vehicle (10 Hz) vs cloud (hours OK). (Ch 10 §10.8.)
- Output contract: (a) long-tail catalog, (b) disengagement explanations, (c)
  training-ready (scenario, clip, labels) tuples.

**Step 2 — the pipeline (the §10.6 diagram, made concrete):**
1. **On-vehicle, always-on:** BEV/occupancy + maneuver detector (cheap, existing)
   emits *event candidates* (hard brake, disengagement, OOD occupancy) + 5 s
   keyframes + state (speed, accel, map context). Event rate: ~10⁻⁴ of frames.
2. **Cloud triage:** small VLM (7B, int4) per candidate: name the event (open-vocab
   caption ≤20 words) + structured JSON {novelty, severity, ODD tags}. Dedupe by
   caption embeddings. (Role A, Ch 10 §10.3.)
3. **Cloud deep-dive (the 1%):** frontier VLM per unique event: full explanation
   ("why did we brake": causal chain), ODD verdict, regression-test candidate flag.
4. **Catalog & gating:** events cluster into scenarios; each gets a *judge spec*
   (Ch 11 §11.8): rubric + metric layer (TTC, IoU, comfort from the *geometric*
   stack). Release gates: new E2E build must not regress the catalog.
5. **Data loop:** approved scenarios → world-model generation (video/occupancy
   conditioned on state+action, Ch 10 §10.7) + real clips → E2E training + VLM
   SFT/DPO pairs ("good explanation" vs "hallucinated explanation" — the
   disengagement logs *are* preference data).

**Step 3 — economics (the numbers that earn the nod):**
- Event rate 10⁻⁴ + 10k vehicles ≈ 10⁵ candidates/day → 7B int4 @ ~150 tok/s/GPU
  on ~8 GPUs handles triage; frontier calls (≈5% of candidates) ≈ 5k/day →
  API or 2–4 H100s with prefix-cached clip contexts.
- Cost is dominated by *triage volume* → the on-vehicle prefilter is the system's
  most important component. (Say that sentence.)

**Step 4 — failure modes you'd pre-empt (senior signal):**
- VLM judge hallucination → metric layer has final say (Ch 11 §11.10);
- template/preprocessing drift breaking prefix cache → versioned pipelines (Ch 9);
- catalog bias (VLM names what it knows) → human review queue on novelty-flagged
  clusters;
- world-model fidelity → validate against fleet geometry before it generates
  training data (Ch 10 §10.12 last Q).

**Step 5 — close:** "The VLM is the *semantic* layer: it names, explains, and
judges. The *geometric* layer — which I've built before — pays for the VLM's
attention and verifies its claims. That split is the architecture."

## 12.6 "Explain it to a PM" questions (depth without jargon)

- **VLM vs a big image classifier?** "A classifier answers 'what classes are present'
  from a fixed list. A VLM answers *anything you can ask in words*, at any resolution
  you choose to pay for — and it *explains*, which is what logs, regulators, and
  debug need."
- **Why isn't this just GPT-4 with eyes?** "GPT-4-class systems *are* VLMs — the
  question is which architecture: bolt-on (fast, controllable, ours) vs native
  (stronger interleaved, frontier-only). For a product: bolt-on on open weights."
- **Why does the image cost 'tokens'?** "The model reads the image the same way it
  reads text — as a sequence. More image detail = longer sequence = more memory and
  compute for every answer. That's the one cost knob that's in the *product* spec."
- **How do you know it's not making things up?** "We measure it: presence/absence
  probes (POPE-style), caption audits (CHAIR), and a calibrated judge over human-
  labeled private clips — with our deterministic metrics (time-to-collision, IoU)
  as the final say. The VLM routes attention; the metrics decide."

## 12.7 Red flags that sink candidates (avoid these)

1. **"The LLM just understands the image."** — No. Say the token path, Ch 6.
2. **Quoting benchmark numbers without the caveat** (multiple-choice floor,
   contamination, what axis it measures — Ch 11 §11.2–11.3).
3. **No numbers anywhere.** Every architecture claim needs a cost: tokens, GB,
   tok/s, $/1M. (Ch 9, §12.3.)
4. **"We'd put the VLM in the control loop."** — Unless the interviewer is selling
   the VLA frontier: latency + verification + cost say no *today* (Ch 10 §10.8).
5. **Confusing the components** (ViT vs connector vs LLM, prefill vs decode,
   SFT vs RL). Know the Ch 1 §1.6 diagram cold.
6. **Dismissing your own background.** The BEV→VLM bridge *is* the differentiator;
   every answer should land on it (§12.1).

## 12.8 The one-page cheat sheet (memorize these 20 facts)

1. VLM = ViT → connector → LLM over [P image tokens, L text tokens]. (Ch 1)
2. One head, text out: detection/grounding is text (or box tokens) generation. (Ch 1)
3. CLIP: dual towers, InfoNCE, in-batch negatives, learned τ; zero-shot = lookup. (Ch 3)
4. SigLIP: sigmoid per-pair → huge-batch-safe, better high-res. (Ch 3)
5. ViT = strided conv into tokens; keep *all* tokens for VLMs; P=(H/14)². (Ch 4)
6. DINOv2: EMA-teacher, crops↔global, dense features, no labels. (Ch 4)
7. 2D/3D RoPE (mRoPE) = layout + video as positions. (Ch 4/8)
8. Connector = space + budget + alignment; linear/MLP keeps P, Q-Former → 32, pool → P/4. (Ch 5)
9. LLaVA: s1 train projector only; s2 instruction-tune with assistant-only loss mask. (Ch 6)
10. Image tokens never get loss weight — they're context, not targets. (Ch 6)
11. Any-res = global + tiles → 576·(1+hw) tokens; budget or die. (Ch 6)
12. LoRA: W+(α/r)BA, r≪d, merge at deploy; QLoRA = 4-bit base + bf16 adapter. (Ch 7)
13. DPO: closed-form RLHF on pairs, ref-anchored; GRPO: group-relative, verifier rewards, no critic. (Ch 7)
14. Prefill = compute-bound; decode = bandwidth-bound (weights+KV per token). (Ch 9)
15. KV/token = 2·layers·kv_heads·head_dim·bytes; GQA and int8 are the levers. (Ch 9)
16. vLLM = paged KV + prefix sharing + continuous batching. (Ch 9)
17. FlashAttention = SRAM tiling + online softmax → O(T) memory, faster. (Ch 9)
18. Spec-decode = draft k, verify in one prefill, distribution-exact, 2–3×. (Ch 9)
19. Vision tokens = the cost: prefill FLOPs + permanent KV tax; prunable mid-network (FastV). (Ch 8/9)
20. AV placement: geometry 10 Hz on-vehicle; semantics 0.1 Hz cloud/edge; data loop around both. (Ch 10)
