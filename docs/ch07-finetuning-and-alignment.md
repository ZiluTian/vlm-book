# Chapter 7 — Fine-tuning and Alignment, Piece by Piece

> 🎯 **The one sentence:** SFT is masked next-token training on instruction data;
> LoRA/QLoRA make that cheap; DPO/GRPO then push the model's *distributions* toward
> preferred answers — with GRPO being the 2024–26 recipe that turned "thinking"
> models on (and now turns multimodal reasoning on).

## 7.1 SFT in one mechanism

Everything in SFT is §6.6: build sequences, mask all but the assistant spans,
cross-entropy. The only "new" engineering is **packing** — many short samples into
one long sequence to keep the GPU saturated:

```python
def pack(samples, max_len):
    # samples: list of (h, target) pairs, each already tokenized+masked
    seq, tgt = [], []
    for h, target in samples:
        if len(seq) + len(h) > max_len: break
        seq += list(h);  tgt += list(target)
        seq += [EOS];     tgt += [EOS]           # EOS between samples = boundary
    return seq[:max_len], tgt[:max_len]          # attend only within one sample
                                          # (segment-level attention mask — don't skip this)
```

Two failure modes: (1) forgetting the **segment attention mask** → samples attend to
each other and quality quietly rots; (2) EOS boundaries missing → the model learns
to run samples together. Both are data-pipeline bugs, not model bugs.

**Check:** why not just use a batch of short sequences instead of packing? →
*GPU utilization: short sequences waste the context budget; packing turns many
short tasks into few long ones (more work per kernel launch, better FLOPs/GPU-sec).*

## 7.2 LoRA: five lines of math

Freeze `W` [d×k]. Add a low-rank bypass `B·A` (B: [d×r], A: [r×k], r ≪ min(d,k)):

```python
def lora_linear(x, W, A, B, alpha, drop):
    r = A.shape[0]
    return x @ W.T + (alpha / r) * ((drop(x) @ A.T) @ B.T)
# trained: A (init N(0, 0.02²)), B (init 0)  →  delta = 0 at step 0 (safe start)
```

- New params: `r·(d+k)` per matrix vs `d·k` for full. With r=64 and d=k=4096,
  that's 524,288 parameters, or 3.125% of this matrix (not of the whole model).
- Applied to q,k,v,o (and often gate/up/down) of selected layers.
- **Merge at deploy:** `W ← W + (α/r)·B·A` → zero inference overhead. This is the
  line that makes LoRA a *training* technique, not a *serving* one.

**Check:** why initialize B to zero? → *the adapter starts as the identity: day-0
model == base model; no capability cliff before training kicks in.*

## 7.3 LoRA in practice (the knobs that matter)

- **Rank:** 16–64 for narrow skills; 128–1024 when you're really reshaping behavior.
  (Yes, "large-rank LoRA" is a real thing — don't cargo-cult r=8.)
- **Where:** attention projections minimum; +FFN for bigger behavior shifts.
- **α (scaling):** usually α = 2·r; it's a learning-rate-like knob on the adapter.
- **When full-FT wins:** you have compute + your data is huge + you want max
  quality. LoRA wins on memory, speed, and *keeping many adapters* (one base,
  N LoRAs for N domains — hot-swappable at serving time).

### Worked example: what LoRA saves, and what it doesn't

**Why:** fewer trainable parameters do not guarantee a long multimodal batch
fits. The frozen backbone still runs, and gradients must pass through it to the
adapters and any trainable projector.

**What:** assume 7B parameters, BF16 weights and gradients, FP32 Adam moments,
and no FP32 master weights. Full fine-tuning needs `7B × (2 + 2 + 8) = 84 GB`
for model state before activations and temporary buffers. With 20M adapter
parameters, LoRA instead needs `14 GB + 20M × 12 bytes ≈ 14.24 GB` under the
same assumptions. If you double the number of image tokens, those model-state
numbers stay the same, but saved activations and attention work still grow.

**How:** count the parameters that actually require gradients, including the
connector, and profile peak allocation on a batch near your real token cap.
If peak usage grows mainly with image and text sequence length, try activation
checkpointing or a smaller microbatch before reducing LoRA rank. If the frozen
weights dominate even on short batches, quantizing or sharding the backbone
addresses a different part of the budget. Compare task accuracy and text-only
regressions at equal data exposure before choosing a configuration.

## 7.4 QLoRA: quantize the base, train the adapter

```
base W  ──► frozen 4-bit NF4 weights, dequantized in blocks for compute
adapter ──► trainable LoRA branch operating on higher-precision activations
memory  ──► 65B fine-tuning demonstrated on one 48 GB GPU in the QLoRA paper
```

The base is frozen at 4-bit; only the bf16 adapter trains. Quality ≈ full-FT for
many SFT tasks at a fraction of the memory. NF4 (normal Float 4) is a
quantization that respects the parameter distribution (most weights are
~normal) — the detail that makes 4-bit work without a quality collapse.
The [QLoRA paper](https://arxiv.org/abs/2305.14314) also quantizes the scales
and uses paged optimizers to handle memory spikes; 4 bits per base parameter
alone is not a peak training memory estimate.

**Check:** where does QLoRA's quality actually come from, given 4-bit weights?
→ *the gradients never touch the base — only the bf16 adapter is optimized;
quantization noise is static and small enough that a good adapter compensates.*

## 7.5 The recipe table (what moves, what doesn't)

| Stage | ViT | Connector | LLM | Typical data |
|---|---|---|---|---|
| pretrain-alignment (LLaVA s1) | frozen | **train** | frozen | image–caption pairs |
| instruction tuning (s2) | frozen | **train** | **train** (FT or LoRA) | instruction/chat |
| domain SFT (your AV data) | frozen (or light FT) | **train** | LoRA/FT | your (image, text) tasks |
| RL (DPO/GRPO) | frozen | frozen | **train** (policy) | prompts + reward/verifier |

The pattern: **the newer the component, the more you're allowed to move it**;
the older, the more you protect it.

**Check:** why is it dangerous to full-FT the ViT during RL? → *the visual features
drift while the reward model (often built on the same features) judges the model —
a moving ruler. Keep the perception side fixed during alignment.*

## 7.6 Learning rates and epochs — the silent killers

- Full-FT LLM: 5e-6–2e-5. LoRA: 1e-4–3e-4. ViT (if touched at all): ≤1e-5.
- 1–3 epochs for SFT. Past that you're usually *memorizing* the instruction set,
  not learning it — and memorization shows up as: worse generalization, more
  repetition, style collapse ("As an AI…").
- Watch **text-only eval** every epoch. The first drop is your overfit alarm.

**Check:** SFT val loss is falling but text-only bench is dropping — diagnosis?
→ *overfitting to the (probably synthetic) SFT distribution; stop, lower LR,
mix in base-style data, or switch to LoRA with fewer steps.*

## 7.7 RLHF: reward model + PPO (the 60-second version)

```
1. Collect: for a prompt, sample K model answers
2. Humans rank them  (or a strong LLM does)
3. Train RM: score(prompt, answer)  →  binary ranking loss over pairs
4. PPO: maximize  E[RM(x,y)]  −  β·KL(π ‖ π_ref)
   policy = the LLM;  π_ref = the SFT model (KL keeps it from drifting)
```

PPO works and was the original recipe (InstructGPT), but: four models in memory
(policy, ref, RM, critic), reward hacking is common, and it's hyperparameter hell.
Which is why the field moved to DPO and then GRPO — next two sections.

**Check:** what does the KL term *buy*? → *it bounds how far the policy can wander
from SFT: you're shaping a distribution, not inventing a new one. Remove it and
the policy chases the RM's blind spots until the text rots.*

## 7.8 DPO: RLHF without the RL

DPO rewrites the PPO objective in closed form: with the right algebra, the
KL-constrained optimum has a known form, so you can **skip the RM and the critic
and train directly on preference pairs**:

```python
def dpo_loss(policy, ref, x, y_win, y_lose, beta=0.1):
    # x: prompt; y_win / y_lose: preferred / dispreferred completions
    lp_w = log_softmax_x(policy(x), y_win)    # log-prob of the whole answer
    lr_w = log_softmax_x(ref(x), y_win)
    lp_l = log_softmax_x(policy(x), y_lose)
    lr_l = log_softmax_x(ref(x), y_lose)
    return -log_sigmoid(beta * ((lp_w - lr_w) - (lp_l - lr_l)))
```

Read it: "raise the *relative* (vs ref) log-prob of the winner, lower the loser's."
No sampling loop, no RM, no critic — two forward passes, stable, good-enough.
Cost: you inherit the preference data's quality; garbage pairs → garbage alignment.

**Check:** why keep the `ref` model in the loss at all? → *it's the KL term in
disguise: without it the model can satisfy "win > lose" by collapsing both to
high-prob junk. ref anchors the distribution.*

## 7.9 GRPO: the R1 recipe (group-relative, no critic)

The DeepSeek-R1 move: **you don't need a learned reward or a critic if your reward
is verifiable** (math answer correct? format valid? tool call succeeded?). Sample a
*group* of G answers per prompt, score each, use within-group ranking as the
advantage:

```python
def grpo_step(policy, ref, prompt, reward_fn, G=8, beta=0.001):
    ys = [sample(policy, prompt) for _ in range(G)]
    R  = [reward_fn(prompt, y) for y in ys]            # e.g. 1.0/0.0 + format bonus
    adv = (tensor(R) - mean(R)) / std(R + 1e-8)        # group-relative advantage
    loss = 0
    for y, a in zip(ys, adv):
        ratio = exp(logp(policy, y) - logp(ref, y))     # per-token, summed
        loss -= a * ratio
        loss += beta * kl(policy, ref)                  # keep the anchor
    return loss / G
```

Why it won: one less model (no critic), no learned RM (the *verifier* is the
reward — a unit test for the answer), and the group normalization makes the
advantage scale-free across prompts. This is how "reasoning" became a *trainable
behavior*: reward correct *process+outcome* → the model learns to show work.
Same machinery, now pointed at *visual* verifiers, is what multimodal reasoning
models (Ch 8 §8.4) are built from.

**Check:** what's the failure mode of verifiable-reward RL? → *rewards that are
cheap to game: format gaming, degenerate-but-correct answers, and (in vision)
reward hacking when the verifier sees what the model can't. Verifier and policy
must be matched in perception.*

## 7.10 Alignment for multimodal (where it's still young)

- **Grounding:** reward answers that point to the right region (boxes in output →
  IoU verifier → GRPO-style loop). This is the visual analog of the math verifier.
- **Hallucination reduction:** penalize objects not present (POPE-style pairs as
  preference data; "the image has X" vs "it does not" — Ch 11).
- **Interleaved reasoning:** the model emits `<think>` text, then *tool calls*
  (crop, zoom, OCR), then more thinking — trained with GRPO on verifiable tasks.
  The VLM becomes an agent that inspects its own image. (Ch 8 §8.4.)

**Check:** why is multimodal RL harder than text RL? → *the verifier is the hard
part: "correct" in vision needs perception (boxes, OCR, video consistency) that
itself can be wrong. In text, a unit test settles it.*

## 7.11 ❓ Interview

**Q: LoRA math, on the whiteboard.** — §7.1–7.2: W + (α/r)BA, r ≪ d, B init 0,
merge at deploy, ~0.1–1% params. Then say which matrices you'd apply it to.

**Q: Full FT vs LoRA vs QLoRA — how do you choose?**
— Compute × data × goal. Max quality + big data + compute: full FT. One-off domain
adaptation: LoRA. 70B-class on one GPU: QLoRA. Many domains, one base: LoRA zoo.
Always: freeze the ViT unless you have a reason (§7.5).

**Q: DPO vs PPO — when would you still use PPO?**
— When you have *online* signal PPO can use that DPO can't express: a reward model
  over live user behavior, or you need to sample *new* behavior (DPO only reshapes
  pairs you already have). Otherwise DPO/GRPO: simpler, cheaper, stable.

**Q: GRPO in one breath, and its prerequisite?**
— Sample a group per prompt, score with a *verifier*, rank within group, train with
  ratio + KL anchor; no critic, no learned RM. Prerequisite: a reward you can
  compute programmatically (answer-correct, format-valid, IoU, tool-success).

**Q: How would you fine-tune a 7B VLM on 50k AV (image, caption+QA) pairs, 4×H100?**
— ViT frozen; connector full-train; LLM LoRA r=128–256 (attention+FFN) @ ~1e-4;
  pack to 8k context; 2 epochs; eval: your private holdout *and* a text-only bench
  every epoch (regression alarm); then DPO/GRPO pass if you have a verifier
  (e.g., "does the answer match the ground-truth event label?").
