# Glossary

Fast lookup. "Ch N" = where it's explained in this book.

## The stack

| Term | Meaning |
|---|---|
| **VLM / MLLM** | Vision-language model / multimodal LLM: pixels+text in, text out. Ch 1 |
| **Family A/B/C/D** | contrastive two-tower / encoder-connector-decoder / native multimodal / image-generative. Ch 1 |
| **Vision encoder** | ViT (or DINOv2/SAM) producing P patch features. Ch 4 |
| **Patchify** | Splitting image into 14×14 patches; a stride-14 conv. Ch 4 §4.2 |
| **Connector / projector** | Maps ViT features into LLM space: linear, MLP, Q-Former, resampler, pool. Ch 5 |
| **Q-Former** | 32 learned queries cross-attending image features → fixed 32 tokens (BLIP-2). Ch 5 §5.4 |
| **Perceiver resampler** | Query-attention latents injected into LLM at several depths (Flamingo). Ch 5 §5.6 |
| **Vision tokens** | The P vectors that stand in for the image in the LLM sequence. Ch 1 §1.6 |
| **Native multimodal** | One model pre-trained on interleaved data, pixels tokenized, no bolt-on encoder. Ch 8 §8.1 |
| **VLA** | Vision-Language-Action: VLM + action output (waypoints), trained with driving rewards. Ch 10 §10.5 |
| **World model** | Video/occupancy generation conditioned on (state, action); sim + counterfactuals. Ch 10 §10.7 |

## The LLM core

| Term | Meaning |
|---|---|
| **BPE** | Byte-pair encoding: the tokenizer; vocab V ≈ 32k–150k. Ch 2 §2.2 |
| **Embedding** | token id → D-dim vector. Ch 2 §2.4 |
| **Pre-LN** | normalize before each sub-block; stable deep training. Ch 2 §2.5 |
| **RoPE** | rotary position encoding: relative position inside q·k. Ch 2 §2.7 |
| **2D-RoPE / mRoPE** | positions as (h,w) / (t,h,w) tuples; variable resolution + video. Ch 4 §4.6, Ch 8 §8.2 |
| **GQA** | grouped-query attention: n_kv_heads < n_heads → smaller KV cache. Ch 2 §2.8 |
| **SwiGLU** | gated FFN (up × sigmoid-gate); ~8/3·D hidden. Ch 2 §2.9 |
| **Next-token CE** | the training loss; p(token_t+1 | ≤t). Ch 2 §2.11 |
| **Loss masking** | weight 0 on non-target positions (image tokens, prompts, user turns). Ch 6 §6.6 |
| **Teacher forcing** | train on ground-truth prefixes; no self-exposure. Ch 2 §2.11 |
| **KV cache** | reused key/value tensors per position; makes decode O(1) new compute. Ch 2 §2.13 |
| **Prefill / decode** | parallel input pass (compute-bound) / 1-token steps (bandwidth-bound). Ch 2 §2.12 |
| **EOS** | end-of-sequence token; decode stop. Ch 2 §2.12 |

## Pretraining & encoders

| Term | Meaning |
|---|---|
| **InfoNCE** | contrastive loss: positive vs negatives in the softmax. Ch 3 §3.4 |
| **In-batch negatives** | the other B−1 pairs in the batch are the negatives. Ch 3 §3.3 |
| **Learned temperature** | τ (logit scale) co-tuned with batch size; sharpness knob. Ch 3 §3.5 |
| **SigLIP** | sigmoid per-pair contrastive; no batch coupling; high-res friendly. Ch 3 §3.8 |
| **Zero-shot classification** | match image embedding against text-embedding class anchors. Ch 3 §3.9 |
| **Prompt ensemble** | average many text templates per class; +3–5 pts. Ch 3 §3.10 |
| **DINOv2** | self-distilled dense ViT (EMA teacher, crops↔global, centering/sharpening, registers). Ch 4 §4.7 |
| **Register tokens** | learned "junk" slots absorbing attention artifacts. Ch 4 §4.4 |
| **SAM / SAM2** | segmentation encoder (Hiera) + promptable mask decoder; SAM2 adds video memory. Ch 4 §4.10 |
| **OpenCLIP / LAION** | open CLIP-style pretraining on ~400M web pairs. Ch 3 §3.1 |

## Training & alignment

| Term | Meaning |
|---|---|
| **SFT** | supervised fine-tuning: masked next-token on instruction data. Ch 7 §7.1 |
| **Packing** | many samples into one long sequence (with segment attention masks). Ch 7 §7.1 |
| **Chat template** | byte-exact role/turn markup; must match train and serve. Ch 6 §6.12 |
| **LoRA** | low-rank bypass W+(α/r)BA; r≪d; merge at deploy. Ch 7 §7.2 |
| **QLoRA** | 4-bit (NF4) frozen base + bf16 LoRA; 65B on one GPU. Ch 7 §7.4 |
| **RLHF** | reward model + PPO + KL anchor. Ch 7 §7.7 |
| **DPO** | direct preference optimization: closed-form RLHF on (win, lose) pairs. Ch 7 §7.8 |
| **GRPO** | group-relative policy optimization: verifier rewards, within-group advantage, no critic. Ch 7 §7.9 |
| **Reward hacking** | policy gaming the reward's blind spots. Ch 7 §7.9 |
| **Verifier** | programmatic reward (answer-correct, IoU, format) enabling GRPO. Ch 7 §7.9 |
| **Catastrophic forgetting** | SFT/RL eroding base capability; watched via text-only evals. Ch 7 §7.6 |

## Serving & efficiency

| Term | Meaning |
|---|---|
| **Bandwidth-bound** | decode speed set by HBM bytes/token, not FLOPs. Ch 9 §9.2 |
| **KV/token** | 2·layers·kv_heads·head_dim·bytes — the per-token memory tax. Ch 9 §9.3 |
| **Paged KV** | block-allocated KV with per-request block tables (vLLM). Ch 9 §9.4 |
| **Prefix sharing / caching** | identical input prefixes share physical KV blocks. Ch 9 §9.4, §9.7 |
| **Continuous batching** | per-step request admission/exit; throughput↔p99 dial. Ch 9 §9.5 |
| **FlashAttention** | SRAM-tiled attention, online softmax, O(T) memory. Ch 9 §9.6 |
| **Speculative decoding** | draft k tokens, verify in one pass; distribution-exact. Ch 9 §9.8 |
| **W4A16 / W8A8 / int8 KV** | serving quantization menu: weights-only, activations, cache. Ch 9 §9.9 |
| **Token diet / pruning** | cutting vision tokens (FastV: mid-net attention; pyramidal; KV compress). Ch 8 §8.3–8.5 |
| **FastV** | prune low-attention vision tokens at mid-network; ~2–3× decode. Ch 8 §8.4 |
| **Dynamic / any-res tiling** | global + tile tokens; 576·(1+hw); adaptive grid. Ch 6 §6.13 |

## Driving & evaluation

| Term | Meaning |
|---|---|
| **Data engine (Role A)** | VLM mining/naming/fleet clips → long-tail catalog. Ch 10 §10.3 |
| **Language supervision (Role B)** | text intermediates (DriveGPT/DriveLM/Senna) improving E2E. Ch 10 §10.4 |
| **VLM-as-policy (Role C)** | EMMA/OpenEMMA: image+state → trajectory+explanation. Ch 10 §10.5 |
| **Hybrid monitor (Role D)** | real-time BEV drives; async VLM explains/ODD-checks. Ch 10 §10.6 |
| **ODD** | operational design domain; VLM zero-shot OOD/ODD detection. Ch 10 §10.3 |
| **TTC** | time-to-collision; the deterministic metric layer under VLM judges. Ch 10 §10.10 |
| **SOTIF / ISO 26262** | safety-of-intended-function / functional safety; why VLM judges rank, metrics decide. Ch 10 §10.10 |
| **MMBench / MMMU** | general / reasoning VLM benchmarks (MC floor, contamination). Ch 11 §11.2 |
| **POPE / CHAIR / MMVP** | hallucination probes: presence QA / caption audit / minimal pairs. Ch 11 §11.6 |
| **Hallucination (types)** | object / attribute / spatial / text / prior-driven. Ch 11 §11.4 |
| **LLM-as-judge** | strong-model rubric scoring; self/verbosity/position biases; calibrate on humans. Ch 11 §11.7 |
| **Contamination** | bench items in training data; detect via recency + minimal pairs. Ch 11 §11.3 |
| **Stratified eval** | private test set split by ODD/scene/severity; the product contract. Ch 11 §11.8 |
