# Chapter 2 — The LLM Core, One Piece at a Time

> 🎯 **The one sentence:** a decoder LLM is a sequence of pre-LN transformer layers
> over embedded tokens, trained with masked next-token cross-entropy, and decoded
> one token at a time using a KV cache. Build it below, piece by piece.

## 2.1 A token is just an index

The whole model eats integers. "The quick fox" → `[392, 3287, 1094]`. That's it —
an LLM is a function `int → probability over ints`, run repeatedly.

**Check:** what's the vocab size doing? → *it sets the width of the final head:
logits are `[T, V]`, V ≈ 32k–150k.*

## 2.2 BPE, in thirty seconds

Byte-Pair Encoding: start from raw bytes, repeatedly merge the most frequent
adjacent pair into a new token, until you have V tokens. Frequent words = 1 token;
rare words = several tokens; nothing is ever un-learnable.

**Check:** is "autonomous" one token? → *probably not — you'd see it split like
`auto` `nomo` `us`. Tokenization is greedy and frequency-driven, not linguistic.*

## 2.3 An image is *not* a BPE token (in the main family)

Pixels never go through BPE in encoder-connector models. Instead:
ViT → P feature vectors → each vector **behaves like one token** in the sequence.
(Discrete VQ image tokens exist in Family C / image-gen models — §8.5 — but the
interview default is: image = P soft vectors, not P integers.)

**Check:** so where do the image's "ids" come from? → *there are none — the vision
encoder's output vectors *are* the embedded tokens, skipping the lookup table.*

## 2.4 Embedding: index → vector

```python
def embed(ids):            # ids: [B, T]
    return nn.Embedding(V, D)(ids)   # -> [B, T, D]
```

One `[D]` vector per token. For image tokens, this step is *skipped* — the
connector's output lands here directly (§1.3). Two sources feed the same sequence.

## 2.5 The layer skeleton: pre-LN + residual

Every layer is the same shape:

```python
def layer(x):              # x: [B, T, D]
    x = x + attn(ln1(x))   # attention path, normalized first
    x = x + ffn(ln2(x))    # FFN path, normalized first
    return x
```

"Pre-LN" = normalize *before* the sub-block. Why? Gradients stay well-scaled at
depth; you can train 100 layers without warmup tricks. (You've hit this exact
problem in deep CNNs — it's the same disease, same cure.)

**Check:** what does the residual line buy you? → *a clean highway for gradients,
and the identity "this layer just adds a correction."*

## 2.6 Attention: the only part worth writing out

```python
def attn(q, k, v):         # q: [B, T, D]; k, v: [B, T_total, D_kv]
    # causal: position i may see only positions <= i
    s = q @ k.T / sqrt(head_dim)          # [B, T, T_total]
    s = causal_mask(s)
    return softmax(s) @ v                 # [B, T, D_kv]
```

That's the entire "magic": **weighted average of value vectors, with the weights
coming from a q·k similarity.** Everything else in an LLM is plumbing around this.

🔗 **BEV bridge:** you've implemented this exact operation as BEVFormer's spatial
cross-attention — there, BEV tokens were the queries. Here, every token is a query.

**Check:** why divide by `sqrt(head_dim)`? → *keeps the dot products in a range
where softmax has gradient; without it, deep heads go to one-hot and die.*

## 2.7 RoPE: put position *inside* the dot product

No learned position vectors. Instead, rotate q and k by an angle that grows with
position, so that `q_i · k_j` depends on the *difference* `j − i` — exactly like a
relative positional encoding, computed for free:

```python
def rope(x, pos):         # x: [B, T, D]; pos: int, the position of this batch of tokens
    # split D into pairs; rotate pair m by angle pos * theta_m
    return rotate_pairs(x, theta=1/10000 ** (arange(0, D, 2)/D)) * ...
```

Two consequences you must know:
1. **No position table to run out of** → modest length extrapolation for free.
2. **Order is load-bearing.** Permuting tokens changes every dot product. Spatial
   layout in images must be *encoded into positions* (2D/3D RoPE — Ch 4, §8.2).

**Check:** absolute or relative? → *relative: the encoding sees `j − i`, not `i`
and `j` separately.*

## 2.8 GQA: shrink the KV cache without killing quality

MHA: every query head has its own k/v heads → KV cache = `n_heads` of them.
MQA: one shared k/v pair → cache tiny, quality drops.
**GQA: groups of query heads share a k/v head → `n_kv_heads` < `n_heads`, e.g. 32 → 8.**

```python
# per-token KV cache size = 2 * n_layers * n_kv_heads * head_dim * bytes
# 8B GQA-8 model, fp16:  2 * 32 * 8 * 128 * 2  =  128 KB / token
# same model, MHA-32:    2 * 32 * 32* 128 * 2  =  512 KB / token
```

4× less cache memory, small quality cost. This is *the* KV-memory knob.

**Check:** if you serve 1k users × 8k context on an 8B GQA model, how much KV
memory? → *1000 × 8000 × 128 KB ≈ 1 TB. (You'll see why serving needs tricks: Ch 9.)*

## 2.9 The FFN: SwiGLU

```python
def ffn(x):                # x: [B, T, D]
    up   = x @ W_up.T      # [B, T, D_ff]
    gate = x @ W_gate.T
    return (gate * sigmoid(gate)) * up @ W_down.T
```

The FFN is where most parameters live (~2/3 of the model). SwiGLU's gated form is
slightly better per-param than a plain 2-layer MLP; `D_ff ≈ (8/3)·D`, so your
FLOPs estimate needs that 8/3, not 2.

**Check:** which holds more params, attention projections or the FFN? → *the FFN,
roughly 2:1.*

## 2.10 Assemble it: the full forward

```python
def gpt_forward(h, cache=None):
    # h: [B, T, D]  (embedded text AND/OR vision tokens)
    for i, blk in enumerate(layers):        # N blocks
        h = blk(h, cache[i])                # adds new k/v into cache[i]
    return lm_head(final_ln(h))             # [B, T, V]
```

That's the entire model. If you can narrate §2.5–§2.9 on the whiteboard, you
understand the LLM half of every VLM you'll interview for.

## 2.11 The training loss: next-token CE, with masking

```python
def train_step(ids, loss_mask):
    # loss_mask: [B, T], 1 = supervise this position, 0 = don't
    logits = gpt_forward(embed(ids))        # [B, T, V]
    loss = cross_entropy(logits[:, :-1], ids[:, 1:], reduction="none")  # [B, T-1]
    return (loss * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum().clamp(min=1)
```

Two facts to keep straight:
- **Teacher forcing:** training always conditions on ground-truth prefixes — no
  sampling, no exposure to the model's own mistakes (hence the RL stages, Ch 7).
- **Masking is the technique.** In SFT you supervise only the assistant's reply;
  image tokens and user turns get 0. Mask it wrong and the model learns to
  "parrot the image." (This one line of code is Ch 6's heart.)

**Check:** why not supervise the prompt tokens? → *the model is being trained to
answer, not to reproduce; weighting the prompt would distort the distribution it
should condition on.*

## 2.12 Inference: two regimes, one cache

```python
def generate(h0, max_new):
    cache = new_caches()
    logits = gpt_forward(h0, cache)         # PREFILL: S tokens, parallel, compute-bound
    for _ in range(max_new):
        t = sample(logits[:, -1])
        if t == EOS: return
        logits = gpt_forward(embed(t), cache)   # DECODE: 1 token, memory-bound
```

| | prefill | decode |
|---|---|---|
| tokens | S at once | 1 at a time |
| bound by | FLOPs | memory bandwidth (weights + KV reads) |
| attention cost | O(S²·D) | O(T_total·D) |

The same model is a *compute* problem once and a *bandwidth* problem the other
time. Serving (Ch 9) is almost entirely about these two rows.

## 2.13 The KV cache, in one diagram

```
step 0 (prefill):  K,V of tokens [1..S]      → cache [S, D_kv]
step 1:            K,V of token [S+1]        → cache [S+1, D_kv]
step 2:            K,V of token [S+2]        → cache [S+2, D_kv]
...
attention at step n: q_n · K[1..S+n]  →  only NEW q is computed; K,V are REUSED
```

Nothing is recomputed. That's the win — and that's why the cache is the model's
*working set*, and serving is a cache-management problem.

🔗 **BEV bridge:** this is your temporal BEV cache — except it grows without bound
until the context window closes, so it's a memory-planning problem, not a fixed buffer.

## 2.14 The two numbers you must estimate out loud

**FLOPs/token ≈ 2 · N_params** (each weight does one multiply-add per token),
plus the O(T_total·D_kv) KV read. 7B model → ~14 GFLOP/token.

**Memory:**
- weights: `N × bytes/param` (fp16: 14 GB for 7B; int4: ~4 GB)
- KV/token: `2 · n_layers · n_kv_heads · head_dim · bytes` (§2.8's formula)

Do both in the interview. If you slip a factor of two (forget K *and* V),
say so and fix it out loud — self-correction reads as competence.

**Check:** a 70B model, 80 layers, 8 kv heads, head_dim 128, fp16, 32k context —
KV memory per user? → *2·80·8·128·2 B = 320 KB/token → ×32k ≈ 10 GB. That's your
answer, with arithmetic shown.*

## 2.15 Where CV intuition breaks

1. **No pooling pyramid.** Token count is a budget you set; every dropped patch is
   dropped detail, forever (after prefill).
2. **Receptive field = context length**, metered in memory.
3. **No heads.** No box regressor, no NMS. "Detection" is text — or special tokens
   at the frontier (§8.4). Coordinates are tokens, not floats.
4. **Order is geometry.** Spatial info must go into position encodings.
5. **The image is only its P tokens.** Drop tokens → drop detail, instantly (§8.3).

## 2.16 ❓ Interview

**Q: Pre-LN vs post-LN?** — Pre-LN normalizes the input of each sub-block; stable
at depth, no warmup gymnastics; the residual stream stays clean.

**Q: MHA vs MQA vs GQA?** — Number of k/v heads: full, shared-by-all, or
grouped. The knob you turn is KV-cache memory vs quality; GQA is the practical middle.

**Q: Why decode autoregressively instead of predicting the answer in parallel?**
— `p(y₁..yₙ) = ∏ p(y_t | y_<t)` is what one loss (next-token CE) gives you for free;
variable-length, open-ended output; and the same factorization supports agents/tools.
Add for depth: image *generation* does go parallel (diffusion) — see §8.5.

**Q: Why RoPE instead of learned positional embeddings?** — No table to run out of;
relative geometry baked into q·k; cheap; extrapolates modestly.

**Q: What breaks if you remove the KV cache?** — Decode becomes O(T²) per step
(recompute attention every token) — nobody does this; the cache is *the* inference trick.
