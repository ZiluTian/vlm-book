# Chapter 11 — Evaluation: Benchmarks, Hallucination, and Building Your Own

> 🎯 **The one sentence:** public VLM benchmarks saturate and leak, so the real
> evaluation skill is building a *private, task-specific* eval — with a rubric,
> a judge, and a metric layer — and knowing when a benchmark number is lying to you.

## 11.1 Why VLM eval is harder than LLM eval

An LLM eval is (input text → judge the text). A VLM eval adds three failure axes
the LLM eval can't see:
1. **Perception failures** — the model *saw* the wrong thing (wrong color, missed
   object, misread text).
2. **Grounding failures** — it saw right, *pointed wrong* (spatial errors: left/right,
   "behind", distances).
3. **Modality-gap failures** — it answered from *language priors* instead of the
   image (the image is irrelevant to the answer).

Your eval design must separate these, or "the model got it wrong" tells you nothing
about *what to fix*. (Fix for #1 = data/resolution; #2 = positions/architecture;
#3 = training/alignment. Three different engineering programs.)

**Check:** a model says "there is a red car" and the image has a red *truck* —
which failure class? → *perception (#1, class confusion) — and it's the one a
text-only judge often marks as "plausible." Hence the image-aware judge, §11.8.*

## 11.2 The public benchmark landscape (orientation table)

| Benchmark | Measures | Quirk / how it's gamed |
|---|---|---|
| MMBench / MMBench-V | general ability, 20+ sub-areas | multiple-choice → guessable; language-prior leakage |
| MMMU | college-level multimodal reasoning (charts, science) | OCR + reasoning mix; chart-reading dominates |
| OCRBench / DocVQA / ChartQA | fine text in images | the "can it read small text" probe — very task-specific |
| VideoMME / MVBench | video understanding | long-video attention + temporal reasoning |
| HallusionBench | visual *illusion* traps (perception vs logic) | measures the perception/logic gap specifically |
| POPE / CHAIR / MMVP | hallucination (object presence, details) | POPE = binary "is X present?"; MMVP = minimal pairs (one word differs) |
| OpenGrounding / RefCOCO | grounding accuracy (IoU) | the metric layer for "where" |
| (Driving) DriveLM / LingoQA / NuPrompt / BDD-X | scene QA, language-supervised driving | small, domain-specific — see Ch 10 |

Rule of thumb: **multiple-choice benchmarks measure recall + priors; open-ended
benchmarks measure generation quality; grounding benchmarks measure geometry.**
Quote a number only after naming which of the three it is.

**Check:** why does a multiple-choice VLM benchmark overestimate real ability?
→ *4-way guessing gives a 25% floor; and the model can pick from *language* alone
when the options are well-formed. Open-ended + image-verified is the honest metric.*

## 11.3 Saturation and contamination — the two silent killers

- **Saturation:** top models cross 85%+ on general benches within a year; the
  benchmark stops differentiating. (This is why 2025–26 moved to harder,
  reasoning-heavy, and *private* evals.)
- **Contamination:** bench items leak into training data (web-scraped). A model
  that "knows" the answer from pretraining is not demonstrating the capability.
  Detection: **held-out / recency** items (created after the model's cutoff),
  minimal-pair edits (MMVP-style: flip one attribute — a memorized answer stays
  fixed while a *perceiving* model tracks the change).

**Check:** which eval design is contamination-proof by construction? → *anything
generated *after* the cutoff for your model, or private fleet/sim data — i.e.,
§11.5's custom set. Public benches are trend indicators, not proofs.*

## 11.4 Hallucination: the types you must name

1. **Object hallucination** — claims a present/absent object ("a dog is walking"
   when there is none).
2. **Attribute errors** — wrong color/size/count of a *real* object.
3. **Spatial errors** — left/right, before/after, near/far flips.
4. **Text hallucination** — invented text in the image (fake signs, fake plates).
5. **Prior-driven answers** — the image is *irrelevant* to the correct answer;
   the model answers from web priors ("What's on the sign?" → a generic sign).

Types 1–2 are the POPE/CHAIR territory; 3 is grounding; 4 is OCR+hallucination;
5 is the modality-gap axis from §11.1.

**Check:** why are *absent* objects the hardest case (POPE's "No" answers)?
→ *the model's prior says "probably yes" (objects co-occur with the question's
context); answering "No" requires the image to *actively veto* the prior — a
harder computation than confirming it.*

## 11.5 Why models hallucinate (the mechanisms, briefly)

- **Weak visual grounding in mid/layers:** by the time the LM "decides," the image
  tokens may be low-attention (literally measurable — the same signal FastV uses,
  §8.4). The answer is then ~text prior.
- **Token competition:** the LM head's distribution is dominated by high-frequency
  image-relevant words ("car", "road") — a prior bias baked into the vocab.
- **No "I can't see" training signal:** web captions never say "the sign is too
  blurry to read"; the model wasn't taught *uncertainty*, so it confabulates.
- **Decoding amplifies:** temperature/top-p sampling can pick the second-best
  (prior) token where the image evidence is thin.

**Check:** which mechanism does *token pruning* (FastV) risk making worse?
→ *#1 — if you prune the very vision tokens that carried the thin evidence,
the prior wins even more. Pruning and hallucination are the same dial, cut
both ways.*

## 11.6 Measuring it: POPE and CHAIR, in two lines

- **POPE:** binary QA — "Is there a {object} in the image?" over present/absent
  objects; report accuracy **separately** for the "Yes" and "No" splits (the
  "No" split is the honest number).
- **CHAIR:** caption-based. `CHAIR_i` is the fraction of mentioned object
  instances that are hallucinated; `CHAIR_s` is the fraction of captions with
  at least one hallucinated object. Neither measures omission; track recall
  separately if missing objects matters.

**Check:** POPE accuracy 92% overall but 84% on the "No" split — what's the story?
→ *the model says "yes" to things that aren't there (prior-driven #5/#1). Overall
hides it; the split exposes it. Always report the split.*

## 11.7 LLM-as-judge (and its biases)

The 2024–26 standard for open-ended VLM eval: a strong model scores answers
against a rubric + the image.

```python
def judge(answer, image, rubric, ground_truth=None):
    return strong_vlm.ask(
        f"IMAGE: {image}\nANSWER: {answer}\nGROUND TRUTH: {ground_truth}\n"
        f"Score 1-5 per criterion: perception_accuracy, grounding, faithfulness, completeness.\n"
        f"Return JSON: {{perception: .., grounding: .., ...}}",
    )
```

Known biases (name at least two in an interview):
- **Self-preference** — the judge favors answers in *its own* style (use a judge
  *different* from the model under test where possible).
- **Verbosity bias** — longer answers score higher unless the rubric says otherwise.
- **Position bias** — in pairwise judging, order of the two answers sways the call
  (swap order, average).
- **Image-attention decay** — the judge also glances at the image; for fine
  perception, feed it *crops* (your detectors as the judge's tools — the §8.6 loop).

**Check:** how do you calibrate a VLM judge? → *a 100–300 item human-labeled subset:
correlate judge scores vs human scores (Spearman), weight rubric criteria to match,
re-check monthly. Judge is a *proxy* — measure the proxy's error.*

## 11.8 Building your own eval set (the recipe, driving-flavored)

```
1. SCOPE:  the task ("explain the ego maneuver" / "detect ODD events") — one per set
2. ITEMS:  200–1000 clips/frames, stratified by ODD (weather, time, scene, severity)
3. LABELS: ground truth written by humans (2 annotators, adjudicated) —
           the *only* part you can't automate
4. RUBRIC: 3–5 scored criteria (perception, grounding, faithfulness, completeness)
5. JUDGE:  VLM judge (§11.7) calibrated on 10% of the set
6. SPLIT:  train/val/test discipline — the eval set NEVER touches training data
7. FREEZE: version it (eval_v3), keep past versions (regression history)
8. REPORT: per-criterion + per-ODD-stratum + the hallucination splits (§11.6)
```

The two things that make it *yours*: stratification (you know the ODDs — §10.3)
and the metric layer (TTC, IoU, comfort — the deterministic grounding under the
judge, §10.10). Public bench + this = a defensible eval story.

**Check:** why stratify by ODD instead of random sampling? → *random sampling
overweights the common 99% (where any model looks fine) and underweights the
long tail (where your product lives). Stratification forces the tail into the score.*

## 11.9 Regression discipline (the CV engineer's superpower)

You've run ablations for years; apply the same hygiene to VLMs:
- **Track text-only ability every SFT/RL epoch** — multimodal gains often come with
  text regressions (the Ch 7 §7.6 alarm).
- **Ablate the *pipeline*, not just the model:** template, resolution, pooling,
  temperature, judge version — each is a "hyperparameter" with its own curve.
- **Two-model sanity check:** if model A beats model B on your set *and* on a public
  bench *and* doesn't regress text-only — you can trust the delta. One axis → suspect.

**Check:** your new 7B beats the baseline on the driving set but loses 3 points on
MMMU — ship it? → *depends on the product: if the eval set *is* the product, yes
(with the trade documented); if "general intelligence" is a requirement, no — you
found a capability regression the product needs. The eval set should be the
contract, not an afterthought.*

### Worked example: did the model use the image or guess from the prompt?

**Why:** a VLM can answer “yes” to “Is there a person at the crosswalk?” because
the wording predicts the answer in training, not because it saw a person. A
release comparison needs to distinguish that shortcut from genuine visual gains.

**What:** build 100 matched pairs of clips for the *same* question, one with a
clearly visible person in the crosswalk and one with a clearly empty crosswalk.
Keep camera view and scene conditions similar, and review both labels. Suppose
a model gets both answers right on 60 pairs, only the positive right on 30,
only the negative right on 5, and neither right on 5. Its item accuracy is
`(2×60 + 30 + 5) / 200 = 77.5%`. But paired accuracy is **60%**, positive
recall is **90%**, and the false-positive rate is **35%**. The pooled score
alone doesn't reveal that asymmetry. An always-yes baseline gets 50% item
accuracy and 0% paired accuracy.

**How:** compare candidate and baseline on the exact same examples with a frozen
template, preprocessing, and output parser. Use a deterministic scorer for the
binary answers; count abstentions separately and report the coverage rather than
silently dropping them. Keep pairs intact when bootstrapping the candidate-minus-
baseline metric; if clips come from the same drive, resample by independent drive
or by a larger cluster that keeps related clips and pairs together. Report the
false-positive rate and paired accuracy by condition (e.g., glare vs. clear
daylight) with sample counts and uncertainty. Set release thresholds before
seeing the results, and send visually ambiguous cases to human review instead
of treating the judge as ground truth.

## 11.10 ❓ Interview

**Q: How would you evaluate a VLM for *our* AV product? (10-min design)**
— Walk §11.8: scope → stratified fleet/sim clips → human GT → rubric → calibrated
VLM judge + deterministic metric layer (TTC/IoU/comfort) → frozen, versioned eval →
report per-ODD + hallucination splits. Close with: "public benches are the trend
line; the private stratified eval is the contract."

**Q: A model's MMBench went up but POPE-"No" dropped. Diagnose.**
— General-capability gains often come from stronger *language priors* (more web
text in pretraining/SFT) — which inflates "Yes" answers to absent objects (#5).
Fix: hallucination-targeted data (negative pairs, DPO), or grounding rewards
(§10.11 verifier), and re-measure the split — not the overall.

**Q: Why can't we just use GPT-5 as the judge for everything?**
— Self-preference, verbosity, and *its own* perception errors (it hallucinates too —
§11.4). Calibrate on human labels, use a different model family than the candidate
when possible, feed crops for fine perception, and keep the deterministic metric
layer as the final say (§10.10).

**Q: Hallucination — cause, measurement, fix, in 30 seconds.**
— Cause: weak mid-layer visual grounding + token prior bias + no uncertainty
  training (§11.5). Measure: POPE (No-split), CHAIR, MMVP minimal pairs (§11.6).
  Fix: negative-pair DPO, grounding rewards (IoU verifier + GRPO), resolution/data
  for the thin-evidence cases, and *never* token-prune the evidence (§11.5 check).

**Q: Your eval says the model is great; the product team says it's terrible. Who's right?**
— The eval is wrong about *the product*: check stratification (§11.8 #2 — are the
failure ODDs in the set?), rubric alignment (does "good" in the rubric = "good" in
the product?), and judge calibration (§11.7). Reconcile with 50 human-reviewed
failure cases — they'll point at the missing stratum.
