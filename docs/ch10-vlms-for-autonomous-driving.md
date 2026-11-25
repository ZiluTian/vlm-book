# Chapter 10 — VLMs for Autonomous Driving (Your Bridge Back)

> 🎯 **The one sentence:** a VLM in an AV program is almost never in the 10 Hz control
> loop — it's the **semantic layer**: data engine, long-tail miner, explanation
> generator, and (at the frontier) an end-to-end policy that *thinks in language*.
> Your BEV expertise is exactly what those systems still need underneath.

## 10.1 Where a VLM adds value (and where it doesn't)

**Adds:** open-vocabulary semantics ("construction crew working on the right
shoulder"), long-tail *naming* and *explanation*, cross-modal QA over clips
("why did we brake?"), zero-shot OOD detection, human-readable logs, simulation
scenario authoring.

**Doesn't:** 10–30 Hz perception (too slow, too costly), metric geometry
(VLMs don't give you a calibrated point cloud — *you* do), deterministic guarantees
(probabilistic text ≠ SOTIF evidence), cheap mass inference (your BEV runs on a
SoC; a VLM runs on a GPU farm or a big edge GPU).

**Check:** one sentence on the division of labor? → *"The BEV/occupancy stack owns
geometry at 10 Hz; the VLM owns semantics at 0.1–1 Hz (and in the cloud, at any Hz
that money allows)."*

## 10.2 The five roles (the taxonomy to recite)

| Role | What it is | Representative work |
|---|---|---|
| A. **Data engine** | VLM labels, captions, filters, and enriches fleet data | auto-captioning, rare-event mining, QA for annotation |
| B. **Language supervision** | text intermediate targets improve an E2E driver | DriveGPT, DriveLM, LMDrive, Senna |
| C. **VLM-as-policy** | one model sees + reasons + emits trajectory | EMMA (Waymo), OpenEMMA |
| D. **Hybrid monitor** | real-time stack drives; VLM watches asynchronously | VLM safety-copilot, ODD watcher |
| E. **World model / sim** | generates video/occupancy to *make* data and test | GAIA-1, DriveDreamer, OccWorld, Cosmos-class |

2025–26 surveys (VLA4AD et al.) frame B/C as **VLA models** — vision-language-*action*
— and note the open problems are exactly the ones you'd expect: spatial grounding
precision, real-time efficiency, and formal verification (ISO 26262 / SOTIF mapping
is now standard vocabulary in this literature).

**Check:** which role is *today's* production workhorse vs the research frontier?
→ *A + D are in production (data engines, cloud monitors); C is the frontier; E is
the data-multiplier behind all of it.*

## 10.3 Role A — the data engine (the money role)

```python
def mine_long_tail(clip_bank, vlm):
    events = []
    for clip in clip_bank:
        # 1) cheap prefilter: your existing stack flags "something odd"
        #    (no detection but hard brake; OOD occupancy; etc.)
        if not cheap_prefilter(clip): continue
        # 2) VLM names it: zero-shot, open vocabulary
        desc = vlm.ask(frames(clip), "Describe anything unusual, in ≤20 words.")
        # 3) VLM scores it (structured output → JSON)
        score = vlm.ask(frames(clip), SCENARIO_RUBRIC_JSON)   # {novelty, severity, ...}
        # 4) cluster + dedupe descriptions (embedding similarity)
        events.append((clip, desc, score))
    return cluster_by_embedding(events)          # → long-tail scenario catalog
```

This is the role that pays for VLM programs: a 7B VLM + your BEV prefilter turns a
petabyte clip bank into a *labeled* long-tail catalog — the thing your sim and
your E2E training both starve for. (LLaVA's data engine, §6.11, was this trick on
the web; here it's your fleet.)

**Check:** why prefilter before calling a VLM on the whole archive? → *fewer
clips need expensive review, but the trigger can hide the events you most want
to discover. Measure that miss rate separately.*

### Worked example: find unusual clips without hiding the missed ones

**Why:** a shortlist can look excellent to reviewers while missing the entire
class of construction detours that never triggered the original detector.

**What:** suppose the archive contains 100,000 clips and a cheap trigger sends
5,000 to a VLM for offline tagging. That's 20× fewer VLM calls than reviewing
every clip with the VLM, not necessarily 20× lower total cost. If human reviewers
find 150 valid detours in the first 300 ranked results, precision@300 is 50%.
That tells you about the review queue, not the fraction of all detours found.

**How:** review a stratified sample of both accepted and rejected clips and
record each stratum's sampling rate, so you can estimate how many true events
the trigger missed. Separately check whether the VLM ranks accepted clips well.
If rejects contain many true detours, expand or diversify the trigger before
replacing the VLM. If recall at the trigger is adequate but the queue is noisy,
change the ranking prompt or rubric. Keep the output as an offline candidate
label until reviewed; it is not a driving command.

## 10.4 Role B — language as a supervisory signal

Idea (DriveGPT/DriveLM family): an E2E policy is hard to supervise with just
trajectories. Insert **language**: the model (or a teacher VLM) first *describes
and reasons* ("a van is cutting in from the right; brake and yield"), then emits
the action. Language carries commonsense the raw pixels don't, and gives you a
**debuggable intermediate**.

```python
# teacher: strong VLM labels (image sequence) -> (caption, reason, action)
# student: small E2E model imitates; optionally predicts text tokens too
def drive_step(ego_img_seq, student):
    h  = student.vision(ego_img_seq)          # your BEV-style encoder works fine here
    h  = student.llm(h + prompt)             # small decoder
    reasoning = h.sample(max_new=reason_len)  # "van cutting in..."
    action  = h.action_head(reasoning_h)     # waypoints / accel+steer
    return action, reasoning                 # reasoning = your debug log
```

Why it works: language is a *bottleneck that keeps the relevant semantics*, and a
*lever* (you can reward good reasoning, §10.11 GRPO). Why it's hard: the text
bottleneck drops geometry — the model must still get meters right from a 576-token
image. That tension *is* the research agenda.

**Check:** why not just have the small policy *read* the teacher's text at test
time? → *you can (test-time guidance), but latency + fragility: production wants the
policy to *have* the reasoning, not fetch it.*

## 10.5 Role C — VLM-as-policy (EMMA and the VLA wave)

EMMA (Waymo, 2024): a ~30B MLLM that consumes camera images + driving state and
outputs **trajectory + natural-language explanation** jointly, trained with
behavioral cloning + RL. OpenEMMA (open reproduction) made the recipe public:
image tokens in, waypoints + text out, on standard MLLM infrastructure.
The 2025–26 VLA4AD literature (DriveGPT-4, LMDrive, Senna, GPT-4V-AD, Cosmos-class
world models) is all variations of §10.4–10.5:

- **Action as tokens:** waypoints quantized into a small token vocab (like §8.7
  box tokens) — or an action head on the LM hidden states.
- **State as text:** speed, throttle, map context formatted into the prompt.
- **Training:** BC on fleet data → (optionally) GRPO with a driving reward
  (collision-free, progress, comfort) → the R1 recipe, with a *driving* verifier.

**Check:** the honest weakness list for VLM-as-policy? → *latency (decode is
sequential — 10 Hz is brutal), geometric precision (tokens quantize meters),
verification (no formal guarantees yet), cost (a 30B model per vehicle-hour is a
cloud bill). Which is why production today = C in sim/cloud, A+D on the car.*

## 10.6 Role D — the hybrid monitor (the realistic architecture)

```
ON VEHICLE (10-30 Hz):            CLOUD (0.1-1 Hz, async):
  cameras → BEV/occupancy →        clip ─► VLM: "Why brake?" ─► explanation
  planning → control              ─► ODD check ─► flag if out-of-distribution
        │                            ─► rare-event catalog (§10.3)
        └── 10 Hz state + keyframes ─►  ─► VLM-as-judge on sim scenarios
             (event-triggered upload)     ─► regression test: new build vs catalog
```

The VLM is a **second brain at low Hz**: it explains the stack's decisions (SAE/
regulator-facing logs), catches what the fixed-class stack misses (novel, long-tail,
semantic), and judges new builds against the long-tail catalog. The BEV stack keeps
driving; the VLM keeps *meaning*.

**Check:** event-triggered vs always-on upload — which, and why? → *event-triggered
(odd events, disengagements, hard maneuvers): always-on is a bandwidth/cost disaster
and 99.9% of frames are boring. Trigger = your cheap systems deciding what the
expensive system looks at — §10.3's rule, at the system level.*

## 10.7 Role E — world models: generating the data you can't drive

GAIA-1 family: a video diffusion model conditioned on (state, action) generates
what the cameras *would* have shown. Uses:
- **Sim:** roll out rare scenarios on demand (the long-tail catalog, animated);
- **Counterfactuals:** "what if we hadn't braked?" — evaluation without the fleet;
- **Pretext data:** generated (image, text) pairs for VLM pretraining on-domain.
OccWorld-class work does the same with 3D occupancy (closer to your world:
generated BEV sequences). The 2026 surveys treat world models as the *data
infrastructure* behind B/C — the loop is: fleet → VLM catalogs → world model
generates → policy trains → fleet again.

**Check:** why is a *video diffusion* model (not a token AR model) the natural
world-model form? → *video is dense, regular, continuous — parallel denoising
fits (§8.8); plus diffusion gives you the action-conditioned generative objective
you want for counterfactuals.*

## 10.8 The latency budget (the table that ends arguments)

| Task | Budget | Who does it |
|---|---|---|
| control | < 10 ms, 100 Hz | classic MPC/LQR |
| perception (BEV/occ) | 10–30 ms, 10–30 Hz | your stack, on SoC |
| planning (waypoints) | 50–100 ms, 10 Hz | E2E policy / planner |
| VLM explanation / ODD | 100 ms–2 s, 0.1–1 Hz | VLM, vehicle GPU or cloud |
| data curation / sim | hours | cloud VLM + world model |

Rule: **the slower the task, the more language it can afford.** Everything in
§10.2–10.7 is a placement on this table.

## 10.9 On-vehicle vs cloud: the economics (one paragraph)

On-vehicle 3–7B VLM (int4, short context): ~$0.01–0.05/hour of compute you already
have; latency 0.5–3 s; quality "good enough for ODD/explain". Cloud 70B+: ~10–50×
quality on hard semantic tasks; latency 1–10 s; cost per 1k clips ≈ a truck of
edge compute. Production answer is *always* both: car handles routine semantics,
cloud handles the hard 1% and the whole data loop. (And prefix caching, §9.7,
makes the cloud side sane for multi-turn clip review.)

## 10.10 The VLM as judge (your eval infra, Ch 11, on driving)

The highest-leverage VLM use after data engines: **judging**.
- Sim scenarios: "did the ego yield appropriately to the merging truck?" — rubric +
  VLM judge, with your metric stack (TTC, comfort) as the *grounding* layer.
- Regression: new E2E build vs long-tail catalog — VLM scores each scenario,
  trend over time, gate releases on it.
- Log review: human review queues ranked by VLM severity score (humans look at the
  2%, not the 100%).
Design rule: the VLM judge *ranks and explains*; the **metrics decide** (the judge
routes attention, it doesn't replace instrumentation).

**Check:** why not let the VLM judge alone set the pass/fail? → *it's probabilistic
and itself long-tail-weak; SOTIF/26262 evidence needs deterministic, auditable
criteria. VLM = triage + explanation layer over the metric layer.*

## 10.11 Your BEV expertise, priced in VLM teams

What a VLM-driven AV team *cannot* build without you:
1. **Geometry:** calibrated BEV/occupancy that VLMs can't produce — the VLM's
   "where" must come from your stack (grounding handoff: image crops *from* your
   detections fed to the VLM at full res — the §8.6 tool loop, with your detectors
   as the tools).
2. **The cheap prefilter** (§10.3, §10.6): without your 10 Hz stack, the VLM program
   is a cost center.
3. **The verifier** for GRPO-style driving RL: collision/TTC/comfort checks are
   *your* metrics — the reward function is perception.
4. **Sim grounding:** world-model outputs must match BEV geometry or they're fiction.

Interview narrative (memorize the shape): *"I've built the 10 Hz geometric layer.
VLMs add the semantic layer at 0.1 Hz — and the data loop around both. The hard
problems left are exactly the interfaces between them: grounding, verification,
and cost."*

## 10.12 ❓ Interview

**Q: Where would you put a VLM in our AV stack? (system design, 15 min)**
— Walk §10.6's diagram: on-vehicle BEV drives; event-triggered uploads; cloud VLM
for explain/ODD/catalog; VLM-as-judge over sim with metric grounding; on-vehicle
small VLM for routine semantics. End with the latency table (§10.8) and the
data-loop (§10.7). Mention the frontier (C: VLA policies) and why it's not in the
control loop yet (latency, verification, cost).

**Q: Why isn't the VLM in the real-time loop?**
— Three numbers: decode latency (sequential tokens, §9.2), cost per frame, and
verification (no guarantees). Add: "and the BEV already *is* the right
representation for 10 Hz — tokens for geometry at 10 Hz is a waste of both."

**Q: How do you handle long-tail events a VLM sees but the BEV stack doesn't?**
— Bidirectional loop: VLM names them (Role A) → scenarios cataloged (Role E) →
synthetic training + regression tests → BEV stack patched/extended. The VLM is the
*sensor for the sensor's gaps* — say it that way.

**Q: Design a verifier for GRPO on a driving policy.**
— Reward = weighted sum: collision-free (hard gate, from BEV/TTC), route progress,
comfort (jerk), rule compliance (right-of-way from map+detections); format bonus
for valid waypoint tokens; KL anchor to BC policy. All terms computed from the
*geometric* stack — the VLM proposes, the metrics dispose.

**Q: World model for your program — build or buy, and what do you need from it?**
— Build on a diffusion backbone conditioned on (BEV state, action); validate against
fleet clips with *your* geometry (parallax, object consistency) before trusting it
for data; first use = long-tail catalog animation (§10.7), second = counterfactual
eval. The BEV conditioning is what makes it *driving* rather than *video*.
