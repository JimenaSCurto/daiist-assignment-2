# Assignment 2 Report — Early fire & smoke detection from cameras

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

> **How this file works.** Blocks between `<!-- AUTO:… -->` markers are rewritten by
> `uv run python main.py train` on every run (facts and computed numbers only). Every
> `✍️ YOUR ANALYSIS` block is yours: replace the guiding questions with your own answers.
> Re-run `EVAL_ONLY=1 uv run python main.py train` after changing `DECISIONS` in `train.py`.

## 1. Business problem, and the cost of being wrong

The unit of decision is **one camera frame**. For each frame the monitoring centre chooses
**Pass** (no action), **Review** (a human operator looks at the frame) or **Alarm** (escalate
or dispatch automatically). A frame is **positive** if it contains fire **or** smoke.

> ✍️ **YOUR ANALYSIS**
> - Who runs the cameras and who acts on the alerts?
> - What does each outcome (TP, FN, FP, TN, a review) cost, and why is a missed fire worse than a false alarm? Roughly by how much?
> - Is a missed *frame* the same as a missed *fire*, given that fires persist across many frames?
> - What decision would change because this model exists?

## 2. Dataset

<!-- AUTO:dataset -->
**Source:** Roboflow Universe, workspace `dangerzone-6iqvs`, project `smoke_fire-idefg`, version `5` ([link](https://universe.roboflow.com/dangerzone-6iqvs/smoke_fire-idefg/dataset/5)). Export format `yolov11`. Licence: CC BY 4.0.

**Classes:** `fire`, `smoke`. Positive class at decision level: a frame with at least one `fire` or `smoke` box. 

| Split | Images | Background (no labels) | Fire only | Smoke only | Both | Neither fire nor smoke | Fire-or-smoke prevalence |
|---|---|---|---|---|---|---|---|
| train | 10,296 | 1,689 (16.4%) | 309 | 8,097 | 201 | 1,689 | 83.6% |
| val | 2,573 | 411 (16.0%) | 80 | 2,037 | 45 | 411 | 84.0% |
| test | 2,570 | 413 (16.1%) | 70 | 2,045 | 42 | 413 | 83.9% |

| Instances | train | val | test |
|---|---|---|---|
| fire | 866 | 227 | 194 |
| smoke | 8,798 | 2,177 | 2,176 |

| Box area / image area | Boxes (all) | Median | IQR (Q1–Q3) | Share < 1% | Test median | Test share < 1% |
|---|---|---|---|---|---|---|
| fire | 1,287 | 3.71% | 1.67%–7.52% | 13.1% | 3.67% | 17.5% |
| smoke | 13,151 | 37.52% | 19.05%–55.22% | 0.6% | 35.56% | 0.5% |

**Test-set prevalence** of fire-or-smoke frames: 83.9%.

**Low-count flag** (< 100 test instances): none.

![Dataset overview](weights/dataset_overview.png)
<!-- /AUTO:dataset -->

### What the dataset's composition implies

> ✍️ **YOUR ANALYSIS**
> - Which per-class numbers can you trust, and which are too thin?
> - Does the test split look like your production cameras (scene types, distance, time of day, what the negatives are)?
> - What does the share of small boxes imply for detecting *early* smoke?
> - What does the gap between the test prevalence and your assumed base rate imply?
> - Why does this dataset actually serve the problem above rather than merely resemble it?

## 3. Training

<!-- AUTO:training -->
| Setting | Value |
|---|---|
| Model | `yolo11n.pt` (COCO-pretrained, finetuned) |
| Device | Apple MPS (Apple M4) (`mps`) |
| Platform | macOS-15.6-arm64-arm-64bit |
| Versions | python 3.12.12, torch 2.11.0, ultralytics 8.4.173 |
| Epochs (requested / completed) | 1 / 1 |
| Best epoch | 1 |
| Image size | 640 |
| Batch | 16 |
| Patience | 10 |
| Train fraction | 0.05 |
| Train images used | 515 |
| Seed | 0 |
| Optimizer / lr0 / momentum | AdamW / 0.001667 / 0.937 |
| Augmentation (mosaic / fliplr / hsv_v) | 1.0 / 0.5 / 0.4 |
| Wall-clock | 3.2 min (191 s/epoch) |
| Best val (P / R / mAP50 / mAP50-95) | 0.002 / 0.375 / 0.002 / 0.000 |
| Started | 2026-10-25T14:59:47+00:00 |

Full settings: `weights/run_meta.json`.
<!-- /AUTO:training -->

> ✍️ **YOUR ANALYSIS**
> - Why this model size and these settings? What would you change with more compute?

## 4. Results, baselines and failure modes

<!-- AUTO:results -->
**Positive class:** frame contains fire or smoke (any ground-truth box of either class).

**Overall (test):** mAP50 = 0.002, mAP50-95 = 0.001.

| Class | Precision | Recall | AP50 | AP50-95 | Test instances |
|---|---|---|---|---|---|
| fire | 0.000 | 0.170 | 0.000 | 0.000 | 194 |
| smoke | 0.005 | 0.625 | 0.005 | 0.001 | 2,176 |

*precision/recall at the single confidence that maximises mean F1 (Ultralytics convention); AP is threshold-free.*

![Detection confusion matrix](weights/confusion_detection.png)

**Baselines on the same test split** (frame level; costs use the DECISIONS block at the production base rate):

| Model / policy | mAP50 | Frame recall | Frame FPR | Cost / 1,000 frames | TP/FN/FP/TN (test) |
|---|---|---|---|---|---|
| Finetuned (model alone at t*) | 0.002 | 0.0% | 0.0% | €50,000.00 | 0/2157/0/413 |
| COCO zero-shot | 0.000 | 0.0% | 0.0% | €50,000.00 | 0/2157/0/413 |
| Predict nothing | 0.000 | 0.0% | 0.0% | €50,000.00 | 0/2157/0/413 |
| Always alarm | — | 100.0% | 100.0% | €150,000.00 | 2157/0/413/0 |

**COCO zero-shot** (`yolo11n.pt`, conf ≥ 0.25): COCO has no fire/smoke class. It detected something in 1,207 positive and 193 negative test frames. Top classes on positive frames: `car` (271), `chair` (242), `person` (215), `bed` (104), `couch` (88). On negative frames: `person` (70), `pizza` (20), `vase` (11), `train` (10), `boat` (10).
<!-- /AUTO:results -->

> ✍️ **YOUR ANALYSIS**
> - Which class carries the headline number, and which one is dragging?
> - Where does the model fail? Read the confusion matrix: what does it confuse with what, and what does it miss?
> - What do smoke false positives probably look like (clouds, fog, dust, steam)?
> - Does the model beat both baselines, and by how much *where it matters* for the decision?

## 5. Cost matrix and operating point

<!-- AUTO:costing -->
| Decision input | Value used |
|---|---|
| `cost_fn` | 50,000.0 |
| `cost_fp` | 150.0 |
| `cost_tp` | 150.0 |
| `cost_tn` | 0.0 |
| `base_rate` | 0.001 |
| `review_seconds` | 20 |
| `reviewer_cost_per_hour` | 25.0 |
| `human_recall` | 0.95 |
| `human_fpr` | 0.02 |
| `n_cameras` | 200 |
| `frames_per_camera_per_day` | 1,440 |
| `productive_hours_per_fte_day` | 6.5 |
| `paid_hours_per_fte_day` | 8.0 |
| `min_on_duty_fte_24_7` | 4.2 |
| `available_reviewer_fte` | 5.0 |
| `cost_review` (derived) | €0.1389 per reviewed frame |
| `frames_per_day` (derived) | 288,000 |

![Cost vs threshold](weights/cost_curve.png)

**Model alone:** t* = 0.105, expected cost €50,000.00 per 1,000 frames (bootstrap 95% CI €50,000.00–€50,000.00); frame recall at t* 0.0% (95% CI 0.0%–0.0%), FPR 0.00%.

**Three-way at feasible (lo, hi):** €49,658.52 per 1,000 (95% CI €49,468.57–€49,828.72). Bootstrap: 1000 stratified resamples of the test set, thresholds held fixed.

Test prevalence 83.9% vs assumed base rate 0.001: all costs reweight the test-measured TPR/FPR to the base rate.

**Sensitivity to `cost_fn`:**

| cost_fn (EUR) | t* | Model alone / 1k | Unconstrained (lo, hi) | Feasible (lo, hi) | Feasible / 1k | Recommended |
|---|---|---|---|---|---|---|
| 5000 (×0.10) | 0.105 | €5,000.00 | (0.01, 0.11) | (0.05, 0.11) | €4,993.93 | three-way |
| 8891.4 (×0.18) | 0.105 | €8,891.40 | (0.01, 0.11) | (0.05, 0.11) | €8,866.48 | three-way |
| 15811.4 (×0.32) | 0.105 | €15,811.39 | (0.01, 0.11) | (0.04, 0.11) | €15,740.94 | three-way |
| 28117.1 (×0.56) | 0.105 | €28,117.07 | (0.00, 0.11) | (0.04, 0.11) | €27,949.07 | three-way |
| 50000 (×1.00) | 0.105 | €50,000.00 | (0.00, 0.11) | (0.04, 0.11) | €49,658.52 | three-way |
| 88914 (×1.78) | 0.105 | €88,913.97 | (0.00, 0.11) | (0.04, 0.11) | €88,263.99 | three-way |
| 158114 (×3.16) | 0.013 | €133,533.16 | (0.00, 0.11) | (0.01, 0.01) | €146,308.44 | model alone |
| 281171 (×5.62) | 0.010 | €146,818.55 | (0.00, 0.11) | (0.01, 0.01) | €147,107.14 | model alone |
| 500000 (×10.00) | 0.010 | €147,731.61 | (0.00, 0.11) | (0.01, 0.01) | €148,527.45 | model alone |

**Sensitivity to `base_rate`:**

| base_rate | t* | Model alone / 1k | Unconstrained (lo, hi) | Feasible (lo, hi) | Feasible / 1k | Recommended |
|---|---|---|---|---|---|---|
| 0.0001 (×0.10) | 0.105 | €5,000.00 | (0.01, 0.11) | (0.05, 0.11) | €4,993.29 | three-way |
| 0.000178 (×0.18) | 0.105 | €8,891.40 | (0.01, 0.11) | (0.05, 0.11) | €8,865.89 | three-way |
| 0.000316 (×0.32) | 0.105 | €15,811.39 | (0.01, 0.11) | (0.04, 0.11) | €15,740.16 | three-way |
| 0.000562 (×0.56) | 0.105 | €28,117.07 | (0.00, 0.11) | (0.04, 0.11) | €27,948.57 | three-way |
| 0.001 (×1.00) | 0.105 | €50,000.00 | (0.00, 0.11) | (0.04, 0.11) | €49,658.52 | three-way |
| 0.001778 (×1.78) | 0.105 | €88,913.97 | (0.00, 0.11) | (0.04, 0.11) | €88,264.88 | three-way |
| 0.003162 (×3.16) | 0.013 | €133,573.30 | (0.00, 0.11) | (0.01, 0.01) | €146,316.54 | model alone |
| 0.005623 (×5.62) | 0.010 | €146,835.81 | (0.00, 0.11) | (0.01, 0.01) | €147,124.46 | model alone |
| 0.01 (×10.00) | 0.010 | €147,765.20 | (0.00, 0.11) | (0.01, 0.01) | €148,561.18 | model alone |

**Break-even points** (one input scaled, all others held at the values above):

- Recommended policy vs `cost_fn`: switches at **3,748** (×0.075); below → do nothing, above → three-way.
- Recommended policy vs `cost_fn`: switches at **1.253e+05** (×2.51); below → three-way, above → model alone.
- Recommended policy vs `cost_fn`: switches at **1.565e+06** (×31.3); below → model alone, above → always alarm.
- Recommended policy vs `cost_fp`: switches at **4.777** (×0.0318); below → always alarm, above → model alone.
- Recommended policy vs `cost_fp`: switches at **59.77** (×0.398); below → model alone, above → three-way.
- Recommended policy vs `cost_fp`: switches at **2,663** (×17.8); below → three-way, above → do nothing.
- Recommended policy vs `base_rate`: switches at **7.2e-05** (×0.0722); below → do nothing, above → three-way.
- Recommended policy vs `base_rate`: switches at **0.002506** (×2.51); below → three-way, above → model alone.
- Recommended policy vs `base_rate`: switches at **0.03047** (×30.5); below → model alone, above → always alarm.
- Recommended policy vs `review_seconds`: switches at **0.4062** (×0.0203); below → review everything, above → three-way.
- Recommended policy vs `review_seconds`: switches at **167.9** (×8.4); below → three-way, above → do nothing.
- Feasible review band in use vs `review_seconds`: switches at **167.9** (×8.4); below → review band used, above → no review band.
<!-- /AUTO:costing -->

> ✍️ **YOUR ANALYSIS**
> - Justify every figure in the DECISIONS table: "I assumed X because Y, and the answer changes if X drops below Z". Use the break-even points above.
> - Where does your base rate come from?
> - Which single assumption would most change your answer if it were wrong?

## 6. What a person still has to check

<!-- AUTO:review -->
**Recommended operating point (feasible optimum):** lo = 0.04, hi = 0.11 (pass if s < lo, review if lo ≤ s < hi, alarm if s ≥ hi). Recommended policy overall: **three-way**.

|  | Feasible optimum | Unconstrained optimum |
|---|---|---|
| (lo, hi) | (0.04, 0.11) | (0.00, 0.11) |
| Cost / 1,000 frames | €49,658.52 | €5,810.44 |
| of which outcomes / staffing | €49,655.60 / €2.92 | €5,639.50 / €170.94 |
| Share sent to review | 1.694% | 100.000% |
| Frames reviewed per day | 4,879 | 288,000 |
| Reviewer hours per day | 27.1 | 1,600.0 |
| Workload FTE | 4.17 | 246.15 |
| FTE needed (incl. 24/7 floor) | 4.20 | 246.15 |
| FTE available | 5 | 5 |
| Feasible | yes | no |
| Missed per 1,000 true fire/smoke frames | 992.1 | 50.0 |

Staff capacity: at most 2.031% of frames (5,850 per day) can be reviewed with 5 FTE; 4,757 of 5,151 (lo, hi) pairs on the 0.01 grid are feasible.

**Policy comparison** (same base rate and costs; per 1,000 frames and per day):

| Policy | Cost / 1,000 | Cost / day | Review share | Reviews / day | FTE needed | Feasible | Missed / 1,000 true |
|---|---|---|---|---|---|---|---|
| (a) Do nothing / never alarm | €50,000.00 | €14,400,000.00 | 0.00% | 0 | 0.0 | yes | 1000.0 |
| (b) Human reviews every frame | €5,810.44 | €1,673,406.77 | 100.00% | 288,000 | 246.2 | **no** | 50.0 |
| (c) Model alone at t*=0.105 | €50,000.00 | €14,400,000.00 | 0.00% | 0 | 0.0 | yes | 1000.0 |
| (d) Three-way at feasible optimum (0.04, 0.11) | €49,658.52 | €14,301,653.35 | 1.69% | 4,879 | 4.2 | yes | 992.1 |

Staffing cost = FTE needed × paid hours per FTE-day × hourly cost. It includes the review time itself (share × `cost_review`) plus coverage/idle time.
<!-- /AUTO:review -->

> ✍️ **YOUR ANALYSIS**
> - Is this staffing realistic for 24/7 monitoring?
> - What breaks if the camera count or frame rate doubles?
> - How does your design compare with doing nothing, a person checking everything, and the model deciding alone?
> - Would you ship it, and with which guardrails?

## 7. Dashboard

<!-- AUTO:dashboard -->
```bash
uv run python main.py app
```

Opens on http://127.0.0.1:7860. It loads only `weights/` (never trains, never needs the dataset).

- **Header:** live KPIs: expected cost / 1,000 frames, share sent to review, reviewers needed (FTE, flagged when over capacity), missed fires per 1,000 true fire frames.
- **Decide:** lo/hi sliders (start at the feasible optimum), reset and model-only controls; 2×3 pass/review/alarm matrix (per 1,000 at the base rate, or raw test counts), final outcome after review, cost-vs-threshold curve with t* and the current band, and the policy comparison.
- **Try a frame:** upload an image; boxes ≥ lo are drawn and the frame gets a Pass / Review / Alarm badge from the current sliders.
- **Model:** overall and per-class detection metrics, frame-level per-class P/R at the operating point, detection confusion matrix, baselines, dataset summary.
- **Assumptions:** the DECISIONS used, derived review cost, bootstrap intervals, and a what-if panel that never overwrites the report numbers.
<!-- /AUTO:dashboard -->

## 8. Limitations & next steps

> ✍️ **YOUR ANALYSIS**
> - Real limitations you found, and concretely how you would address each with more time or data. No generic hedging.

## 9. Generative AI use disclosure

The pipeline (`train.py`) and dashboard (`app.py`) code, and this report's skeleton and auto-generated
factual blocks, were written by Claude Code (Anthropic) from the student's specification.

> ✍️ **YOUR ANALYSIS**
> - Complete this disclosure per the syllabus AI Policy: what else you used AI for, and confirm that the costs, base rate, thresholds, staffing assumptions and all analysis above are your own.
