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
| Epochs (requested / completed) | 30 / 30 |
| Best epoch | 30 |
| Image size | 640 |
| Batch | 16 |
| Patience | 10 |
| Train fraction | 1.0 |
| Train images used | 10296 |
| Seed | 0 |
| Optimizer / lr0 / momentum | AdamW / 0.001667 / 0.937 |
| Augmentation (mosaic / fliplr / hsv_v) | 1.0 / 0.5 / 0.4 |
| Training time | 18.3 h (36.6 min/epoch, 2 sessions: interrupted and resumed) |
| Best val (P / R / mAP50 / mAP50-95) | 0.844 / 0.789 / 0.840 / 0.573 |
| Started | 2026-10-26T08:36:51+00:00 |

Full settings: `weights/run_meta.json`.
<!-- /AUTO:training -->

> ✍️ **YOUR ANALYSIS**
> - Why this model size and these settings? What would you change with more compute?

## 4. Results, baselines and failure modes

<!-- AUTO:results -->
**Positive class:** frame contains fire or smoke (any ground-truth box of either class).

**Overall (test):** mAP50 = 0.824, mAP50-95 = 0.574.

| Class | Precision | Recall | AP50 | AP50-95 | Test instances |
|---|---|---|---|---|---|
| fire | 0.804 | 0.577 | 0.669 | 0.343 | 194 |
| smoke | 0.990 | 0.938 | 0.979 | 0.805 | 2,176 |

*precision/recall at the single confidence that maximises mean F1 (Ultralytics convention); AP is threshold-free.*

![Detection confusion matrix](weights/confusion_detection.png)

**Baselines on the same test split** (frame level; costs use the DECISIONS block at the production base rate):

| Model / policy | mAP50 | Frame recall | Frame FPR | Cost / 1,000 frames | TP/FN/FP/TN (test) |
|---|---|---|---|---|---|
| Finetuned (model alone at t*) | 0.824 | 93.0% | 0.7% | €4,705.12 | 2007/150/3/410 |
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

**Model alone:** t* = 0.625, expected cost €4,705.12 per 1,000 frames (bootstrap 95% CI €3,524.18–€6,104.04); frame recall at t* 93.0% (95% CI 92.0%–94.1%), FPR 0.73%.

**Three-way at feasible (lo, hi):** €2,686.54 per 1,000 (95% CI €2,248.77–€3,122.98). Bootstrap: 1000 stratified resamples of the test set, thresholds held fixed.

Test prevalence 83.9% vs assumed base rate 0.001: all costs reweight the test-measured TPR/FPR to the base rate.

**Sensitivity to `cost_fn`:**

| cost_fn (EUR) | t* | Model alone / 1k | Unconstrained (lo, hi) | Feasible (lo, hi) | Feasible / 1k | Recommended |
|---|---|---|---|---|---|---|
| 5000 (×0.10) | 0.772 | €667.15 | (0.32, 0.78) | (0.32, 0.78) | €452.19 | three-way |
| 8891.4 (×0.18) | 0.772 | €1,082.09 | (0.25, 0.78) | (0.32, 0.78) | €645.40 | three-way |
| 15811.4 (×0.32) | 0.772 | €1,819.97 | (0.12, 0.78) | (0.32, 0.78) | €989.00 | three-way |
| 28117.1 (×0.56) | 0.772 | €3,132.12 | (0.09, 0.78) | (0.32, 0.78) | €1,600.00 | three-way |
| 50000 (×1.00) | 0.625 | €4,705.12 | (0.08, 0.78) | (0.32, 0.78) | €2,686.54 | three-way |
| 88914 (×1.78) | 0.469 | €6,771.37 | (0.00, 0.78) | (0.32, 0.78) | €4,618.71 | three-way |
| 158114 (×3.16) | 0.469 | €10,236.18 | (0.00, 0.78) | (0.32, 0.78) | €8,054.64 | three-way |
| 281171 (×5.62) | 0.255 | €15,764.38 | (0.00, 0.78) | (0.25, 0.63) | €13,670.25 | three-way |
| 500000 (×10.00) | 0.255 | €25,097.85 | (0.00, 0.78) | (0.25, 0.63) | €23,303.00 | three-way |

**Sensitivity to `base_rate`:**

| base_rate | t* | Model alone / 1k | Unconstrained (lo, hi) | Feasible (lo, hi) | Feasible / 1k | Recommended |
|---|---|---|---|---|---|---|
| 0.0001 (×0.10) | 0.772 | €546.55 | (0.32, 0.78) | (0.32, 0.78) | €323.93 | three-way |
| 0.000178 (×0.18) | 0.772 | €971.92 | (0.25, 0.78) | (0.32, 0.78) | €528.24 | three-way |
| 0.000316 (×0.32) | 0.772 | €1,728.34 | (0.12, 0.78) | (0.32, 0.78) | €891.56 | three-way |
| 0.000562 (×0.56) | 0.772 | €3,073.47 | (0.09, 0.78) | (0.32, 0.78) | €1,537.63 | three-way |
| 0.001 (×1.00) | 0.625 | €4,705.12 | (0.08, 0.78) | (0.32, 0.78) | €2,686.54 | three-way |
| 0.001778 (×1.78) | 0.469 | €6,880.57 | (0.00, 0.78) | (0.32, 0.78) | €4,729.61 | three-way |
| 0.003162 (×3.16) | 0.469 | €10,539.57 | (0.00, 0.78) | (0.32, 0.78) | €8,362.77 | three-way |
| 0.005623 (×5.62) | 0.255 | €16,411.52 | (0.00, 0.78) | (0.25, 0.63) | €14,327.93 | three-way |
| 0.01 (×10.00) | 0.255 | €26,357.58 | (0.00, 0.78) | (0.12, 0.28) | €24,579.74 | three-way |

**Break-even points** (one input scaled, all others held at the values above):

- Recommended policy vs `cost_fn`: switches at **865.2** (×0.0173); below → model alone, above → three-way.
- Recommended policy vs `cost_fn`: switches at **3.793e+06** (×75.9); below → three-way, above → always alarm.
- Recommended policy vs `cost_fp`: switches at **1.975** (×0.0132); below → always alarm, above → three-way.
- Recommended policy vs `cost_fp`: switches at **1.184e+04** (×78.9); below → three-way, above → model alone.
- Recommended policy vs `base_rate`: switches at **1.4e-05** (×0.0144); below → model alone, above → three-way.
- Recommended policy vs `base_rate`: switches at **0.07077** (×70.8); below → three-way, above → always alarm.
- Recommended policy vs `review_seconds`: switches at **167.8** (×8.39); below → three-way, above → model alone.
- Feasible review band in use vs `review_seconds`: switches at **167.8** (×8.39); below → review band used, above → no review band.
<!-- /AUTO:costing -->

> ✍️ **YOUR ANALYSIS**
> - Justify every figure in the DECISIONS table: "I assumed X because Y, and the answer changes if X drops below Z". Use the break-even points above.
> - Where does your base rate come from?
> - Which single assumption would most change your answer if it were wrong?

## 6. What a person still has to check

<!-- AUTO:review -->
**Recommended operating point (feasible optimum):** lo = 0.32, hi = 0.78 (pass if s < lo, review if lo ≤ s < hi, alarm if s ≥ hi). Recommended policy overall: **three-way**.

|  | Feasible optimum | Unconstrained optimum |
|---|---|---|
| (lo, hi) | (0.32, 0.78) | (0.08, 0.78) |
| Cost / 1,000 frames | €2,686.54 | €2,391.06 |
| of which outcomes / staffing | €2,683.22 / €3.32 | €2,382.36 / €8.70 |
| Share sent to review | 1.942% | 5.087% |
| Frames reviewed per day | 5,592 | 14,651 |
| Reviewer hours per day | 31.1 | 81.4 |
| Workload FTE | 4.78 | 12.52 |
| FTE needed (incl. 24/7 floor) | 4.78 | 12.52 |
| FTE available | 5 | 5 |
| Feasible | yes | no |
| Missed per 1,000 true fire/smoke frames | 49.7 | 41.7 |

Staff capacity: at most 2.031% of frames (5,850 per day) can be reviewed with 5 FTE; 3,262 of 5,151 (lo, hi) pairs on the 0.01 grid are feasible.

**Policy comparison** (same base rate and costs; per 1,000 frames and per day):

| Policy | Cost / 1,000 | Cost / day | Review share | Reviews / day | FTE needed | Feasible | Missed / 1,000 true |
|---|---|---|---|---|---|---|---|
| (a) Do nothing / never alarm | €50,000.00 | €14,400,000.00 | 0.00% | 0 | 0.0 | yes | 1000.0 |
| (b) Human reviews every frame | €5,810.44 | €1,673,406.77 | 100.00% | 288,000 | 246.2 | **no** | 50.0 |
| (c) Model alone at t*=0.625 | €4,705.12 | €1,355,074.30 | 0.00% | 0 | 0.0 | yes | 69.5 |
| (d) Three-way at feasible optimum (0.32, 0.78) | €2,686.54 | €773,723.32 | 1.94% | 5,592 | 4.8 | yes | 49.7 |

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
