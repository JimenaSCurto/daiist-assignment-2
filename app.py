"""
Assignment 2 dashboard: Smoke & Fire Triage.

Run with `uv run python main.py app`. It reads only what train.py wrote to
weights/ (best.pt plus the JSON/CSV artifacts), never trains, never downloads,
and never needs the dataset. Every number on the Decide tab is recomputed with
numpy from weights/test_scores.csv using the same cost functions as train.py.
"""

import html
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw, ImageFont

from train import (derived, describe_policy, optimise, policy_cost, share_at_or_above,
                   sweep_thresholds, three_way_rates)

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
REQUIRED = ["best.pt", "run_meta.json", "dataset_stats.json", "test_scores.csv",
            "eval.json", "baselines.json", "decision.json"]
PRED_CONF = 0.001
MAX_BOXES_DRAWN = 50

PASS_C, REVIEW_C, ALARM_C = "#2F7F72", "#C98A0B", "#D2491B"
ACCENT, INK, MUTED, GRID, SURFACE = "#4F46E5", "#1E293B", "#64748B", "#E2E8F0", "#FFFFFF"
OUTCOMES = [("Pass", "○", PASS_C), ("Review", "◐", REVIEW_C), ("Alarm", "●", ALARM_C)]
WHATIF_KEYS = ("cost_fn", "cost_fp", "base_rate", "review_seconds")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def load_artifacts():
    missing = [name for name in REQUIRED if not (WEIGHTS / name).is_file()]
    if missing:
        return None, missing
    art = {name.split(".")[0]: json.loads((WEIGHTS / name).read_text())
           for name in REQUIRED if name.endswith(".json")}
    art["scores"] = pd.read_csv(WEIGHTS / "test_scores.csv")
    return art, []


ART, MISSING = load_artifacts()
if ART:
    DEC = ART["decision"]
    REPORT_D = DEC["decisions"]
    SCORES = ART["scores"]
    _y, _s = SCORES.y_true.to_numpy(), SCORES.s.to_numpy()
    POS, NEG = np.sort(_s[_y == 1]), np.sort(_s[_y == 0])
    N_POS, N_NEG = POS.size, NEG.size
    CURVE_T = sweep_thresholds(_s)
    CLASS_NAMES = ART["eval"]["class_names"]


# ---------------------------------------------------------------------------
# Pure computation (cost maths lives in train.py and is shared)
# ---------------------------------------------------------------------------
def scenario_decisions(cost_fn, cost_fp, base_rate, review_seconds):
    D = dict(REPORT_D)
    D.update(cost_fn=float(cost_fn), cost_fp=float(cost_fp),
             base_rate=min(max(float(base_rate), 1e-6), 0.999), review_seconds=float(review_seconds))
    return D


@lru_cache(maxsize=64)
def scenario(cost_fn, cost_fp, base_rate, review_seconds):
    """Everything that depends on DECISIONS but not on the sliders."""
    D = scenario_decisions(cost_fn, cost_fp, base_rate, review_seconds)
    o = optimise(POS, NEG, D, CURVE_T)
    f = o["feasible_idx"]
    lo_f, hi_f = float(o["grid_lo"][f]), float(o["grid_hi"][f])
    t_star = o["t_star"]
    tpr, fpr = float(share_at_or_above(POS, t_star)), float(share_at_or_above(NEG, t_star))
    kw = {"n_pos": N_POS, "n_neg": N_NEG}
    policies = [
        describe_policy("(a) Do nothing", 0, 0, 0, 0, D, **kw),
        describe_policy("(b) Review everything", 0, 1, 0, 1, D, lo=0.0, hi=1.01, **kw),
        describe_policy(f"(c) Model alone, t*={t_star:.3f}", tpr, 0, fpr, 0, D, lo=t_star, hi=t_star, **kw),
        describe_policy(f"(d) Three-way ({lo_f:.2f}, {hi_f:.2f})", *three_way_rates(POS, NEG, lo_f, hi_f), D,
                        lo=lo_f, hi=hi_f, **kw),
    ]
    return {"D": D, "t_star": t_star, "lo": lo_f, "hi": hi_f, "curve_cost": 1000 * o["cost"],
            "policies": policies, "recommended": o["recommended"]}


def current_policy(lo, hi, sc):
    return describe_policy("Current sliders", *three_way_rates(POS, NEG, lo, hi), sc["D"],
                           lo=lo, hi=hi, n_pos=N_POS, n_neg=N_NEG)


def decide_frame(s, lo, hi):
    return "Alarm" if s >= hi else "Review" if s >= lo else "Pass"


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def eur(x):
    return f"€{x:,.0f}" if abs(x) >= 100 else f"€{x:,.2f}"


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def table(header, rows, cls="tbl"):
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def style(fig, height, **layout):
    fig.update_layout(
        template="none", height=height, margin=dict(l=64, r=24, t=40, b=48),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, showlegend=False,
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", size=12, color=INK),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK, size=12)),
        **layout,
    )
    axis = dict(showgrid=False, zeroline=False, showline=True, linecolor=GRID, ticks="outside",
                tickcolor=GRID, tickfont=dict(color=MUTED, size=11), title_font=dict(color=INK, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def kpi_html(cur, sc):
    D = sc["D"]
    over = cur["fte_needed"] > D["available_reviewer_fte"] + 1e-9
    fte_note = (f'<span class="flag">⚠ over capacity ({D["available_reviewer_fte"]:g} available)</span>'
                if over else f'<span class="sub">of {D["available_reviewer_fte"]:g} available</span>')
    cards = [
        ("Expected cost / 1,000 frames", eur(cur["cost_per_1000"]),
         f'<span class="sub">{eur(cur["cost_per_day"])} per day</span>', ""),
        ("Share sent to review", pct(cur["review_share"], 2),
         f'<span class="sub">{cur["review_frames_per_day"]:,.0f} frames / day</span>', ""),
        ("Reviewers needed (FTE)", f'{cur["fte_needed"]:.1f}', fte_note, " warn" if over else ""),
        ("Missed fires / 1,000 true fire frames", f'{cur["missed_per_1000_true_frames"]:.1f}',
         f'<span class="sub">after human review</span>', ""),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi{extra}"><div class="label">{label}</div><div class="value">{value}</div>{sub}</div>'
        for label, value, sub, extra in cards) + "</div>"


def matrix_fig(cur, view):
    raw = view.startswith("Raw")
    m = cur["matrix_test_counts"] if raw else cur["matrix_per_1000"]
    rows = [("Fire / smoke", m["fire_smoke"]), ("Clear", m["clear"])]
    fig = go.Figure()
    for i, (row_name, vals) in enumerate(rows):
        total = sum(vals) or 1
        for j, (name, icon, colour) in enumerate(OUTCOMES):
            share = vals[j] / total
            fig.add_shape(type="rect", x0=j - 0.47, x1=j + 0.47, y0=i - 0.45, y1=i + 0.45,
                          line=dict(color=colour, width=1.5), fillcolor=colour, opacity=0.08 + 0.5 * share,
                          layer="below")
            text = f"{vals[j]:,}" if raw else (f"{vals[j]:,.1f}" if vals[j] >= 1 else f"{vals[j]:.3f}")
            fig.add_annotation(x=j, y=i, text=f"<b>{text}</b><br><span style='font-size:11px'>"
                                              f"{pct(share)} of row</span>",
                               showarrow=False, font=dict(size=15, color=INK))
    hover_x, hover_y, hover_t = [], [], []
    for i, (row_name, vals) in enumerate(rows):
        for j, (name, _, _) in enumerate(OUTCOMES):
            hover_x.append(j), hover_y.append(i)
            hover_t.append(f"{row_name} → {name}: {vals[j]:,.3f}" + (" frames (test)" if raw else " per 1,000"))
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=60, opacity=0),
                             hovertext=hover_t, hoverinfo="text"))
    style(fig, 280, title=dict(text="Raw test-set frames" if raw else
                               f"Expected frames per 1,000 at base rate {cur_base_rate(cur):g}",
                               font=dict(size=13), x=0.01))
    fig.update_xaxes(tickvals=[0, 1, 2], ticktext=[f"{icon} {n}" for n, icon, _ in OUTCOMES],
                     range=[-0.55, 2.55], showline=False, ticks="", side="top", tickfont=dict(size=13, color=INK))
    fig.update_yaxes(tickvals=[0, 1], ticktext=["Fire / smoke", "Clear"], range=[1.55, -0.55],
                     showline=False, ticks="", tickfont=dict(size=13, color=INK))
    fig.update_layout(margin=dict(l=100, r=16, t=70, b=16))
    return fig


def cur_base_rate(cur):
    m = cur["matrix_per_1000"]
    return round(sum(m["fire_smoke"]) / 1000, 6)


def final_html(cur):
    f = cur["final_per_1000"]
    fmt = lambda x: f"{x:,.3f}" if x < 10 else f"{x:,.1f}"
    return ('<div class="sub-title">Final outcome after human review, per 1,000 frames</div>' +
            table(["", "Escalated", "Not escalated"],
                  [["<b>Fire / smoke</b>", f"TP {fmt(f['TP'])}", f"FN {fmt(f['FN'])}"],
                   ["<b>Clear</b>", f"FP {fmt(f['FP'])}", f"TN {fmt(f['TN'])}"]], cls="tbl small"))


def curve_fig(sc, lo, hi, cur):
    pol = {p["policy"][:3]: p for p in sc["policies"]}
    fig = go.Figure()
    if hi > lo:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=REVIEW_C, opacity=0.14, line_width=0,
                      annotation_text="review band", annotation_position="top",
                      annotation_font=dict(color=INK, size=11))
    fig.add_trace(go.Scatter(x=CURVE_T, y=sc["curve_cost"], mode="lines", line=dict(color=ACCENT, width=2),
                             hovertemplate="t = %{x:.3f}<br>€%{y:,.2f} / 1,000<extra>model alone</extra>"))
    for key, dash, label in (("(a)", "dash", "do nothing"), ("(b)", "dot", "review everything")):
        y = pol[key]["cost_per_1000"]
        fig.add_hline(y=y, line=dict(color=MUTED, dash=dash, width=1))
        # log axis: annotations take log10(y); add_hline's own annotation mis-places them
        fig.add_annotation(x=0.99, y=np.log10(max(y, 1e-9)), xref="x", yref="y", text=f"{label} {eur(y)}",
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=INK, size=11))
    ts, cs = sc["t_star"], pol["(c)"]["cost_per_1000"]
    fig.add_vline(x=ts, line=dict(color=ACCENT, width=1), opacity=0.5)
    fig.add_trace(go.Scatter(x=[ts], y=[cs], mode="markers+text", marker=dict(color=ACCENT, size=10),
                             text=[f"t* = {ts:.3f}"], textposition="bottom right",
                             textfont=dict(color=INK, size=12),
                             hovertemplate=f"t* = {ts:.3f}<br>{eur(cs)} / 1,000<extra></extra>"))
    fig.add_trace(go.Scatter(x=[(lo + hi) / 2], y=[cur["cost_per_1000"]], mode="markers",
                             marker=dict(symbol="diamond", color=REVIEW_C, size=12, line=dict(color=INK, width=1)),
                             hovertemplate=f"current three-way ({lo:.2f}, {hi:.2f})<br>"
                                           f"{eur(cur['cost_per_1000'])} / 1,000<extra></extra>"))
    style(fig, 380, title=dict(text="Expected cost vs threshold (model alone)", font=dict(size=13), x=0.01))
    fig.update_xaxes(title_text="threshold t on frame score", range=[0, 1])
    fig.update_yaxes(title_text="€ per 1,000 frames (log)", type="log", dtick=1, tickprefix="€",
                     tickformat=",.0f")
    return fig


def policy_fig(sc, cur):
    items = sc["policies"] + [cur]
    names = [p["policy"] for p in items]
    vals = [max(p["cost_per_1000"], 1e-6) for p in items]
    colours = [MUTED] * (len(items) - 1) + [ACCENT]
    hover = [f"{p['policy']}<br>{eur(p['cost_per_1000'])} / 1,000 · {eur(p['cost_per_day'])} / day<br>"
             f"review share {pct(p['review_share'], 2)} · FTE {p['fte_needed']:.1f}"
             f"{'' if p['feasible'] else ' (over capacity)'}" for p in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colours,
                           text=[eur(v) + ("" if p["feasible"] else " ⚠") for v, p in zip(vals, items)],
                           textposition="outside", textfont=dict(color=INK), hovertext=hover,
                           hoverinfo="text", cliponaxis=False))
    style(fig, 300, title=dict(text="Policy comparison, € per 1,000 frames (⚠ = over capacity)",
                               font=dict(size=13), x=0.01))
    fig.update_xaxes(type="log", title_text="€ per 1,000 frames (log)", dtick=1, tickprefix="€", tickformat=",.0f")
    fig.update_yaxes(autorange="reversed", showline=False, ticks="", tickfont=dict(color=INK, size=12))
    fig.update_layout(margin=dict(l=220, r=90, t=40, b=48))
    return fig


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def update_decide(lo, hi, view, *whatif):
    sc = scenario(*whatif)
    lo, hi = float(lo), float(max(lo, hi))
    cur = current_policy(lo, hi, sc)
    return kpi_html(cur, sc), matrix_fig(cur, view), final_html(cur), curve_fig(sc, lo, hi, cur), policy_fig(sc, cur)


def clamp_lo(lo, hi):
    return gr.update(value=max(lo, hi))


def clamp_hi(lo, hi):
    return gr.update(value=min(lo, hi))


def reset_recommended(*whatif):
    sc = scenario(*whatif)
    return sc["lo"], sc["hi"], False


def model_only(checked, lo, hi, *whatif):
    sc = scenario(*whatif)
    if checked:
        return sc["t_star"], sc["t_star"]
    return sc["lo"], sc["hi"]


def reset_whatif():
    return tuple(REPORT_D[k] for k in WHATIF_KEYS)


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(WEIGHTS / "best.pt"))
    return _MODEL


def detect(image):
    if image is None:
        return []
    result = get_model().predict(image, conf=PRED_CONF, verbose=False)[0]
    names = result.names
    return [{"cls": names[int(c)], "conf": float(cf), "xyxy": [float(v) for v in b]}
            for c, cf, b in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                result.boxes.xyxy.cpu().numpy())]


def class_colour(name):
    n = name.lower()
    return ALARM_C if ("fire" in n or "flame" in n) else "#475569" if "smoke" in n else "#94A3B8"


def render_frame(image, dets, lo, hi):
    if image is None:
        return None, '<div class="badge empty">Upload a frame to score it.</div>'
    lo, hi = float(lo), float(max(lo, hi))
    s_fire = max([d["conf"] for d in dets if "fire" in d["cls"].lower() or "flame" in d["cls"].lower()],
                 default=0.0)
    s_smoke = max([d["conf"] for d in dets if "smoke" in d["cls"].lower()], default=0.0)
    s = max(s_fire, s_smoke)
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width = max(3, round(min(img.size) / 160))
    try:
        font = ImageFont.load_default(size=max(12, round(min(img.size) / 40)))
    except TypeError:
        font = ImageFont.load_default()
    shown = sorted((d for d in dets if d["conf"] >= lo), key=lambda d: -d["conf"])[:MAX_BOXES_DRAWN]
    for d in reversed(shown):
        colour = class_colour(d["cls"])
        x0, y0, x1, y1 = d["xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        label = f" {d['cls']} {d['conf']:.2f} "
        tb = draw.textbbox((x0, y0), label, font=font)
        ty = y0 - (tb[3] - tb[1]) - 4 if y0 - (tb[3] - tb[1]) - 4 > 0 else y0
        tb = draw.textbbox((x0, ty), label, font=font)
        draw.rectangle([tb[0], tb[1] - 2, tb[2], tb[3] + 2], fill=colour)
        draw.text((x0, ty), label, fill="#FFFFFF", font=font)
    outcome = decide_frame(s, lo, hi)
    name, icon, colour = next(o for o in OUTCOMES if o[0] == outcome)
    badge = (f'<div class="badge" style="--c:{colour}"><div class="badge-main"><span class="icon">{icon}</span>'
             f'{name}</div><div class="badge-sub">s = {s:.3f} &nbsp;·&nbsp; s<sub>fire</sub> = {s_fire:.3f}'
             f' &nbsp;·&nbsp; s<sub>smoke</sub> = {s_smoke:.3f}<br>review from {lo:.2f}, alarm from {hi:.2f}'
             f' &nbsp;·&nbsp; {len(shown)} box(es) ≥ lo drawn</div></div>')
    return img, badge


# ---------------------------------------------------------------------------
# Static content (Model, Assumptions tabs)
# ---------------------------------------------------------------------------
def model_tab_html():
    ev, base, stats = ART["eval"], ART["baselines"], ART["dataset_stats"]
    inst = stats["splits"]["test"]["instances"]
    overall = (f'<div class="kpis compact"><div class="kpi"><div class="label">mAP50 (test)</div>'
               f'<div class="value">{num(ev["overall"]["mAP50"])}</div></div>'
               f'<div class="kpi"><div class="label">mAP50-95 (test)</div>'
               f'<div class="value">{num(ev["overall"]["mAP50-95"])}</div></div></div>')
    per_class = table(["Class", "Precision", "Recall", "AP50", "AP50-95", "Test instances"],
                      [[f"<b>{html.escape(c['class'])}</b>", num(c["precision"]), num(c["recall"]),
                        num(c["AP50"]), num(c["AP50-95"]), f"{inst.get(c['class'], c['test_instances']):,}"]
                       for c in ev["per_class"]])
    op = DEC["recommended_operating_point"]
    frame_rows = []
    for c in DEC["frame_level_per_class"]:
        for key, label in (("at_t_star", f"model alone, t*={DEC['t_star']:.3f}"),
                           ("flagged_at_lo", f"flagged (s ≥ lo={op['lo']:.2f})"),
                           ("alarm_at_hi", f"alarm (s ≥ hi={op['hi']:.2f})")):
            r = c[key]
            frame_rows.append([f"<b>{html.escape(c['class'])}</b>", label, num(r["precision"]),
                               num(r["recall"]), num(r["fpr"]), f"{r['TP']}/{r['FN']}/{r['FP']}/{r['TN']}"])
    frame_tbl = table(["Class", "Rule", "Precision", "Recall", "FPR", "TP/FN/FP/TN"], frame_rows)
    bl = [("Finetuned (model alone at t*)", base["finetuned"]), ("COCO zero-shot", base["coco_zero_shot"]),
          ("Predict nothing", base["predict_nothing"])]
    if "always_alarm" in base:
        bl.append(("Always alarm", base["always_alarm"]))
    base_tbl = table(["Model / policy", "mAP50", "Frame recall", "Frame FPR", "Cost / 1,000"],
                     [[f"<b>{n}</b>", num(b["mAP50"]), pct(b["frame_recall"]), pct(b["frame_fpr"]),
                       eur(b["cost_per_1000"])] for n, b in bl])
    coco = base["coco_zero_shot"]
    coco_note = (f'<p class="note">COCO zero-shot top classes on positive frames: '
                 f'{", ".join(f"{html.escape(n)} ({c})" for n, c in coco["top_classes_positive_frames"]) or "none"}'
                 f'. On negative frames: '
                 f'{", ".join(f"{html.escape(n)} ({c})" for n, c in coco["top_classes_negative_frames"]) or "none"}.</p>')
    return (overall + '<div class="sub-title">Detection, per class (test split)</div>' + per_class +
            f'<p class="note">{html.escape(ev["per_class_note"])}.</p>'
            '<div class="sub-title">Frame level, per class (raw test counts)</div>' + frame_tbl +
            '<div class="sub-title">Baselines on the same test split</div>' + base_tbl + coco_note)


def detection_cm_fig():
    cm = ART["eval"]["confusion_matrix"]
    m = np.array(cm["matrix"])
    labels = cm["labels"]
    fig = go.Figure(go.Heatmap(z=np.log1p(m), x=labels, y=labels, colorscale=[[0, "#F8FAFC"], [1, "#475569"]],
                               showscale=False, customdata=m,
                               hovertemplate="true %{y} → predicted %{x}: %{customdata:,}<extra></extra>"))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            fig.add_annotation(x=labels[j], y=labels[i], text=f"{m[i, j]:,}", showarrow=False,
                               font=dict(color="#FFFFFF" if np.log1p(m[i, j]) > 0.6 * np.log1p(m.max()) else INK,
                                         size=13))
    style(fig, 360, title=dict(text=f"Detection confusion matrix (test, conf ≥ {cm['conf']})",
                               font=dict(size=13), x=0.01))
    fig.update_xaxes(title_text="predicted", showline=False, ticks="")
    fig.update_yaxes(title_text="true", autorange="reversed", showline=False, ticks="")
    return fig


def dataset_html():
    stats = ART["dataset_stats"]
    sp, names = stats["splits"], stats["class_names"]
    split_tbl = table(["Split", "Images", "Background", "Fire-or-smoke prevalence"] + names,
                      [[s, f"{v['images']:,}", f"{v['background_images']:,} ({v['background_pct']:.1f}%)",
                        pct(v["positive_prevalence"])] + [f"{v['instances'][n]:,}" for n in names]
                       for s, v in sp.items()])
    box_tbl = table(["Class", "Median box area", "Share < 1% of image"],
                    [[n, pct(a["all"].get("median"), 2), pct(a["all"].get("small_share"))]
                     for n, a in stats["box_area_relative"].items()])
    low = stats["low_count_classes_test"]
    warn = (f'<p class="flag">⚠ Fewer than {stats["low_count_threshold"]} test instances: '
            f'{", ".join(low)}. Treat those per-class numbers as unreliable.</p>' if low else
            f'<p class="note">Every class has at least {stats["low_count_threshold"]} test instances.</p>')
    src = stats["source"]
    return (f'<p class="note">{html.escape(src["workspace"])}/{html.escape(src["project"])} '
            f'v{src["version"]} · licence {html.escape(str(src["licence"]))}</p>' + split_tbl + box_tbl + warn)


def assumptions_html():
    D = REPORT_D
    rows = [[f"<code>{k}</code>", f"{v:,}" if isinstance(v, (int, float)) else html.escape(str(v))]
            for k, v in D.items()]
    rows.append(["<code>cost_review</code> (derived)", f"€{derived(D)['cost_review']:.4f} per reviewed frame"])
    rows.append(["<code>frames_per_day</code> (derived)", f"{derived(D)['frames_per_day']:,}"])
    b = DEC["bootstrap"]
    ci = lambda x, f=eur: f"{f(x['mean'])} (95% CI {f(x['lo95'])} – {f(x['hi95'])})"
    boot = table(["Quantity", "Bootstrap estimate"], [
        [f"Cost / 1,000, model alone at t*={DEC['t_star']:.3f}", ci(b["model_alone_cost_per_1000"])],
        [f"Cost / 1,000, three-way ({DEC['recommended_operating_point']['lo']:.2f}, "
         f"{DEC['recommended_operating_point']['hi']:.2f})", ci(b["three_way_cost_per_1000"])],
        ["Frame recall at t*", ci(b["model_alone_recall"], pct)],
    ])
    return ('<div class="sub-title">DECISIONS used in training (read-only)</div>' +
            table(["Input", "Value"], rows) +
            f'<div class="sub-title">Uncertainty ({b["n_resamples"]} stratified bootstrap resamples)</div>' + boot)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
).set(block_shadow="none", block_border_width="1px")

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto; }
h1.title { font-size: 1.5rem; font-weight: 650; margin: 0.25rem 0 0; color: var(--body-text-color); }
p.lede { margin: 0.2rem 0 0.75rem; color: var(--body-text-color-subdued); }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.kpis.compact { grid-template-columns: repeat(2, minmax(0, 220px)); margin-bottom: 12px; }
@media (max-width: 760px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px 14px;
       background: var(--block-background-fill); }
.kpi .label { font-size: 0.78rem; color: var(--body-text-color-subdued); }
.kpi .value { font-size: 1.6rem; font-weight: 650; color: var(--body-text-color);
              font-variant-numeric: tabular-nums; line-height: 1.25; }
.kpi .sub { font-size: 0.78rem; color: var(--body-text-color-subdued); }
.kpi.warn { border-color: #D2491B; box-shadow: inset 3px 0 0 #D2491B; }
.flag { font-size: 0.8rem; font-weight: 600; color: var(--body-text-color); }
.sub-title { font-weight: 600; margin: 18px 0 6px; color: var(--body-text-color); }
.note { font-size: 0.82rem; color: var(--body-text-color-subdued); margin: 6px 0; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 0.88rem; font-variant-numeric: tabular-nums; }
table.tbl th { text-align: left; font-weight: 600; color: var(--body-text-color-subdued);
               border-bottom: 1px solid var(--border-color-primary); padding: 6px 10px; }
table.tbl td { border-bottom: 1px solid var(--border-color-primary); padding: 6px 10px;
               color: var(--body-text-color); }
table.tbl.small { max-width: 460px; }
.badge { border: 1px solid var(--c); border-left: 8px solid var(--c); border-radius: 10px;
         padding: 16px 18px; background: var(--block-background-fill); }
.badge.empty { --c: var(--border-color-primary); color: var(--body-text-color-subdued); }
.badge-main { font-size: 2rem; font-weight: 700; color: var(--body-text-color); }
.badge-main .icon { color: var(--c); margin-right: 10px; }
.badge-sub { margin-top: 6px; font-size: 0.88rem; color: var(--body-text-color-subdued);
             font-variant-numeric: tabular-nums; }
.legend-chip { display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; font-size: 0.85rem;
               color: var(--body-text-color); }
.legend-chip i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
"""

LEGEND = ('<div>' + "".join(f'<span class="legend-chip"><i style="background:{c}"></i>{icon} {n}</span>'
                            for n, icon, c in OUTCOMES) + '</div>')


def build_missing():
    with gr.Blocks(title="Smoke & Fire Triage") as demo:
        gr.HTML('<h1 class="title">Smoke & Fire Triage</h1>')
        gr.Markdown(
            "### The trained model and its artifacts are not here yet\n\n"
            f"Missing from `weights/`: {', '.join(f'`{m}`' for m in MISSING)}.\n\n"
            "Run the training pipeline first:\n\n```bash\nuv run python main.py train\n```")
    return demo


def build():
    sc0 = scenario(*(REPORT_D[k] for k in WHATIF_KEYS))
    with gr.Blocks(title="Smoke & Fire Triage") as demo:
        gr.HTML('<h1 class="title">Smoke & Fire Triage</h1>'
                '<p class="lede">Per-frame triage for fixed cameras: pass, send to a human, or raise an alarm. '
                'All figures are expected values at the production base rate, from the test-split scores.</p>')
        kpis = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Decide"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        lo = gr.Slider(0, 1, value=sc0["lo"], step=0.01, label="Review from (lo)",
                                       info="Frames scoring below lo pass.")
                        hi = gr.Slider(0, 1, value=sc0["hi"], step=0.01, label="Alarm from (hi)",
                                       info="Frames scoring at or above hi alarm; between lo and hi go to review.")
                    with gr.Column(scale=1, min_width=220):
                        reset = gr.Button("Reset to recommended", variant="primary")
                        only = gr.Checkbox(label="Model only (lo = hi = t*)", value=False)
                        view = gr.Radio(["Per 1,000 frames at base rate", "Raw test-set counts"],
                                        value="Per 1,000 frames at base rate", label="Matrix units")
                gr.HTML(LEGEND)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        matrix = gr.Plot(label="Pass / review / alarm by actual class", show_label=False)
                    with gr.Column(scale=2):
                        final = gr.HTML()
                curve = gr.Plot(label="Cost vs threshold", show_label=False)
                policies = gr.Plot(label="Policy comparison", show_label=False)

            with gr.Tab("Try a frame"):
                with gr.Row(equal_height=False):
                    with gr.Column():
                        upload = gr.Image(type="pil", label="Upload a camera frame", sources=["upload", "clipboard"])
                    with gr.Column():
                        annotated = gr.Image(type="pil", label="Detections with confidence ≥ lo",
                                             interactive=False)
                        badge = gr.HTML('<div class="badge empty">Upload a frame to score it.</div>')
                gr.Markdown("Decision uses the lo / hi sliders on the **Decide** tab and updates when they move.",
                            elem_classes="note")
                dets = gr.State([])

            with gr.Tab("Model"):
                gr.HTML(model_tab_html())
                gr.Plot(detection_cm_fig(), show_label=False)
                with gr.Accordion("Dataset summary", open=False):
                    gr.HTML(dataset_html())

            with gr.Tab("Assumptions"):
                gr.HTML(assumptions_html())
                gr.HTML('<div class="sub-title">What-if (does not change the report)</div>'
                        '<p class="note">Changes recompute t*, the optimum, KPIs and charts live.</p>')
                with gr.Row():
                    w_fn = gr.Number(REPORT_D["cost_fn"], label="cost_fn (€ per missed frame)", minimum=0)
                    w_fp = gr.Number(REPORT_D["cost_fp"], label="cost_fp (€ per false alarm)", minimum=0)
                    w_pi = gr.Number(REPORT_D["base_rate"], label="base_rate (share of positive frames)",
                                     minimum=1e-6, maximum=0.999)
                    w_rs = gr.Number(REPORT_D["review_seconds"], label="review_seconds (per frame)", minimum=1)
                reset_w = gr.Button("Reset to report values")

        whatif = [w_fn, w_fp, w_pi, w_rs]
        decide_in = [lo, hi, view, *whatif]
        decide_out = [kpis, matrix, final, curve, policies]
        frame_in, frame_out = [upload, dets, lo, hi], [annotated, badge]

        demo.load(update_decide, decide_in, decide_out)
        lo.release(clamp_lo, [lo, hi], hi).then(update_decide, decide_in, decide_out) \
            .then(render_frame, frame_in, frame_out)
        hi.release(clamp_hi, [lo, hi], lo).then(update_decide, decide_in, decide_out) \
            .then(render_frame, frame_in, frame_out)
        view.change(update_decide, decide_in, decide_out)
        reset.click(reset_recommended, whatif, [lo, hi, only]).then(update_decide, decide_in, decide_out) \
            .then(render_frame, frame_in, frame_out)
        only.input(model_only, [only, lo, hi, *whatif], [lo, hi]).then(update_decide, decide_in, decide_out) \
            .then(render_frame, frame_in, frame_out)
        for w in whatif:
            w.change(update_decide, decide_in, decide_out)
        reset_w.click(reset_whatif, None, whatif)
        upload.change(detect, upload, dets).then(render_frame, frame_in, frame_out)
    return demo


demo = build() if ART else build_missing()

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
