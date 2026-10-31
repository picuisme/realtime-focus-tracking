"""Does TD value prediction warn earlier than the current rules, at the same false-alarm rate?

Data: DMD (Vicomtech Driver Monitoring Dataset), RGB face camera, frame-level
annotations. Features come from scripts/extract_features-style MediaPipe runs
(one CSV per session with columns frame,t,person,ear,mar,gaze,brow,pitch,yaw,roll)
plus the session's OpenLABEL JSON. DMD is CC BY-NC-ND, so neither the features
nor the labels are committed; only the report is.

Tasks
  microsleep  eyes annotated closed for >= 1 s           (DMD drowsiness, s5)
  yawn        yawning annotated for >= 2 s                (DMD drowsiness, s5)
  off_road    gaze annotated not on road for >= 2 s       (DMD distraction, s1-s4)

Methods, all scored per sample and turned into alarms by a threshold
  rule        the shipped DriverRules: alarm once the rule state lasts long enough (reactive)
  instant     logistic regression: "is the event happening now?"
  supervised  logistic regression: "will the event be happening within the next H s?"
  td_label    LSTD(lambda) value of the annotated event cumulant (gamma: ~2 s ahead)
  td_rule     LSTD(lambda) value of the rule's own cumulant: needs no labels, so it
              can keep learning on each user in the app

Thresholds for the learned methods are set on the training people so that their
false alarms per hour match the rule's on those people; the test people are
then scored once. 5-fold cross-validation grouped by person.

    python scripts/evaluate_early_warning.py --dmd /path/to/dmd --report reports/early_warning_dmd.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from focus.early_warning import FEATURES, Scaler, gamma_for, lstd_lambda, reference, rule_flags, state_features  # noqa: E402

TAU_S = 2.0      # value horizon
H_S = 3.0        # "early" window for the supervised target and for counting a detection
LAMBDA = 0.8
MERGE_GAP_S = 0.3

TASKS = {
    "microsleep": dict(kind="drowsiness", actions=["eyes_state/close"], min_s=1.0, rule="closed", rule_s=1.5),
    "yawn": dict(kind="drowsiness", actions=["yawning/Yawning with hand", "yawning/Yawning without hand"], min_s=2.0,
                 rule="yawn", rule_s=1.0),
    "off_road": dict(kind="distraction", actions=["gaze_on_road/not_looking_road"], min_s=2.0, rule="off", rule_s=2.0),
}


# ── data ──────────────────────────────────────────────────────────────────

def runs(mask):
    out, start = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i)); start = None
    return out


def merge_and_filter(mask, dt, min_s):
    """Close gaps shorter than MERGE_GAP_S, then keep runs of at least min_s."""
    m = mask.copy()
    rr = runs(m)
    for (s0, e0), (s1, e1) in zip(rr, rr[1:]):
        if (s1 - e0) * dt < MERGE_GAP_S:
            m[e0:s1] = True
    keep = np.zeros_like(m)
    for s, e in runs(m):
        if (e - s) * dt >= min_s:
            keep[s:e] = True
    return keep


def frame_mask(js: dict, types: list[str], n_frames: int) -> np.ndarray:
    m = np.zeros(n_frames + 1, bool)
    for a in js["openlabel"]["actions"].values():
        if a["type"] in types:
            for iv in a.get("frame_intervals", []):
                m[iv["frame_start"]: iv["frame_end"] + 1] = True
    return m


def load(dmd: Path, kind: str):
    sessions = []
    for csv in sorted((dmd / "feat" / kind).glob("*.csv")):
        name = csv.stem
        js_path = dmd / "ann" / kind / f"{name}_rgb_ann_{kind}.json"
        if not js_path.exists():
            continue
        df = pd.read_csv(csv)
        if len(df) < 200:
            continue
        js = json.loads(js_path.read_text())
        shift = js["openlabel"]["streams"]["face_camera"]["stream_properties"].get("sync", {}).get("frame_shift", 0)
        subject = re.match(r"(g[A-Z]_\d+)", name).group(1)
        sessions.append(dict(name=name, subject=subject, df=df, js=js, shift=shift))
    return sessions


def prepare(sess, task):
    df = sess["df"]
    dt = float(np.median(np.diff(df.t.values)))
    ref = reference(df)
    X = state_features(df, ref)
    m = frame_mask(sess["js"], task["actions"], int(df.frame.max()) + abs(sess["shift"]) + 1)
    idx = np.clip(df.frame.values + sess["shift"], 0, len(m) - 1)
    gt = merge_and_filter(m[idx], dt, task["min_s"])
    closed, off = rule_flags(df, ref)
    raw = {"closed": closed, "off": off, "yawn": (df.person.values == 1) & (df.mar.values > 0.5)}[task["rule"]]
    run = np.zeros(len(raw))
    for i in range(len(raw)):
        run[i] = (run[i - 1] + dt if i else dt) if raw[i] else 0.0
    h = int(round(H_S / dt))
    fut = np.array([gt[i + 1: i + 1 + h].any() for i in range(len(gt))])
    return dict(X=X, gt=gt, fut=fut, rule_raw=raw.astype(float), rule_score=run, dt=dt,
                subject=sess["subject"], hours=len(df) * dt / 3600)


# ── alarms and event metrics ──────────────────────────────────────────────

def alarm_onsets(score, thr, dt, refractory_s=3.0):
    above = score > thr
    edges = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    on, last = [], -1e9
    for i in edges:
        if (i - last) * dt >= refractory_s:
            on.append(i); last = i
    return np.array(on, int)


def event_scores(p, score, thr):
    """Recall, latency (alarm time minus event start, s; negative = early) and false alarms."""
    dt = p["dt"]
    al = alarm_onsets(score, thr, dt)
    ev = runs(p["gt"])
    h = int(round(H_S / dt))
    lat, used = [], np.zeros(len(al), bool)
    for s, e in ev:
        hit = np.where((al >= s - h) & (al < e))[0]
        if len(hit):
            lat.append((al[hit[0]] - s) * dt)
            used[hit] = True
    return dict(n_events=len(ev), detected=len(lat), latencies=lat, false_alarms=int((~used).sum()), hours=p["hours"])


def summarise(parts):
    n = sum(r["n_events"] for r in parts)
    det = sum(r["detected"] for r in parts)
    lat = np.concatenate([r["latencies"] for r in parts if r["latencies"]]) if det else np.array([])
    hours = sum(r["hours"] for r in parts)
    fa = sum(r["false_alarms"] for r in parts)
    return dict(events=n, recall=det / max(n, 1), median_latency_s=float(np.median(lat)) if len(lat) else None,
                early_share=float((lat < 0).mean()) if len(lat) else None,
                false_alarms_per_h=fa / max(hours, 1e-9), hours=hours)


def fa_rate(ps, scores, thr):
    r = [event_scores(p, s, thr) for p, s in zip(ps, scores)]
    return sum(x["false_alarms"] for x in r) / max(sum(x["hours"] for x in r), 1e-9)


def match_threshold(ps, scores, target):
    """Lowest threshold whose false-alarm rate on these sessions is <= target."""
    allv = np.concatenate(scores)
    cands = np.unique(np.quantile(allv, np.linspace(0.5, 0.9995, 200)))
    for thr in cands:
        if fa_rate(ps, scores, thr) <= target:
            return float(thr)
    return float(cands[-1])


FA_GRID = [5, 10, 20, 40, 80]


def curve(pairs):
    """Recall and median latency at fixed false alarms per hour, over the pooled
    out-of-fold scores. Comparing whole curves avoids depending on one threshold."""
    ps = [p for p, _ in pairs]; ss = [s for _, s in pairs]
    allv = np.concatenate(ss)
    cands = np.unique(np.quantile(allv, np.linspace(0.3, 0.9999, 300)))
    rows = []
    for thr in cands:
        r = summarise([event_scores(p, s, thr) for p, s in zip(ps, ss)])
        rows.append((r["false_alarms_per_h"], r["recall"], r["median_latency_s"]))
    out = {}
    for fa in FA_GRID:
        ok = [r for r in rows if r[0] <= fa]
        best = max(ok, key=lambda r: r[1]) if ok else (None, 0.0, None)
        out[str(fa)] = {"recall": best[1], "median_latency_s": best[2]}
    return out


# ── models ────────────────────────────────────────────────────────────────

def fit_logit(ps, sc, key):
    X = np.vstack([sc(p["X"]) for p in ps]); y = np.concatenate([p[key] for p in ps])
    return LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(X, y)


def run_task(name, task, sessions):
    P = [prepare(s, task) for s in sessions]
    P = [p for p in P if p["gt"].any() or True]
    groups = np.array([p["subject"] for p in P])
    gamma = gamma_for(TAU_S, float(np.median([p["dt"] for p in P])))
    methods = ["rule", "instant", "supervised", "td_label", "td_rule"]
    per = {m: [] for m in methods}
    oof = {m: [] for m in methods}
    auc_y, auc_s = [], {m: [] for m in methods}
    weights = []
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(P, groups=groups):
        Ptr, Pte = [P[i] for i in tr], [P[i] for i in te]
        sc = Scaler.fit(np.vstack([p["X"] for p in Ptr]))
        models = {
            "instant": fit_logit(Ptr, sc, "gt"),
            "supervised": fit_logit(Ptr, sc, "fut"),
        }
        w_lab = lstd_lambda(((sc(p["X"]), p["gt"].astype(float)) for p in Ptr), gamma, LAMBDA)
        w_rule = lstd_lambda(((sc(p["X"]), p["rule_raw"]) for p in Ptr), gamma, LAMBDA)
        weights.append(w_lab)

        def score(m, p):
            if m == "rule":
                return p["rule_score"]
            if m in models:
                return models[m].predict_proba(sc(p["X"]))[:, 1]
            w = w_lab if m == "td_label" else w_rule
            return sc(p["X"]) @ w * (1 - gamma)

        rule_thr = task["rule_s"] - 1e-6
        target = fa_rate(Ptr, [p["rule_score"] for p in Ptr], rule_thr)
        for m in methods:
            thr = rule_thr if m == "rule" else match_threshold(Ptr, [score(m, p) for p in Ptr], target)
            for p in Pte:
                s = score(m, p)
                per[m].append(event_scores(p, s, thr))
                oof[m].append((p, s))
                keep = ~p["gt"]
                auc_s[m].append(s[keep])
        for p in Pte:
            auc_y.append(p["fut"][~p["gt"]])
    y = np.concatenate(auc_y)
    out = {"sessions": len(P), "subjects": int(len(set(groups))), "gamma": gamma, "tau_s": TAU_S, "H_s": H_S,
           "methods": {}}
    for m in methods:
        out["methods"][m] = summarise(per[m])
        out["methods"][m]["auroc_next_H_s"] = float(roc_auc_score(y, np.concatenate(auc_s[m]))) if y.any() else None
        out["methods"][m]["curve"] = curve(oof[m])
    w = np.mean(weights, 0)
    out["td_label_top_weights"] = sorted(zip(FEATURES, w.round(3).tolist()), key=lambda t: -abs(t[1]))[:8]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmd", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "early_warning_dmd.json")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    a = ap.parse_args()
    cache, rep = {}, {"protocol": __doc__.split("    python")[0].strip(), "tasks": {}}
    for name in a.tasks:
        t = TASKS[name]
        if t["kind"] not in cache:
            cache[t["kind"]] = load(a.dmd, t["kind"])
        if not cache[t["kind"]]:
            print("no data for", name); continue
        rep["tasks"][name] = run_task(name, t, cache[t["kind"]])
        print(name, json.dumps(rep["tasks"][name]["methods"], indent=1))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
