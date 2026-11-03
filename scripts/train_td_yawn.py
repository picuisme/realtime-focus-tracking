"""Label-free yawn detector by TD value prediction, for the web app.

Train (no labels): UTA-RLDD, 60 people, 3 Hz. The cumulant is the app's own rule
signal "mouth wide open" (MAR > 0.5 with a face), so the value V(s) is the
discounted amount of wide-open mouth expected in the next ~2 s.

Threshold (no labels): chosen on UTA-RLDD so that the TD detector raises as many
yawn alerts per hour as the current rule (mouth open >= 1.2 s) does there.

Test (labels, never used for training or the threshold): DMD drowsiness
recordings, 13 people, yawns annotated per frame, down-sampled to 3 Hz like the
app. Compares the current rule, the TD detector as shipped, and the TD detector
that keeps learning on each recording with TD(lambda) (what the app does).

    python scripts/train_td_yawn.py --uta data/uta_rldd_features.csv --dmd /path/to/dmd \
        --export web/models/td_yawn.json --report reports/td_yawn.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from focus.early_warning import FEATURES, OnlineTD, Scaler, gamma_for, lstd_lambda, reference, state_features  # noqa: E402

HZ = 3.0
TAU_S = 2.0
LAMBDA = 0.8
ALPHA = 0.002
YAWN_MAR = 0.5
RULE_OPEN_S = 1.2
REFRACTORY_S = 3.0


def uta_sessions(csv):
    df = pd.read_csv(csv, usecols=["session_id", "subject_id", "timestamp", "ear_score", "mar_score", "gaze_ratio",
                                   "person_detected", "mp_pitch", "mp_yaw"])
    for sid, g in df.groupby("session_id", sort=False):
        g = g.sort_values("timestamp")
        yield pd.DataFrame({"t": g.timestamp.values, "person": g.person_detected.values, "ear": g.ear_score.values,
                            "mar": g.mar_score.values, "gaze": g.gaze_ratio.values, "pitch": g.mp_pitch.fillna(0).values,
                            "yaw": g.mp_yaw.fillna(0).values})


def cumulant(df):
    return ((df.person.values == 1) & (df.mar.values > YAWN_MAR)).astype(float)


def rule_alerts(df):
    """The app's rule: count a yawn once the mouth has been wide open for 1.2 s."""
    c, dt = cumulant(df), 1 / HZ
    out, run, counted = [], 0.0, False
    for i, v in enumerate(c):
        if v:
            run += dt
            if run >= RULE_OPEN_S and not counted:
                out.append(i); counted = True
        else:
            run, counted = 0.0, False
    return np.array(out, int)


def alerts(score, thr):
    above = score > thr
    edges = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    out, last = [], -1e9
    for i in edges:
        if (i - last) / HZ >= REFRACTORY_S:
            out.append(i); last = i
    return np.array(out, int)


def per_hour(ds, al):
    return sum(len(a) for a in al) / (sum(len(d) for d in ds) / HZ / 3600)


def online_scores(model, X, c):
    td = OnlineTD(w=model["w"].copy(), scaler=model["scaler"], gamma=model["gamma"], lam=LAMBDA, alpha=ALPHA)
    out = np.empty(len(X))
    for t in range(len(X)):
        out[t] = td.step(X[t], c[t - 1] if t else 0.0)
    return out


def dmd_sessions(dmd: Path):
    import re
    from evaluate_early_warning import frame_mask, merge_and_filter
    for csv in sorted((dmd / "feat" / "drowsiness").glob("*.csv")):
        js = dmd / "ann" / "drowsiness" / f"{csv.stem}_rgb_ann_drowsiness.json"
        if not js.exists():
            continue
        full = pd.read_csv(csv)
        d = full.iloc[::3].reset_index(drop=True)          # ~10 Hz -> ~3.3 Hz, like the app
        d = d.rename(columns={})[["t", "person", "ear", "mar", "gaze", "pitch", "yaw"]]
        ann = json.loads(js.read_text())
        m = frame_mask(ann, ["yawning/Yawning with hand", "yawning/Yawning without hand"], int(full.frame.max()) + 1)
        gt = merge_and_filter(m[np.clip(full.frame.values[::3], 0, len(m) - 1)], 1 / HZ, 2.0)
        yield re.match(r"(g[A-Z]_\d+)", csv.stem).group(1), d, gt


def event_metrics(gt, al, hz=HZ, early_s=3.0):
    from evaluate_early_warning import runs
    ev = runs(gt)
    used = np.zeros(len(al), bool)
    lat = []
    for s, e in ev:
        hit = np.where((al >= s - early_s * hz) & (al < e))[0]
        if len(hit):
            lat.append((al[hit[0]] - s) / hz); used[hit] = True
    return len(ev), lat, int((~used).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uta", required=True)
    ap.add_argument("--dmd", type=Path)
    ap.add_argument("--export", type=Path, default=ROOT / "web" / "models" / "td_yawn.json")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "td_yawn.json")
    a = ap.parse_args()

    U = list(uta_sessions(a.uta))
    gamma = gamma_for(TAU_S, 1 / HZ)
    Xs = [state_features(d, reference(d)) for d in U]
    sc = Scaler.fit(np.vstack(Xs))
    w = lstd_lambda(((sc(X), cumulant(d)) for X, d in zip(Xs, U)), gamma, LAMBDA)
    model = {"w": w, "scaler": sc, "gamma": gamma}
    V = [sc(X) @ w * (1 - gamma) for X in Xs]

    target = per_hour(U, [rule_alerts(d) for d in U])
    cands = np.unique(np.quantile(np.concatenate(V), np.linspace(0.9, 0.99999, 400)))
    thr = min(cands, key=lambda t: abs(per_hour(U, [alerts(v, t) for v in V]) - target))
    rep = {"protocol": __doc__.split("    python")[0].strip(),
           "train": {"dataset": "UTA-RLDD", "sessions": len(U), "hours": round(sum(len(d) for d in U) / HZ / 3600, 1)},
           "threshold": float(thr), "rule_alerts_per_hour_uta": target,
           "td_alerts_per_hour_uta": per_hour(U, [alerts(v, thr) for v in V])}

    if a.dmd:
        res = {k: {"events": 0, "lat": [], "fa": 0, "hours": 0.0} for k in ("rule", "td", "td_online")}
        for subj, d, gt in dmd_sessions(a.dmd):
            X = state_features(d, reference(d))
            c = cumulant(d)
            outs = {"rule": rule_alerts(d), "td": alerts(sc(X) @ w * (1 - gamma), thr),
                    "td_online": alerts(online_scores(model, X, c), thr)}
            for k, al in outs.items():
                n, lat, fa = event_metrics(gt, al)
                r = res[k]; r["events"] += n; r["lat"] += lat; r["fa"] += fa; r["hours"] += len(d) / HZ / 3600
        rep["test_dmd_yawn"] = {k: {"events": r["events"], "recall": len(r["lat"]) / max(r["events"], 1),
                                    "median_latency_s": float(np.median(r["lat"])) if r["lat"] else None,
                                    "false_alarms_per_h": r["fa"] / r["hours"]} for k, r in res.items()}

    a.export.parent.mkdir(parents=True, exist_ok=True)
    a.export.write_text(json.dumps({"features": FEATURES, "mean": sc.mean.tolist(), "scale": sc.scale.tolist(),
                                    "w": w.tolist(), "gamma": gamma, "lambda": LAMBDA, "alpha": ALPHA, "hz": HZ,
                                    "tau_s": TAU_S, "threshold": float(thr), "cumulant": "mar > 0.5 with a face",
                                    "trained_on": "UTA-RLDD (60 people), no labels"}, indent=1))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
