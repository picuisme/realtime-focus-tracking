"""Train and evaluate the drowsiness detector on UTA-RLDD features.

Labels come from the dataset (alert / low-vigilant / drowsy per video), never
from the rules. Evaluation is grouped by person: UTA's own 5 folds when all are
present, otherwise GroupKFold over subjects.

    python scripts/train_drowsiness.py --csv data/uta_rldd_features.csv \
        --report reports/drowsiness.json --export web/models/drowsiness.json

Models compared
- perclos   : the PERCLOS-60 s value itself (no training) — the classic baseline
- logistic  : standardised logistic regression on the 13 window features (shipped to the web)
- lightgbm  : gradient-boosted trees, to show how much a non-linear model adds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from focus.metrics import grouped_bootstrap_ci  # noqa: E402
from focus.temporal import FEATURES, session_windows  # noqa: E402

SEED = 0


def build(csv_paths):
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    rows = []
    for sid, g in df.groupby("session_id", sort=False):
        g = g.sort_values("timestamp")
        X, idx = session_windows(g)
        if not len(X):
            continue
        lab = g["gt_label"].iloc[0]
        subj = g["subject_id"].iloc[0]
        f = pd.DataFrame(X, columns=FEATURES)
        f["t"] = g["timestamp"].values[idx]
        f["session"], f["subject"], f["label3"] = sid, subj, lab
        f["fold"] = str(subj).split("/")[0] if "/" in str(subj) else "na"
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def models():
    out = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, random_state=SEED))}
    try:
        from lightgbm import LGBMClassifier
        out["lightgbm"] = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                                         verbose=-1)
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "drowsiness.json")
    ap.add_argument("--export", type=Path, default=ROOT / "web" / "models" / "drowsiness.json")
    a = ap.parse_args()

    W = build(a.csv)
    data = W[W.label3.isin([0, 2])].copy()             # alert vs drowsy; low-vigilant kept for the 3-class view
    data["y"] = (data.label3 == 2).astype(int)
    X, y, groups = data[FEATURES].values, data.y.values, data.subject.values
    folds = data.fold.values if data.fold.nunique() >= 3 and "na" not in set(data.fold) else None
    n_subj = len(np.unique(groups))
    splitter = list(GroupKFold(n_splits=min(5, n_subj)).split(X, y, folds if folds is not None else groups))

    oof = {"perclos": data["perclos60"].values.astype(float)}
    for name, m in models().items():
        p = np.zeros(len(y))
        for tr, te in splitter:
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        oof[name] = p

    report = {"task": "UTA-RLDD alert (0) vs drowsy (10), windows of 60 s every 1 s",
              "subjects": int(n_subj), "videos": int(data.session.nunique()), "windows": int(len(y)),
              "split": "UTA folds" if folds is not None else f"GroupKFold by subject ({len(splitter)} folds)",
              "features": FEATURES, "models": {}}
    for name, s in oof.items():
        auc, lo, hi = grouped_bootstrap_ci(y, s, groups)
        vid = data.assign(s=s).groupby("session").agg(s=("s", "mean"), y=("y", "first"))
        vauc = roc_auc_score(vid.y, vid.s) if vid.y.nunique() == 2 else float("nan")
        thr = np.median(s)
        report["models"][name] = {"window_auroc": auc, "window_auroc_ci95": [lo, hi], "video_auroc": float(vauc),
                                  "balanced_accuracy_at_median": float(balanced_accuracy_score(y, s > thr))}

    # low-vigilant videos: where does the shipped model put them?
    final = models()["logistic"].fit(X, y)
    mid = W[W.label3 == 1]
    if len(mid):
        sc = final.predict_proba(mid[FEATURES].values)[:, 1]
        report["low_vigilant_mean_score"] = float(sc.mean())
        report["alert_mean_score"] = float(final.predict_proba(W[W.label3 == 0][FEATURES].values)[:, 1].mean())
        report["drowsy_mean_score"] = float(final.predict_proba(W[W.label3 == 2][FEATURES].values)[:, 1].mean())

    sc_, lr = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
    export = {"features": FEATURES, "mean": sc_.mean_.tolist(), "scale": sc_.scale_.tolist(),
              "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
              "trained_on": report["task"], "subjects": report["subjects"],
              "window_auroc_oof": report["models"]["logistic"]["window_auroc"]}
    report["logistic_coefficients"] = dict(zip(FEATURES, map(float, lr.coef_[0])))
    for p, obj in [(a.report, report), (a.export, export)]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
