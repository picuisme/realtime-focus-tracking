"""Final, one-shot evaluation of the drowsiness detector on held-out people.

Protocol (fixed before the held-out folds were looked at):
- Development set: UTA-RLDD folds 1-2 (18 people). All model choices were made
  there with 5-fold cross-validation by person (scripts/train_drowsiness.py).
- Held-out set: UTA-RLDD folds 3-5 (36 people), used once, here.
- Candidates, all trained on the development set only:
    perclos        PERCLOS-60 s alone, mapped to 0-1 by a logistic fit (same ranking as raw PERCLOS)
    logit_eye      standardised logistic regression on the 5 eye features, C = 0.03
    logit_all      the same on all 13 window features, C = 0.03 (shown for comparison)
    lightgbm       gradient-boosted trees on all 13 features (comparison)
- Decision rule: ship logit_eye only if its held-out AUROC is higher than perclos's;
  otherwise ship perclos. The shipped model is then refit on all 54 people.

    python scripts/evaluate_holdout.py --csv data/uta_rldd_features.csv \
        --report reports/drowsiness_holdout.json --export web/models/drowsiness.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from focus.metrics import grouped_bootstrap_ci  # noqa: E402
from focus.temporal import FEATURES  # noqa: E402
from train_drowsiness import best_threshold, build  # noqa: E402

DEV_FOLDS = {"Fold1", "Fold2"}
EYE = ["ear_rel_mean20", "ear_rel_std20", "perclos60", "closures_per_min60", "longest_closure60"]
CANDIDATES = {"perclos": (["perclos60"], 1.0), "logit_eye": (EYE, 0.03), "logit_all": (FEATURES, 0.03)}


def fit_logit(cols, C, X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=0)).fit(X[cols], y)


def export(model, cols, path: Path, meta: dict):
    """Write the model over the full 13-feature vector the web app computes
    (unused features get weight 0), so web/core.js needs no change."""
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    coef, mean, scale = [0.0] * len(FEATURES), [0.0] * len(FEATURES), [1.0] * len(FEATURES)
    for k, c in enumerate(cols):
        i = FEATURES.index(c)
        coef[i], mean[i], scale[i] = float(lr.coef_[0][k]), float(sc.mean_[k]), float(sc.scale_[k])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": FEATURES, "used": cols, "mean": mean, "scale": scale, "coef": coef,
                                "intercept": float(lr.intercept_[0]), **meta}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "drowsiness_holdout.json")
    ap.add_argument("--export", type=Path, default=ROOT / "web" / "models" / "drowsiness.json")
    a = ap.parse_args()

    W = build(a.csv)
    W = W[W.label3.isin([0, 2])].copy()
    W["y"] = (W.label3 == 2).astype(int)
    dev, test = W[W.fold.isin(DEV_FOLDS)], W[~W.fold.isin(DEV_FOLDS)]
    if test.empty:
        raise SystemExit("no held-out folds in the data yet")

    scores, thresholds = {}, {}
    for name, (cols, C) in CANDIDATES.items():
        m = fit_logit(cols, C, dev, dev.y)
        scores[name] = m.predict_proba(test[cols])[:, 1]
        thresholds[name] = best_threshold(dev.y.values, m.predict_proba(dev[cols])[:, 1])
    try:
        from lightgbm import LGBMClassifier
        lg = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1).fit(dev[FEATURES], dev.y)
        scores["lightgbm"] = lg.predict_proba(test[FEATURES])[:, 1]
        thresholds["lightgbm"] = best_threshold(dev.y.values, lg.predict_proba(dev[FEATURES])[:, 1])
    except ImportError:
        pass

    y, g = test.y.values, test.subject.values
    rep = {"protocol": __doc__.split("    python")[0].strip(),
           "development": {"subjects": int(dev.subject.nunique()), "videos": int(dev.session.nunique()), "windows": len(dev)},
           "held_out": {"subjects": int(test.subject.nunique()), "videos": int(test.session.nunique()), "windows": len(test),
                        "folds": sorted(test.fold.unique().tolist())},
           "models": {}}
    for name, s in scores.items():
        auc, lo, hi = grouped_bootstrap_ci(y, s, g)
        vid = test.assign(s=s).groupby("session").agg(s=("s", "mean"), y=("y", "first"))
        rep["models"][name] = {"window_auroc": auc, "window_auroc_ci95": [lo, hi],
                               "video_auroc": float(roc_auc_score(vid.y, vid.s)),
                               "balanced_accuracy": float(balanced_accuracy_score(y, s > thresholds[name])),
                               "threshold_from_dev": thresholds[name]}
    shipped = "logit_eye" if rep["models"]["logit_eye"]["window_auroc"] > rep["models"]["perclos"]["window_auroc"] else "perclos"
    rep["decision"] = {"rule": "ship logit_eye only if its held-out AUROC beats perclos", "shipped": shipped}

    cols, C = CANDIDATES[shipped]
    final = fit_logit(cols, C, W, W.y)          # refit on all labelled people for deployment
    export(final, cols, a.export, {"model": shipped, "subjects": int(W.subject.nunique()),
                                   "window_auroc_oof": rep["models"][shipped]["window_auroc"],
                                   "trained_on": "UTA-RLDD alert vs drowsy, 60 s windows",
                                   "evaluated": "held-out UTA-RLDD folds, see reports/drowsiness_holdout.json"})
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
