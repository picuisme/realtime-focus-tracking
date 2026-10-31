"""Does the study-mode signal track engagement on DAiSEE?

DAiSEE: 9 068 ten-second clips of 112 students watching lectures, each with an
engagement level 0-3 from crowd annotators. Low engagement (0-1) is rare (~6 %).
Clips are too short for the 60 s window features, so each clip is summarised by
simple per-clip statistics, relative to the person's own median over their clips.

Scores compared on the official Test split (people never seen in Train/Validation):
  rule_fraction   share of samples the shipped study rule calls distracted (no training)
  logistic        logistic regression on the clip summaries, trained on Train + Validation

    python scripts/evaluate_daisee.py --csv data/daisee_features.csv --report reports/daisee.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from focus.metrics import grouped_bootstrap_ci  # noqa: E402

FEATS = ["ear_rel", "ear_std", "closed_frac", "mar_mean", "abs_yaw", "abs_pitch", "pose_std", "gaze_dev",
         "gaze_std", "face_missing", "rule_fraction"]


def clip_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    face = df.person_detected == 1
    df["yaw"] = df.mp_yaw.where(face)
    df["pitch"] = df.mp_pitch.where(face)
    df["ear"] = df.ear_score.where(face & (df.ear_score > 0))
    ref = df.groupby("subject_id")[["ear", "yaw", "pitch"]].median().add_prefix("ref_")
    df = df.join(ref, on="subject_id")
    df["ear_r"] = df.ear / df.ref_ear
    g = df.groupby("clip_id")
    t = pd.DataFrame({
        "subject": g.subject_id.first(), "split": g.source.first().str.replace("daisee-", ""),
        "engagement": g.gt_label.first(),
        "ear_rel": g.ear_r.mean(), "ear_std": g.ear_r.std(), "closed_frac": g.ear_r.apply(lambda s: (s < 0.7).mean()),
        "mar_mean": g.mar_score.mean(),
        "abs_yaw": g.apply(lambda x: (x.yaw - x.ref_yaw).abs().mean(), include_groups=False),
        "abs_pitch": g.apply(lambda x: (x.pitch - x.ref_pitch).abs().mean(), include_groups=False),
        "pose_std": g.yaw.std().fillna(0) + g.pitch.std().fillna(0),
        "gaze_dev": g.gaze_ratio.apply(lambda s: (s - 0.5).abs().mean()), "gaze_std": g.gaze_ratio.std(),
        "face_missing": 1 - g.person_detected.mean(), "rule_fraction": g.is_distracted_label.mean(),
    })
    t = t.dropna(subset=["engagement"])
    t[FEATS] = t[FEATS].fillna(t[FEATS].median())
    t["low"] = (t.engagement <= 1).astype(int)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "daisee.json")
    a = ap.parse_args()
    t = clip_table(pd.read_csv(a.csv))
    tr, te = t[t.split != "Test"], t[t.split == "Test"]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)).fit(tr[FEATS], tr.low)
    scores = {"rule_fraction": te.rule_fraction.values, "logistic": m.predict_proba(te[FEATS])[:, 1]}
    rep = {"protocol": __doc__.split("    python")[0].strip(),
           "train": {"clips": len(tr), "people": int(tr.subject.nunique()), "low_share": float(tr.low.mean())},
           "test": {"clips": len(te), "people": int(te.subject.nunique()), "low_share": float(te.low.mean())},
           "models": {}}
    for k, s in scores.items():
        auc, lo, hi = grouped_bootstrap_ci(te.low.values, s, te.subject.values)
        rep["models"][k] = {"auroc": auc, "auroc_ci95": [lo, hi],
                            "average_precision": float(average_precision_score(te.low, s)),
                            "chance_average_precision": float(te.low.mean())}
    coef = m.named_steps["logisticregression"].coef_[0]
    rep["logistic_weights"] = dict(sorted(zip(FEATS, coef.round(3).tolist()), key=lambda kv: -abs(kv[1])))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
