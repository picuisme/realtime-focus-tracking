"""Analyse the within-subject study (docs/STUDY_PROTOCOL.md).

Reads every summary_*.json (and optional decisions_*.csv, quiz.csv) under --dir,
computes per-condition means and paired differences with bootstrap CIs
resampled by participant, plus a mixed model when statsmodels is installed.

    python scripts/analyze_study.py --dir study_data/ --out reports/study.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METRICS = ["focused_minutes", "mean_return_s", "nudges_per_hour", "useful_nudge_rate", "not_helpful",
           "self_rated_focus", "self_rated_annoyance", "quiz_score"]
PAIRS = [("C", "B"), ("B", "A"), ("C", "A")]


def load(d: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(d.rglob("summary_*.json"))]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no summary_*.json under {d}")
    df["started_at"] = pd.to_datetime(df["started_at"])
    df = df.sort_values("started_at")
    df["session_order"] = df.groupby("participant").cumcount() + 1
    quiz = d / "quiz.csv"
    if quiz.exists():
        df = df.merge(pd.read_csv(quiz), on=["participant", "condition"], how="left")
    return df


def paired(df, metric, a, b, n=2000, seed=0):
    w = df.pivot_table(index="participant", columns="condition", values=metric, aggfunc="mean")
    if a not in w or b not in w:
        return None
    diff = (w[a] - w[b]).dropna().values
    if len(diff) < 2:
        return None
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"n": int(len(diff)), "mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/study.json"))
    a = ap.parse_args()
    df = load(a.dir)
    present = [m for m in METRICS if m in df.columns]
    rep = {"participants": int(df.participant.nunique()), "sessions": int(len(df)),
           "by_condition": df.groupby("condition")[present].mean().round(3).to_dict(orient="index"), "paired": {}}
    for m in present:
        rep["paired"][m] = {f"{x}-{y}": paired(df, m, x, y) for x, y in PAIRS}
    try:
        import statsmodels.formula.api as smf
        rep["mixed_models"] = {}
        for m in present:
            sub = df.dropna(subset=[m])
            if sub.participant.nunique() >= 4:
                fit = smf.mixedlm(f"{m} ~ C(condition, Treatment('B')) + session_order", sub, groups=sub["participant"]).fit()
                rep["mixed_models"][m] = {k: [float(v), float(fit.pvalues[k])] for k, v in fit.params.items() if "condition" in k}
    except ImportError:
        rep["mixed_models"] = "statsmodels not installed"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
