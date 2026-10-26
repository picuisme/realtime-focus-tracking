"""Off-policy evaluation and a new starting policy from real nudge logs.

Input: decisions_*.csv exported by the web app or the desktop app (columns
action, p, why, reward, x). Only rows with why == 'policy' are used: those are
the decisions the logging policy actually made, with known propensities.

    python scripts/fit_policy_from_logs.py --dir study_data/ --out reports/ope.json \
        --export-prior web/models/nudge_prior.json

Reports IPS / SNIPS / DR estimates (with participant-bootstrap CIs) for: never nudge, the v2.1 rule, and the
greedy policy of a LinUCB refit on the logs (cross-fitted so it is not scored
on its own training rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from focus.bandit import ACTIONS, D, LinUCB, doubly_robust, ips, snips  # noqa: E402


def load(d: Path) -> list[dict]:
    frames = [pd.read_csv(p) for p in sorted(d.rglob("decisions_*.csv"))]
    if not frames:
        raise SystemExit(f"no decisions_*.csv under {d}")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["why"] == "policy") & df["action"].isin(ACTIONS)]
    logs = []
    for _, r in df.iterrows():
        x = np.array([float(v) for v in str(r["x"]).split()])
        if len(x) != D or not (0 < float(r["p"]) <= 1):
            continue
        logs.append({"x": x, "a": r["action"], "p": float(r["p"]), "r": float(r["reward"]),
                     "participant": r.get("participant", "anon")})
    return logs


def rule(x):
    e = x[1] * 60
    return {"sound" if e >= 90 else "toast" if e >= 9 else "wait": 1.0}


def never(x):
    return {"wait": 1.0}


def with_ci(est, logs, pol, n=300, seed=0):
    """Point estimate + 95 % CI, resampling whole participants."""
    by = {}
    for l in logs:
        by.setdefault(l["participant"], []).append(l)
    keys = list(by)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        sample = [l for k in rng.choice(keys, len(keys)) for l in by[k]]
        boots.append(est(sample, pol))
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return {"value": float(est(logs, pol)), "ci95": [float(lo), float(hi)]}


def fit(logs) -> LinUCB:
    m = LinUCB(epsilon=0.1)
    for l in logs:
        m.update(l["x"], l["a"], l["r"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/ope.json"))
    ap.add_argument("--export-prior", type=Path, default=None)
    a = ap.parse_args()
    logs = load(a.dir)
    people = sorted({l["participant"] for l in logs})
    # cross-fit the learned policy by participant halves
    half = set(people[::2])
    A = [l for l in logs if l["participant"] in half]
    B = [l for l in logs if l["participant"] not in half]
    rep = {"decisions": len(logs), "participants": len(people), "logged_mean_reward": float(np.mean([l["r"] for l in logs])),
           "policies": {}}
    for name, pol_a, pol_b in [("never", never, never), ("rule_v21", rule, rule)]:
        rep["policies"][name] = {k: with_ci(f, logs, pol_a) for k, f in (("ips", ips), ("snips", snips), ("dr", doubly_robust))}
    if A and B:
        ma, mb = fit(A), fit(B)
        greedy = lambda m: (lambda x: m.probs(x, explore=False))
        rep["policies"]["linucb_refit"] = {k: float(np.mean([f(B, greedy(ma)), f(A, greedy(mb))]))
                                           for k, f in (("ips", ips), ("snips", snips), ("dr", doubly_robust))}
    if a.export_prior:
        d = json.loads(fit(logs).to_json())
        d["source"] = f"fit on {len(logs)} real decisions from {len(people)} participants"
        a.export_prior.write_text(json.dumps(d))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
