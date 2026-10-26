"""Compare nudge policies on simulated learners and check the off-policy estimators.

All results are SIMULATED (see focus/simulator.py). They show the bandit and
the estimators behave as intended; they say nothing yet about real people.

    python scripts/simulate_bandit.py --users 200 --sessions 4 --out reports/simulation.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from focus.bandit import COST, LinUCB, doubly_robust, ips, snips  # noqa: E402
from focus.simulator import SimUser, expected_reward, never, rule_v21, rule_v21_probs, run_session  # noqa: E402


def summarise(results):
    mins = sum(r["minutes"] for r in results)
    eps = sum(r["episodes"] for r in results)
    rows = [l for r in results for l in r["log"]]
    nudges = [l for l in rows if l["nudge"]]
    ret = [r["mean_return_s"] for r in results if r["episodes"]]
    return {
        "reward_per_decision": float(np.mean([l["r"] for l in rows])),
        "mean_return_s": float(np.average(ret, weights=[r["episodes"] for r in results if r["episodes"]])),
        "nudges_per_hour": sum(r["nudges"] for r in results) / mins * 60,
        "useful_nudge_rate": float(np.mean([l["refocus"] for l in nudges])) if nudges else None,
        "episodes": eps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--user-ridge", type=float, default=5.0)
    ap.add_argument("--user-alpha", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=Path("reports/simulation.json"))
    ap.add_argument("--export-prior", type=Path, default=None,
                    help="write the population model for the web app (its starting point before real data)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    users = [SimUser.sample(rng) for _ in range(a.users)]
    train, test = users[: a.users // 2], users[a.users // 2:]

    # population prior: one LinUCB learning online over the training users
    prior = LinUCB(seed=a.seed)
    for u in train:
        for s in range(a.sessions):
            run_session(u, lambda x, i: prior.act(x), np.random.default_rng(hash((id(u), s)) % 2**32), learn=prior.update)

    def evaluate(make_choose, personal=False):
        out, logs = [], []
        for k, u in enumerate(test):
            model = prior.personalised(ridge=a.user_ridge, alpha=a.user_alpha, seed=k) if personal else None
            for s in range(a.sessions):
                srng = np.random.default_rng(10_000 + 100 * k + s)   # common random numbers across policies
                if personal:
                    res = run_session(u, lambda x, i: model.act(x), srng, learn=model.update)
                else:
                    res = run_session(u, make_choose, srng)
                out.append(res)
                logs += [dict(l, user=u) for l in res["log"]]
        return out, logs

    report = {"note": "SIMULATED learners; see focus/simulator.py for every assumption.",
              "users_test": len(test), "sessions_per_user": a.sessions, "policies": {}}
    for name, choose, personal in [("never", never, False), ("rule_v21", rule_v21, False),
                                   ("linucb_population", lambda x, i: prior.act(x), False),
                                   ("linucb_personalised", None, True)]:
        res, logs = evaluate(choose, personal)
        report["policies"][name] = summarise(res)
        if name == "linucb_personalised":
            bandit_logs = logs

    # Off-policy check: estimate the rule's value from the bandit's logs, compare with the truth
    truth = float(np.mean([expected_reward(l["user"], next(iter(rule_v21_probs(l["x"]))), l["state"]) for l in bandit_logs]))
    clean = [{k: l[k] for k in ("x", "a", "p", "r")} for l in bandit_logs]
    report["ope_check"] = {"target": "rule_v21", "logged_decisions": len(clean), "true_value": truth,
                           "ips": ips(clean, rule_v21_probs), "snips": snips(clean, rule_v21_probs),
                           "doubly_robust": doubly_robust(clean, rule_v21_probs)}
    if a.export_prior:
        a.export_prior.parent.mkdir(parents=True, exist_ok=True)
        d = json.loads(prior.to_json())
        d["source"] = "trained on simulated learners (focus/simulator.py); replace with a model fit on real logs"
        a.export_prior.write_text(json.dumps(d))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
