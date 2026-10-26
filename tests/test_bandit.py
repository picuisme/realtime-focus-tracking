import numpy as np

from focus.bandit import ACTIONS, LinUCB, context, doubly_robust, ips, snips
from focus.simulator import NudgeBudget


def test_linucb_learns_the_better_action():
    rng = np.random.default_rng(0)
    b = LinUCB(seed=0)
    for _ in range(600):
        x = context(rng.uniform(0, 60), 0.1, 300, 0, 600, 20, ["phone"])
        a, _ = b.act(x)
        b.update(x, a, float(rng.random() < (0.8 if a == "sound" else 0.2)))
    x = context(30, 0.1, 300, 0, 600, 20, ["phone"])
    assert max(b.scores(x, explore=False), key=b.scores(x, explore=False).get) == "sound"


def test_probs_sum_to_one_and_explore():
    p = LinUCB().probs(context(10, 0, 0, 0, 0, 12, []))
    assert abs(sum(p.values()) - 1) < 1e-12 and min(p.values()) >= 0.1 / len(ACTIONS) - 1e-12


def test_personalised_starts_at_the_population_estimate():
    pop = LinUCB(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = context(rng.uniform(0, 60), rng.random(), 300, 0, 600, 20, [])
        pop.update(x, ACTIONS[rng.integers(5)], rng.normal())
    me = pop.personalised()
    x = context(20, 0.2, 100, 1, 900, 9, ["phone"])
    for a in ACTIONS:
        assert np.allclose(me.theta(a), pop.theta(a))


def test_ope_estimators_recover_a_known_value():
    rng = np.random.default_rng(3)
    true_mean = {a: m for a, m in zip(ACTIONS, [0.1, 0.5, 0.3, 0.2, 0.4])}
    logs = []
    for _ in range(20000):
        x = context(rng.uniform(0, 60), 0, 0, 0, 0, 12, [])
        a = ACTIONS[rng.integers(5)]
        logs.append({"x": x, "a": a, "p": 0.2, "r": float(rng.random() < true_mean[a])})
    target = lambda x: {"toast": 1.0}
    for est in (ips, snips, doubly_robust):
        assert abs(est(logs, target) - 0.5) < 0.03


def test_budget_caps_long_run_rate():
    b, n = NudgeBudget(per_hour=6), 0
    for t in range(0, 10 * 3600, 5):
        if b.available(t):
            b.spend(t)
            n += 1
    assert n <= 6 * 10 + 2
