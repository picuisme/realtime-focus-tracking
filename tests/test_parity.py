"""web/core.js must compute exactly what focus/ computes (to 1e-6)."""
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from focus import features as F
from focus import temporal as T
from focus.bandit import ACTIONS, LinUCB, context
from focus.simulator import NudgeBudget

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def rand_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R
    return [float(M[i][j]) for j in range(4) for i in range(4)]   # column-major


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    rng = np.random.default_rng(0)
    lmj = json.loads((ROOT / "tests/fixtures/landmarks_820x1024.json").read_text())
    buf = []
    for i in range(240):
        face = rng.random() > 0.05
        buf.append({"ear_score": float(rng.uniform(0.1, 0.35)) if face else 0.0,
                    "mar_score": float(abs(rng.normal(0.1, 0.2))), "head_pitch": float(rng.normal(0, 10)),
                    "head_yaw": float(rng.normal(0, 15)), "gaze_ratio": float(rng.normal(0.5, 0.05))})
    model = {"coef": rng.normal(size=13).tolist(), "intercept": 0.3, "mean": rng.normal(size=13).tolist(),
             "scale": rng.uniform(0.5, 2, 13).tolist()}
    updates = [[context(*[float(rng.uniform(0, 120)), float(rng.random()), float(rng.uniform(0, 900)),
                          int(rng.integers(0, 5)), float(rng.uniform(0, 3600)), float(rng.uniform(0, 24)),
                          ["phone"] if rng.random() < .5 else ["eyes_closed"]]).tolist(),
                ACTIONS[int(rng.integers(0, 5))], float(rng.normal())] for _ in range(40)]
    c = {"landmarks": lmj, "matrices": [rand_rotation(rng) for _ in range(20)], "buffer": buf, "model": model,
         "context": [42.0, 0.3, 120.0, 2, 1500.0, 21.5, ["head_turned", "phone"]], "updates": updates,
         "probe": updates[3][0],
         "budget_times": [[t, s] for t, s in zip(np.cumsum(rng.uniform(5, 400, 60)).tolist(), (rng.random(60) < .7).tolist())]}
    p = tmp_path_factory.mktemp("parity") / "cases.json"
    p.write_text(json.dumps(c))
    js = json.loads(subprocess.check_output(["node", str(ROOT / "tests/parity.mjs"), str(p)], text=True))
    return c, js


def close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0)


def test_frame_features(cases):
    c, js = cases
    lm = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in c["landmarks"]["lm"]]
    l43 = F.to_4x3(lm, c["landmarks"]["w"], c["landmarks"]["h"])
    assert close([F.ear(l43), F.mar(l43), F.brow_dist(l43), F.gaze_ratio(l43)], js["frame"])


def test_pose(cases):
    c, js = cases
    assert close([F.pose_from_matrix(m) for m in c["matrices"]], js["pose"])


def test_temporal(cases):
    c, js = cases
    ref = T.reference(c["buffer"])
    assert close([ref["ear"], ref["pitch"], ref["yaw"]], [js["reference"][k] for k in ("ear", "pitch", "yaw")])
    f = T.window_features(c["buffer"], ref)
    assert close(f, js["window"])
    m = c["model"]
    z = m["intercept"] + sum(w * (v - mu) / s for w, v, mu, s in zip(m["coef"], f, m["mean"], m["scale"]))
    assert math.isclose(1 / (1 + math.exp(-z)), js["drowsy"], abs_tol=1e-9)


def test_context(cases):
    c, js = cases
    assert close(context(*c["context"]), js["context"])


def test_bandit(cases):
    c, js = cases
    b = LinUCB(alpha=0.5, epsilon=0.1)
    for x, a, r in c["updates"]:
        b.update(np.asarray(x), a, r)
    x = np.asarray(c["probe"])
    s = b.scores(x)
    assert close([s[a] for a in ACTIONS], [js["scores"][a] for a in ACTIONS])
    p = b.probs(x)
    assert close([p[a] for a in ACTIONS], [js["probs"][a] for a in ACTIONS])
    pers = b.personalised(ridge=5, alpha=0.2)
    for xx, a, r in c["updates"][:5]:
        pers.update(np.asarray(xx), a, r)
    ps = pers.scores(x)
    assert close([ps[a] for a in ACTIONS], [js["personal_scores"][a] for a in ACTIONS])


def test_budget(cases):
    c, js = cases
    b, out = NudgeBudget(), []
    for t, spend in c["budget_times"]:
        ok = b.available(t)
        if ok and spend:
            b.spend(t)
        out.append(ok)
    assert out == js["budget"]
