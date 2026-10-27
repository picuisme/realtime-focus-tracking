import numpy as np
import pandas as pd

from focus.early_warning import D, OnlineTD, Scaler, gamma_for, lstd_lambda, state_features


def _chain(n=20000, seed=0):
    """Two-state Markov chain with one-hot features; cumulant 1 in state 1."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.95, 0.05], [0.2, 0.8]])
    s = [0]
    for _ in range(n - 1):
        s.append(rng.choice(2, p=P[s[-1]]))
    s = np.array(s)
    Phi = np.eye(2)[s]
    return P, s, Phi


def test_lstd_matches_true_value():
    P, s, Phi = _chain()
    g = 0.9
    c = (s == 1).astype(float)
    v_true = np.linalg.solve(np.eye(2) - g * P, np.array([0.0, 1.0]))
    # cumulant c_t is received on leaving s_t, so V(s) = c(s) + g * E[V(s')]
    w = lstd_lambda([(Phi, c)], g, lam=0.0, ridge=1e-6)
    assert np.allclose(w, v_true, rtol=0.1)
    assert w[1] > w[0]


def test_online_td_moves_toward_lstd():
    P, s, Phi = _chain(5000, seed=1)
    g = 0.9
    c = (s == 1).astype(float)
    w_star = lstd_lambda([(Phi, c)], g, lam=0.0, ridge=1e-6)
    td = OnlineTD(w=np.zeros(2), scaler=Scaler(np.zeros(2), np.ones(2)), gamma=g, lam=0.0, alpha=0.05)
    for t in range(len(s)):
        td.step(Phi[t], c[t - 1] if t else 0.0)
    assert np.abs(td.w - w_star).max() < 1.0


def test_state_features_shape_and_causal():
    t = np.arange(0, 120, 0.1)
    df = pd.DataFrame({"t": t, "person": 1, "ear": 0.3, "mar": 0.1, "gaze": 0.5, "pitch": 0.0, "yaw": 0.0})
    df.loc[df.t > 100, "ear"] = 0.05
    X = state_features(df)
    assert X.shape == (len(df), D)
    # changing the future must not change the past
    df2 = df.copy(); df2.loc[df2.t > 110, "yaw"] = 60
    X2 = state_features(df2)
    k = int(np.searchsorted(t, 110))
    assert np.allclose(X[:k], X2[:k])


def test_gamma_for():
    assert abs(gamma_for(2.0, 0.1) - np.exp(-0.05)) < 1e-12
