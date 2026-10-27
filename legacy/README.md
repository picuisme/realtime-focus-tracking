# v2.1 (legacy)

The DQN version of Focus Tracker, kept for comparison in the report. Not used by the v3 apps.

Why it was replaced (details in the PRD):
- The environment replays logged data, so actions never change the next state: it is a contextual bandit, not RL.
- `is_distracted_label` and `consecutive_frames` are both inputs and the basis of the reward, so the agent re-learns the hand-written reward table (89 % agreement on its own training data).
- At run time the overrides in `FocusAgentInference` discard most of the agent's choices.
- `calculate_gaze_ratio` paired each iris with the wrong eye (≈2.4 instead of ≈0.5).
- Thresholds were counted in frames while the app ran at 10–15 fps and the training data assumed 3 fps.

Files: `focus_env.py` (environment + inference wrapper), `demo_webcam.py` (v2.1 feature functions, gaze fixed), `train_model.py`, `evaluate_model.py`, `dqn_focus_agent_v2.1.zip`, `scaler_v2.1.pkl`.
