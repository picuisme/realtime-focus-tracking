# Label-free TD yawn detector in the app: not shipped (3 Nov 2026)

Question: should the web app replace its yawn rule with the TD value detector,
learned without labels from the rule's own signal, as suggested by the DMD results
in `reports/early_warning_dmd.json`?

Setting that matches the app: features at 3 Hz, TD trained on UTA-RLDD (60 people,
no labels), threshold set on UTA-RLDD so that TD raises as many alerts per hour as
the rule; tested once on DMD drowsiness recordings (13 people, 76 annotated yawns
of 2 s or more), down-sampled to 3 Hz. Script: `scripts/train_td_yawn.py`,
numbers: `reports/td_yawn.json`.

Best recall at a given number of false alarms per hour on DMD:

| False alarms per hour | Rule (mouth open, best duration) | TD, fixed weights | TD, keeps learning per recording |
| --- | --- | --- | --- |
| 0 | 0.66 | 0.42 | 0.34 |
| 5 | 0.66 | 0.63 | 0.46 |
| 20 | 0.66 | 0.68 | 0.72 |

At the app's own sample rate and across datasets, TD does not beat the rule, and
online learning makes it worse at low false-alarm rates: the acted DMD sessions
contain many yawns, so per-recording learning inflates the value and the fixed
threshold no longer fits. The earlier in-domain DMD result (10 Hz, cross-validated
on DMD itself) does not carry over.

Where the misses come from (the app's per-frame rule, DMD at ~10 Hz):

| Rule: mouth open for at least | Recall | Uncovered yawns | Yawns behind a hand | Median delay | False alarms per hour |
| --- | --- | --- | --- | --- | --- |
| 0.5 s | 0.49 | 34 / 39 | 3 / 37 | 1.3 s | 0 |
| 0.8 s | 0.49 | 34 / 39 | 3 / 37 | 1.6 s | 0 |
| 1.2 s (current) | 0.45 | 32 / 39 | 2 / 37 | 2.0 s | 0 |

The rule already finds most uncovered yawns. Almost half of DMD yawns are made
behind a hand, which no mouth-opening signal can see, rule or TD. Closing that gap
needs a different input (for example, a hand held over the mouth), which cannot be
validated now because the DMD videos are no longer available.
