// Reads a JSON case file, runs web/core.js on it, prints JSON results.
import fs from 'fs';
import * as C from '../web/core.js';

const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const lm = cases.landmarks.lm.map(([x, y, z]) => ({ x, y, z }));
const l43 = C.to43(lm, cases.landmarks.w, cases.landmarks.h);
const out = {
  frame: [C.ear(l43), C.mar(l43), C.browDist(l43), C.gazeRatio(l43)],
  pose: cases.matrices.map(m => { const p = C.poseFromMatrix(m); return [p.pitch, p.yaw, p.roll]; }),
  reference: C.reference(cases.buffer),
  window: C.windowFeatures(cases.buffer, C.reference(cases.buffer)),
  drowsy: C.drowsiness(cases.model, C.windowFeatures(cases.buffer, C.reference(cases.buffer))),
  context: C.context(...cases.context),
};
const bandit = new C.LinUCB({ alpha: 0.5, epsilon: 0.1 });
for (const [x, a, r] of cases.updates) bandit.update(x, a, r);
out.scores = bandit.scores(cases.probe);
out.probs = bandit.probs(cases.probe);
const personal = new C.LinUCB({ alpha: 0.2, epsilon: 0.1, ridge: 5, offset: Object.fromEntries(C.ACTIONS.map(a => [a, bandit.theta(a)])) });
for (const [x, a, r] of cases.updates.slice(0, 5)) personal.update(x, a, r);
out.personal_scores = personal.scores(cases.probe);
const bud = new C.NudgeBudget();
out.budget = cases.budget_times.map(([t, spend]) => { const ok = bud.available(t); if (ok && spend) bud.spend(t); return ok; });
console.log(JSON.stringify(out));
