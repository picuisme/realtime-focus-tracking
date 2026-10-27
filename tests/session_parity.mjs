// Drives core.js episodeStep + resolvePending through a scripted sequence; prints JSON.
import fs from 'fs';
import * as C from '../web/core.js';
const c = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const st = { distractedS: 0, focusedS: 0, episodeStart: null, backSince: null, lastReturnT: null, episodes: [] };
const pending = [], resolved = [], closes = [];
let t = 0, di = 0;
for (const [away, notHelpfulAt] of c.frames) {
  t = Math.round((t + c.dt) * 1e9) / 1e9;
  const closed = C.episodeStep(st, !!away, c.dt, t);
  if (closed) closes.push([t, st.lastReturnT]);
  while (di < c.decisions.length && c.decisions[di][0] <= t + 1e-9) {
    pending.push({ t: c.decisions[di][0], action: c.decisions[di][1], notHelpful: false, resolved: false, returnT: null }); di++;
  }
  if (notHelpfulAt != null && pending[notHelpfulAt]) pending[notHelpfulAt].notHelpful = true;
  resolved.push(...C.resolvePending(pending, t, closed, st.lastReturnT).map(d => [d.t, d.action, d.refocused, d.reward, d.returnT]));
}
console.log(JSON.stringify({ closes, episodes: st.episodes.map(e => [e.start, e.dur]), resolved }));
