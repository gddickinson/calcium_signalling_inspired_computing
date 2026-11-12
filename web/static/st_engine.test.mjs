// Node test: st_engine.js reproduces spatiotemporal_computing.py (tests/fixtures/st.json).
// Run: node web/static/st_engine.test.mjs
import { NumpyRandom } from './np_random.js';
import { Spatiotemporal } from './st_engine.js';
import { fixture, near } from './util.test.mjs';

const row = (a, n, r) => Array.from(a.slice(n * r, n * (r + 1)));
for (const f of fixture('st').runs) {
  const m = new Spatiotemporal(new NumpyRandom(f.seed), { gridSize: f.n });
  near(m.regions.reduce((a, b) => a + b, 0), f.regions_sum, 'regions sum');
  near(row(m.regions, f.n, Math.floor(f.n / 2)), f.regions_row, 'regions row');
  near(row(m.state[2], f.n, Math.floor(f.n / 3)), f.s0_row, 'initial channel 2');
  m.run(f.steps);
  [0, 1, 2].forEach(c => near(row(m.state[c], f.n, Math.floor((c + 1) * f.n / 4)), f.rows[c], `channel ${c} row`));
  near(m.state.map(a => a.reduce((x, y) => x + y, 0)), f.sums, 'sums');
  near(m.correlations(), f.corr, 'correlations', 1e-8);
}
console.log('st_engine: ok');
