// Node test: puff_engine.js reproduces probabilistic_computing.py (tests/fixtures/puff.json).
// Run: node web/static/puff_engine.test.mjs
import { NumpyRandom } from './np_random.js';
import { CalciumPuff, Stochastic } from './puff_engine.js';
import { fixture, near } from './util.test.mjs';

for (const f of fixture('puff').runs) {
  const rng = new NumpyRandom(f.seed);
  const m = new CalciumPuff(rng, { gridSize: f.n, decayRate: f.decay, threshold: f.threshold });
  const row = r => Array.from(r);
  near(row(m.density.slice(f.n * Math.floor(f.n / 3), f.n * (Math.floor(f.n / 3) + 1))), f.density_row, 'density row');
  const res = m.run(f.steps);
  near(res, f.results, 'results');
  const mid = Math.floor(f.n / 2);
  near(row(m.calcium.slice(f.n * mid, f.n * (mid + 1))), f.calcium_row, 'calcium row');
  near(m.calcium.reduce((a, b) => a + b, 0), f.calcium_sum, 'calcium sum');
  near(m.history.at(-1).reduce((a, b) => a + b, 0), f.last_activations, 'activations');
  const sc = new Stochastic(rng);
  near({ multiply: sc.multiply(0.3, 0.7, 500).value, add: sc.add(0.3, 0.4, 500).value,
    subtract: sc.subtract(0.7, 0.3, 500).value, divide: sc.divide(0.3, 0.6, 500).value }, f.ops, 'ops');
}
console.log('puff_engine: ok');
