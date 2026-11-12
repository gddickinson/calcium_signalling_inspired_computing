// Spatiotemporal computing engine: a line-by-line port of SpatiotemporalComputing
// (spatiotemporal_computing/spatiotemporal_computing.py): three coupled fields, each blurred by its own Gaussian
// kernel (periodic edges), mixed by a fixed interaction matrix, driven inside random circular "regions", then
// squashed with tanh. Random numbers come from np_random.js in numpy's order (st_engine.test.mjs, fixture st.json).
import { convolve, normalise, corr } from './grid.js';

export const ST_DEFAULTS = {
  gridSize: 100, diffusionRates: [0.6, 0.2, 0.05],
  // reaction[j][i]: effect of channel j on channel i (rows of reaction_params in the Python)
  reaction: [[0.0, 0.2, -0.1], [0.3, -0.1, 0.2], [-0.2, 0.1, 0.0]],
};

export class Spatiotemporal {
  constructor(rng, opts = {}) {
    const o = { ...ST_DEFAULTS, ...opts };
    this.rng = rng; this.n = o.gridSize; this.C = 3;
    this.rates = o.diffusionRates.slice(); this.reaction = o.reaction.map(r => r.slice());
    const N = this.n * this.n;
    this.state = [0, 1, 2].map(() => new Float64Array(N));
    this.kernels = this.rates.map(r => this._kernel(r));
    this.regions = this._regions();
    this.history = [];
    this.steps = 0;
    this._initPatterns();
  }

  _kernel(rate) {
    const size = 5, c = 2, k = new Float64Array(25);
    for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) {
      const dist = Math.sqrt((i - c) ** 2 + (j - c) ** 2);
      k[i * size + j] = Math.exp(-(dist ** 2) / rate);
    }
    return normalise(k);
  }

  // Five random circles; each cell inside gets a random region type 1-3 (later circles overwrite earlier ones)
  _regions() {
    const n = this.n, g = new Float64Array(n * n), r = this.rng;
    for (let k = 0; k < 5; k++) {
      const cx = r.randint(0, n), cy = r.randint(0, n), rad = r.randint(5, 20);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (Math.sqrt((i - cx) ** 2 + (j - cy) ** 2) < rad) g[i * n + j] = r.randint(1, 4);
      }
    }
    return g;
  }

  // N(0, 0.01) noise everywhere, a 10x10 square in channel 0 (rows/cols 20-29), a bar in channel 1
  // (rows 40-44, cols 10-89) and a top-to-bottom gradient added to channel 2 (fixed cell positions, as in the Python)
  _initPatterns() {
    const n = this.n, s = this.state;
    for (let ch = 0; ch < 3; ch++) for (let i = 0; i < n * n; i++) s[ch][i] += this.rng.normal(0, 0.01);
    for (let i = 20; i < Math.min(30, n); i++) for (let j = 20; j < Math.min(30, n); j++) s[0][i * n + j] = 1.0;
    for (let i = 40; i < Math.min(45, n); i++) for (let j = 10; j < Math.min(90, n); j++) s[1][i * n + j] = 0.8;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s[2][i * n + j] += i / n;
  }

  update() {
    const n = this.n, N = n * n, s = this.state;
    this.history.push(s.map(a => a.slice()));
    if (this.history.length > 100) this.history.shift();
    const next = s.map((a, i) => convolve(a, n, n, this.kernels[i], 5, 'wrap'));
    for (let i = 0; i < 3; i++) {
      const term = new Float64Array(N);
      for (let j = 0; j < 3; j++) { const w = this.reaction[j][i], sj = s[j]; for (let p = 0; p < N; p++) term[p] += w * sj[p]; }
      const ni = next[i];
      for (let p = 0; p < N; p++) { ni[p] += term[p]; ni[p] += this.regions[p] === i + 1 ? 0.1 : 0.0; }
    }
    for (const a of next) for (let p = 0; p < N; p++) a[p] = Math.tanh(a[p]);
    this.state = next; this.steps++;
  }

  run(steps) { for (let k = 0; k < steps; k++) this.update(); }

  // Channel-to-channel correlation of the most recent history frame (the Python analysis uses history[-1])
  correlations(frame = this.history.at(-1) || this.state) {
    return [0, 1, 2].map(i => [0, 1, 2].map(j => (i === j ? 1 : corr(frame[i], frame[j]))));
  }

  // Mean and range of each channel of the current state
  stats() {
    return this.state.map(a => { let lo = Infinity, hi = -Infinity, m = 0; for (const v of a) { m += v; if (v < lo) lo = v; if (v > hi) hi = v; } return { mean: m / a.length, min: lo, max: hi }; });
  }
}
