// Probabilistic computing engine: a line-by-line port of CalciumPuffComputing and StochasticComputing
// (probabilistic_computing/probabilistic_computing.py). Random numbers come from np_random.js in the same order as
// numpy, so a seed replays the Python run (puff_engine.test.mjs checks against tests/fixtures/puff.json).
import { convolve, normalise, corr, linfit } from './grid.js';

export const PUFF_DEFAULTS = { gridSize: 50, decayRate: 0.95, threshold: 0.7, kernelSize: 3 };

export class CalciumPuff {
  constructor(rng, opts = {}) {
    const o = { ...PUFF_DEFAULTS, ...opts };
    this.rng = rng; this.n = o.gridSize; this.decayRate = o.decayRate; this.threshold = o.threshold;
    const n = this.n;
    this.calcium = new Float64Array(n * n);
    this.density = new Float64Array(n * n);
    this._initDensity();
    this.ks = o.kernelSize; this.kernel = this._kernel(o.kernelSize);
    this.history = [];          // last 50 activation grids (Uint8Array); Python keeps all, only 50 are ever read
    this.steps = 0;
    this.densityVsActivation = new Map();
  }

  // Four quadrants with densities 0.2, 0.4, 0.6, 0.8, plus N(0, 0.05) noise, clipped to [0, 1]
  _initDensity() {
    const n = this.n, rs = Math.floor(n / 2), d = this.density;
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) d[r * n + c] = r < rs ? (c < rs ? 0.2 : 0.4) : (c < rs ? 0.6 : 0.8);
    const noise = this.rng.normalN(0, 0.05, n * n);
    for (let i = 0; i < n * n; i++) d[i] = Math.min(1, Math.max(0, d[i] + noise[i]));
  }

  _kernel(size) {
    const k = new Float64Array(size * size), c = Math.floor(size / 2);
    for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) {
      const dist = Math.sqrt((i - c) ** 2 + (j - c) ** 2);
      k[i * size + j] = Math.exp(-(dist ** 2));
    }
    return normalise(k);
  }

  update() {
    const n = this.n, N = n * n, ca = this.calcium;
    for (let i = 0; i < N; i++) ca[i] *= this.decayRate;
    const act = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      const u = this.rng.random();
      act[i] = u < this.density[i] && ca[i] < this.threshold ? 1 : 0;
    }
    this.history.push(act); this.steps++;
    if (this.history.length > 50) this.history.shift();
    for (let i = 0; i < N; i++) if (act[i]) ca[i] += 1.0;
    this.calcium = convolve(ca, n, n, this.kernel, this.ks, 'wrap');
    if (this.steps > 10) this._updateStatistics();
  }

  // Activation probability per 0.1-wide density bin over the last 50 steps, smoothed by a 0.95 running average
  _updateStatistics() {
    const N = this.n * this.n, hl = Math.min(50, this.steps), recent = this.history.slice(-hl);
    for (let i = 0; i < 10; i++) {
      const lower = i * 0.1, upper = (i + 1) * 0.1, key = `${lower.toFixed(1)}-${upper.toFixed(1)}`;
      let cells = 0, hits = 0;
      for (let j = 0; j < N; j++) {
        if (this.density[j] >= lower && this.density[j] < upper) {
          cells++;
          for (const h of recent) hits += h[j];
        }
      }
      if (cells > 0) {
        const prob = hits / (cells * hl);
        const old = this.densityVsActivation.get(key);
        this.densityVsActivation.set(key, old === undefined ? prob : 0.95 * old + 0.05 * prob);
      }
    }
  }

  run(steps) { for (let s = 0; s < steps; s++) this.update(); return this.analyze(); }

  analyze() {
    const keys = [...this.densityVsActivation.keys()].sort();
    const densities = keys.map(k => { const [a, b] = k.split('-').map(Number); return (a + b) / 2; });
    const probabilities = keys.map(k => this.densityVsActivation.get(k));
    if (densities.length < 2) return { densities, probabilities };
    const [slope, intercept] = linfit(densities, probabilities);
    return { densities, probabilities, correlation: corr(densities, probabilities), slope, intercept };
  }

  // Mean of the last 10 activation grids (the "recent activation probability" panel)
  recentActivation() {
    const N = this.n * this.n, out = new Float64Array(N), h = this.history.slice(-10);
    for (const a of h) for (let i = 0; i < N; i++) out[i] += a[i];
    if (h.length) for (let i = 0; i < N; i++) out[i] /= h.length;
    return out;
  }
}

// Stochastic arithmetic on bitstreams. Each op returns { value, streams } (streams: the bitstreams used, for drawing).
export class Stochastic {
  constructor(rng) { this.rng = rng; }
  _bits(p, n) { const b = new Uint8Array(n); for (let i = 0; i < n; i++) b[i] = this.rng.random() < p ? 1 : 0; return b; }
  _check(a, b) { if (!(a >= 0 && a <= 1 && b >= 0 && b <= 1)) throw new Error('Inputs must be probabilities between 0 and 1'); }
  _mean(x) { let s = 0; for (const v of x) s += v; return s / x.length; }

  multiply(a, b, n = 1000) {
    this._check(a, b);
    const A = this._bits(a, n), B = this._bits(b, n), R = A.map((v, i) => v & B[i]);
    return { value: this._mean(R), streams: { a: A, b: B, out: R } };
  }

  add(a, b, n = 1000) {
    this._check(a, b);
    const A = this._bits(a, n), B = this._bits(b, n), S = this._bits(0.5, n), R = A.map((v, i) => (S[i] ? v : B[i]));
    return { value: Math.min(1.0, 2.0 * this._mean(R)), streams: { a: A, b: B, sel: S, out: R } };
  }

  // Correlated streams (one shared random number per bit): a AND NOT b has probability max(0, a - b)
  subtract(a, b, n = 1000) {
    this._check(a, b);
    const u = Float64Array.from({ length: n }, () => this.rng.random());
    const A = u.map(x => (x < a ? 1 : 0)), B = u.map(x => (x < b ? 1 : 0)), R = A.map((v, i) => v & (1 - B[i]));
    return { value: Math.max(0.0, this._mean(R)), streams: { a: Uint8Array.from(A), b: Uint8Array.from(B), out: Uint8Array.from(R) } };
  }

  // Correlated streams: P(a-bit | b-bit) = min(1, a / b)
  divide(a, b, n = 1000, epsilon = 0.01) {
    this._check(a, b);
    if (b < epsilon) return { value: 1.0, streams: null };
    const u = Float64Array.from({ length: n }, () => this.rng.random());
    const A = Uint8Array.from(u, x => (x < a ? 1 : 0)), B = Uint8Array.from(u, x => (x < b ? 1 : 0));
    let nb = 0, nab = 0;
    for (let i = 0; i < n; i++) if (B[i]) { nb++; nab += A[i]; }
    const R = A.map((v, i) => v & B[i]);
    return { value: nb === 0 ? 0.0 : Math.min(1.0, nab / nb), streams: { a: A, b: B, out: R } };
  }
}

export const EXACT = {
  multiply: (a, b) => a * b, add: (a, b) => Math.min(1, a + b),
  subtract: (a, b) => Math.max(0, a - b), divide: (a, b) => (b < 0.01 ? 1 : Math.min(1, a / b)),
};
