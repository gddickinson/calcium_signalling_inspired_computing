// Wave medium engine: a line-by-line port of WaveComputingMedium (wave_based_computing/wave_medium.py) and of
// WaveComputing's logic-gate, memory and filter set-ups (wave_computing.py). Deterministic (no random numbers);
// wave_engine.test.mjs checks it against tests/fixtures/wave.json.
//
// As in the Python (see the module note there), the update writes `previous = current` before using it and uses the
// neighbour mean as "laplacian", so each step is  u <- damping * (u + D*dt^2 * mean8(u)) [edge taper]
// - nonlinearity * u^3, then resonance gain, sources and memory elements.
import { convolve, normalise } from './grid.js';

const KERNEL = normalise(Float64Array.from([0.05, 0.2, 0.05, 0.2, 0.0, 0.2, 0.05, 0.2, 0.05]));
// numpy slice bounds: negative start/stop count from the end, then clamp to [0, n]
const sl = (v, n) => Math.min(n, Math.max(0, v < 0 ? v + n : v));

export const WAVE_DEFAULTS = { width: 100, height: 100, boundary: 'absorbing', damping: 0.99, diffusion: 0.2, nonlinearity: 0.1, timeStep: 0.1 };

export class WaveMedium {
  constructor(opts = {}) {
    const o = { ...WAVE_DEFAULTS, ...opts };
    Object.assign(this, o);
    const N = o.width * o.height;
    this.current = new Float64Array(N); this.previous = new Float64Array(N);
    this.frequency = new Float64Array(N); this.phase = new Float64Array(N);
    this.obstacles = new Uint8Array(N);
    this.sources = []; this.resonant = []; this.memory = new Map();
    this.history = []; this.energy = []; this.steps = 0;
    this._mask = this._boundaryMask();
  }

  addSource([x, y], amplitude, frequency, phase = 0, duration = null) {
    this.sources.push({ x, y, amplitude, frequency, phase, start: 0, duration });
  }
  addObstacle(x0, x1, y0, y1) {
    const { width: W, height: H } = this;
    for (let y = sl(y0, H); y < sl(y1, H); y++) for (let x = sl(x0, W); x < sl(x1, W); x++) this.obstacles[y * W + x] = 1;
  }
  // Python keeps these in a dict keyed by the box, so adding the same box again only updates its frequency
  addResonantRegion(x0, x1, y0, y1, frequency) {
    const r = this.resonant.find(q => q.x0 === x0 && q.x1 === x1 && q.y0 === y0 && q.y1 === y1);
    if (r) r.frequency = frequency; else this.resonant.push({ x0, x1, y0, y1, frequency });
  }
  setMemory([x, y], value) { if (x >= 0 && x < this.width && y >= 0 && y < this.height) this.memory.set(`${x},${y}`, [x, y, value, 0.999]); }
  readMemory([x, y]) { const m = this.memory.get(`${x},${y}`); return m ? m[2] : 0.0; }

  _boundaryMask() {
    const { width: W, height: H } = this, m = new Float64Array(W * H).fill(1), bw = 5;
    for (let i = 0; i < bw; i++) {
      const f = 1.0 - (i / bw) ** 2;
      for (const r of [i, H - i - 1]) if (r >= 0 && r < H) for (let x = 0; x < W; x++) m[r * W + x] *= f;
      for (const c of [i, W - i - 1]) if (c >= 0 && c < W) for (let y = 0; y < H; y++) m[y * W + c] *= f;
    }
    return m;
  }

  update(t) {
    const { width: W, height: H } = this, N = W * H, cur = this.current;
    this.previous = cur.slice();
    const lap = convolve(cur, H, W, KERNEL, 3, 'symm'), c = this.diffusion * this.timeStep ** 2;
    const next = new Float64Array(N);
    for (let i = 0; i < N; i++) next[i] = (2 * cur[i] - this.previous[i] + c * lap[i]) * this.damping;
    if (this.boundary === 'absorbing') for (let i = 0; i < N; i++) next[i] *= this._mask[i];
    else if (this.boundary === 'periodic') {     // as coded: adds half of the opposite edge, row/column in sequence
      for (let x = 0; x < W; x++) next[x] += 0.5 * next[(H - 1) * W + x];
      for (let x = 0; x < W; x++) next[(H - 1) * W + x] += 0.5 * next[x];
      for (let y = 0; y < H; y++) next[y * W] += 0.5 * next[y * W + W - 1];
      for (let y = 0; y < H; y++) next[y * W + W - 1] += 0.5 * next[y * W];
    }
    if (this.nonlinearity > 0) for (let i = 0; i < N; i++) next[i] -= this.nonlinearity * cur[i] ** 3;
    for (let i = 0; i < N; i++) if (this.obstacles[i]) next[i] = 0;
    for (const r of this.resonant) {
      if (this.history.length > 10) {
        const lf = this.localFrequency(r), g = 1.0 + 0.1 * Math.exp(-(((lf - r.frequency) / 0.1) ** 2));
        for (let y = sl(r.y0, H); y < sl(r.y1, H); y++) for (let x = sl(r.x0, W); x < sl(r.x1, W); x++) next[y * W + x] *= g;
      }
    }
    for (const s of this.sources) {
      if (s.duration === null || t < s.start + s.duration) {
        const v = s.amplitude * Math.sin(2 * Math.PI * s.frequency * t + s.phase);
        for (let dx = -1; dx < 2; dx++) for (let dy = -1; dy < 2; dy++) {
          const nx = s.x + dx, ny = s.y + dy;
          if (nx >= 0 && nx < W && ny >= 0 && ny < H) next[ny * W + nx] += v * Math.exp(-(dx ** 2 + dy ** 2) / 2);
        }
      }
    }
    for (const m of this.memory.values()) { const v = m[2]; m[2] = v * m[3]; next[m[1] * W + m[0]] += 0.1 * v; }
    this.current = next;
    if (this.history.length > 5) this._frequencyPhase();
    this.history.push(next.slice());
    if (this.history.length > 100) this.history.shift();
    let e = 0; for (const v of next) e += v * v;
    this.energy.push(e); if (this.energy.length > 100) this.energy.shift();
    this.steps++;
  }

  // Peak of the real FFT of the region's mean over the last 50 frames (DC excluded), in cycles per time unit
  localFrequency({ x0, x1, y0, y1 }) {
    const { width: W, height: H } = this, frames = this.history.slice(-50), n = frames.length;
    if (n < 10) return 0.0;
    const ya = sl(y0, H), yb = sl(y1, H), xa = sl(x0, W), xb = sl(x1, W), cnt = (yb - ya) * (xb - xa);
    const series = frames.map(f => { let s = 0; for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) s += f[y * W + x]; return s / cnt; });
    const nb = Math.floor(n / 2) + 1;
    if (nb <= 1) return 0.0;
    let best = 1, bestMag = -1;
    for (let k = 1; k < nb; k++) {
      let re = 0, im = 0;
      for (let j = 0; j < n; j++) { const a = -2 * Math.PI * k * j / n; re += series[j] * Math.cos(a); im += series[j] * Math.sin(a); }
      const mag = Math.hypot(re, im);
      if (mag > bestMag) { bestMag = mag; best = k; }
    }
    return best * (1.0 / (n * this.timeStep));
  }

  // Per cell over the last 10 frames: sign changes -> frequency, position within the min-max range -> phase
  _frequencyPhase() {
    if (this.history.length < 10) return;
    const rec = this.history.slice(-10), N = this.width * this.height;
    for (let i = 0; i < N; i++) {
      let cross = 0, lo = Infinity, hi = -Infinity, prev = null;
      for (const f of rec) {
        const v = f[i], neg = v < 0 || Object.is(v, -0);
        if (prev !== null && neg !== prev) cross++;
        prev = neg; if (v < lo) lo = v; if (v > hi) hi = v;
      }
      this.frequency[i] = cross / (10 * this.timeStep) / 2;
      this.phase[i] = hi > lo ? (this.current[i] - lo) / (hi - lo) * 2 * Math.PI : 0;
    }
  }

  // Python's run_simulation: t restarts at 0 on every call
  run(steps) { for (let s = 0; s < steps; s++) this.update(s * this.timeStep); }
}

const fromMedium = (m, extra = {}) => new WaveMedium({ width: m.width, height: m.height, boundary: m.boundary, damping: m.damping, diffusion: m.diffusion, nonlinearity: m.nonlinearity, ...extra });

// WaveComputing.wave_logic_gates set-up: two lines of sources (left edge, top edge), a resonant centre box.
// The input patterns are not used by the Python; the set-up depends only on the operation.
export function logicGateMedium(base, op) {
  const m = fromMedium(base), { width: W, height: H } = m, cx = Math.floor(W / 2), cy = Math.floor(H / 2);
  const [amp, f, ph2, half] = { and: [0.3, 0.5, 0, 10], or: [0.2, 0.7, 0, 15], xor: [0.3, 0.6, Math.PI, 0] }[op];
  for (let y = Math.floor(H / 4); y < Math.floor(3 * H / 4); y++) m.addSource([5, y], amp, f, 0);
  for (let x = Math.floor(W / 4); x < Math.floor(3 * W / 4); x++) m.addSource([x, 5], amp, f, ph2);
  if (half) m.addResonantRegion(cx - half, cx + half, cy - half, cy + half, f);
  else m.nonlinearity = 0.2;
  return m;
}

// Gate readout: the largest |amplitude| in the central 20 x 20 box over the last 10 frames
export function gateReadout(m) {
  const { width: W, height: H } = m, cx = Math.floor(W / 2), cy = Math.floor(H / 2);
  const x0 = Math.max(0, cx - 10), x1 = Math.min(W, cx + 10), y0 = Math.max(0, cy - 10), y1 = Math.min(H, cy + 10);
  let best = 0;
  for (const f of m.history.slice(-10)) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) best = Math.max(best, Math.abs(f[y * W + x]));
  return best;
}

// WaveComputing.wave_memory set-up; storage then retrieval (retrieval adds pulse sources along the bottom).
export function memoryMedium(base, data) {
  const m = fromMedium(base, { boundary: 'absorbing', damping: 0.995, diffusion: 0.1, nonlinearity: 0.05 });
  const spacing = Math.floor(m.width / (data.length + 1)), my = Math.floor(m.height / 2);
  data.forEach((v, i) => {
    const x = (i + 1) * spacing, f = 0.5 + 0.3 * (v + 1) / 2;
    m.addSource([x, 5], 0.3 * Math.abs(v), f, v >= 0 ? 0 : Math.PI, 10);
    m.addResonantRegion(x - 5, x + 5, my - 5, my + 5, f);
    m.setMemory([x, my], v);
  });
  m.spacing = spacing;
  return m;
}
export function addRetrievalPulses(m) {
  for (let x = m.spacing; x < m.width; x += m.spacing) m.addSource([x, m.height - 5], 0.5, 0.65, 0, 5);
}
// Retrieved value = 0.7 * stored (decayed) memory value + 0.3 * 2 * mean activity in a 7 x 7 box over 5 frames
export function memoryReadout(m, n) {
  const W = m.width, H = m.height, y = Math.floor(H / 2), frames = m.history.slice(-5);
  return Array.from({ length: n }, (_, i) => {
    const x = (i + 1) * m.spacing;
    const x0 = Math.max(0, x - 3), x1 = Math.min(W, x + 4), y0 = Math.max(0, y - 3), y1 = Math.min(H, y + 4);
    if (!frames.length) return 0;
    const acts = frames.map(f => { let s = 0; for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) s += f[yy * W + xx]; return s / ((y1 - y0) * (x1 - x0)); });
    const avg = acts.reduce((a, b) => a + b, 0) / acts.length;
    return 0.7 * m.readMemory([x, y]) + 0.3 * (avg * 2);
  });
}

// WaveComputing.wave_filter set-up: one source per sample down the left edge, a resonant band per row in the middle
// third tuned to cutoff/2 (lowpass), 1.5 x cutoff (highpass) or cutoff (bandpass).
export function filterMedium(base, signal, type = 'lowpass', cutoff = 0.5) {
  const m = fromMedium(base, { boundary: 'absorbing', damping: 0.99, diffusion: 0.1, nonlinearity: 0.1 });
  const { width: W, height: H } = m, scale = Math.min(1.0, H / signal.length);
  for (let i = 0; i < signal.length; i++) {
    const y = Math.trunc(i * scale); if (y >= H) break;
    m.addSource([5, y], 0.4 * Math.abs(signal[i]), 0.3 + 0.6 * (signal[i] + 1) / 2);
  }
  const f = { lowpass: cutoff / 2, highpass: cutoff * 1.5, bandpass: cutoff }[type];
  if (f !== undefined) for (let y = 0; y < H; y++) m.addResonantRegion(Math.floor(W / 3), Math.floor(2 * W / 3), Math.max(0, y - 2), Math.min(H, y + 3), f);
  m.signalScale = scale;
  return m;
}
// Output: mean over the last 10 frames at x = 4/5 of the width, per sample row, scaled to max |value| = 1
export function filterReadout(m, n) {
  const W = m.width, H = m.height, ox = Math.trunc(4 * W / 5), frames = m.history.slice(-10), out = [];
  for (let i = 0; i < Math.min(n, H); i++) {
    const y = Math.trunc(i * m.signalScale); if (y >= H) break;
    out.push(frames.length ? frames.reduce((s, f) => s + f[y * W + ox], 0) / frames.length : 0);
  }
  const mx = out.length ? Math.max(Math.abs(Math.min(...out)), Math.abs(Math.max(...out))) : 0;
  return mx > 0 ? out.map(v => v / mx) : out;
}
