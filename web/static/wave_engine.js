// Wave medium engine: a line-by-line port of WaveComputingMedium (wave_based_computing/wave_medium.py) and of
// WaveComputing's logic-gate, memory and filter set-ups (wave_computing.py). Deterministic (no random numbers);
// wave_engine.test.mjs checks it against tests/fixtures/wave.json.
//
// Damped wave equation u_tt = c^2 lap(u) - gamma u_t - nonlinearity u^3, leapfrog with a 9-point Laplacian; each
// output step is split into substeps chosen from the CFL condition, so every wave speed is stable.
import { convolve, normalise } from './grid.js';

const KERNEL = normalise(Float64Array.from([0.05, 0.2, 0.05, 0.2, 0.0, 0.2, 0.05, 0.2, 0.05]));
export const LAPLACIAN_SCALE = 0.3, LAMBDA_MAX = 1.6 / 0.3, CFL_SAFETY = 0.9, SPONGE_WIDTH = 5;
// numpy slice bounds: negative start/stop count from the end, then clamp to [0, n]
const sl = (v, n) => Math.min(n, Math.max(0, v < 0 ? v + n : v));

export const WAVE_DEFAULTS = { width: 100, height: 100, boundary: 'absorbing', damping: 0.99, waveSpeed: 10.0, nonlinearity: 0.1, timeStep: 0.1, dx: 1.0 };

export class WaveMedium {
  constructor(opts = {}) {
    const o = { ...WAVE_DEFAULTS, ...opts };
    Object.assign(this, o);
    this.diffusion = this.waveSpeed ** 2;
    const N = o.width * o.height;
    this.current = new Float64Array(N); this.previous = new Float64Array(N);
    this.frequency = new Float64Array(N); this.phase = new Float64Array(N);
    this.obstacles = new Uint8Array(N);
    this.sources = []; this.resonant = []; this.memory = new Map();
    this.history = []; this.energy = []; this.steps = 0;
    this.setTiming();
    this.sponge = this._sponge();
  }

  // Substeps per output step from the CFL condition c * dt * sqrt(LAMBDA_MAX) / dx <= 2 (times CFL_SAFETY)
  setTiming() {
    if (this.waveSpeed <= 0) { this.substeps = 1; this.dtSub = this.timeStep; return; }
    const dtMax = CFL_SAFETY * 2.0 * this.dx / (this.waveSpeed * Math.sqrt(LAMBDA_MAX));
    this.substeps = Math.max(1, Math.ceil(this.timeStep / dtMax - 1e-12));
    this.dtSub = this.timeStep / this.substeps;
  }

  _sponge() {
    const { width: W, height: H } = this, m = new Float64Array(W * H).fill(1), bw = SPONGE_WIDTH;
    for (let i = 0; i < bw; i++) {
      const f = Math.exp(-((0.2 * (bw - i)) ** 2));
      for (const r of [i, H - 1 - i]) if (r >= 0 && r < H) for (let x = 0; x < W; x++) m[r * W + x] = Math.min(m[r * W + x], f);
      for (const c of [i, W - 1 - i]) if (c >= 0 && c < W) for (let y = 0; y < H; y++) m[y * W + c] = Math.min(m[y * W + c], f);
    }
    return m;
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

  laplacian(u) {
    const mean = convolve(u, this.height, this.width, KERNEL, 3, this.boundary === 'periodic' ? 'wrap' : 'symm');
    const s = LAPLACIAN_SCALE * this.dx ** 2, out = new Float64Array(u.length);
    for (let i = 0; i < u.length; i++) out[i] = (mean[i] - u[i]) / s;
    return out;
  }

  // Conserved discrete energy (no damping, sources, nonlinearity or sponge, periodic grid)
  waveEnergy() {
    const lp = this.laplacian(this.previous);
    let kin = 0, pot = 0;
    for (let i = 0; i < lp.length; i++) { const v = (this.current[i] - this.previous[i]) / this.dtSub; kin += v * v; pot += this.current[i] * lp[i]; }
    return 0.5 * kin - 0.5 * this.diffusion * pot;
  }

  // Fraction of velocity kept per substep; no loss in a resonant region whose dominant frequency matches its tuning
  _velocityFactor() {
    const { width: W, height: H } = this, keep = new Float64Array(W * H).fill(this.damping ** (1.0 / this.substeps));
    if (this.history.length > 10) {
      for (const r of this.resonant) {
        const lf = this.localFrequency(r), match = Math.exp(-(((lf - r.frequency) / 0.1) ** 2));
        for (let y = sl(r.y0, H); y < sl(r.y1, H); y++) for (let x = sl(r.x0, W); x < sl(r.x1, W); x++) {
          const k = y * W + x; keep[k] = Math.max(keep[k], keep[k] ** (1.0 - match));
        }
      }
    }
    return keep;
  }

  _sourceField(t) {
    const { width: W, height: H } = this, field = new Float64Array(W * H), covered = new Uint8Array(W * H);
    for (const s of this.sources) {
      if (s.duration !== null && t >= s.start + s.duration) continue;
      const v = s.amplitude * Math.sin(2 * Math.PI * s.frequency * t + s.phase);
      for (let dx = -1; dx < 2; dx++) for (let dy = -1; dy < 2; dy++) {
        const nx = s.x + dx, ny = s.y + dy;
        if (nx >= 0 && nx < W && ny >= 0 && ny < H) { field[ny * W + nx] += v * Math.exp(-(dx ** 2 + dy ** 2) / 2); covered[ny * W + nx] = 1; }
      }
    }
    return [field, covered];
  }

  update(t) {
    const { width: W } = this, N = this.width * this.height, dt = this.dtSub, keep = this._velocityFactor();
    for (let k = 0; k < this.substeps; k++) {
      let u = this.current; const up = this.previous, lap = this.laplacian(u), next = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        let acc = this.diffusion * lap[i];
        if (this.nonlinearity > 0) acc = acc - this.nonlinearity * u[i] ** 3;
        next[i] = u[i] + keep[i] * (u[i] - up[i]) + dt ** 2 * acc;
        if (this.obstacles[i]) next[i] = 0;
      }
      if (this.boundary === 'absorbing') {
        const sp = this.sponge, us = new Float64Array(N);
        for (let i = 0; i < N; i++) { next[i] *= sp[i]; us[i] = u[i] * sp[i]; }
        u = us;
      }
      const [field, covered] = this._sourceField(t + (k + 1) * dt);
      for (let i = 0; i < N; i++) if (covered[i]) next[i] = field[i];
      for (const m of this.memory.values()) next[m[1] * W + m[0]] = m[2];
      this.previous = u; this.current = next;
    }
    for (const m of this.memory.values()) m[2] = m[2] * m[3];
    if (this.history.length > 5) this._frequencyPhase();
    this.history.push(this.current.slice());
    if (this.history.length > 100) this.history.shift();
    let e = 0; for (const v of this.current) e += v * v;
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

const fromMedium = (m, extra = {}) => new WaveMedium({ width: m.width, height: m.height, boundary: m.boundary, damping: m.damping, waveSpeed: m.waveSpeed, nonlinearity: m.nonlinearity, timeStep: m.timeStep, ...extra });

// WaveComputing gates: input A = line of sources at x = 5 (rows H/4..3H/4), input B = line at y = 5 (columns
// W/4..3W/4), each on when its bit is 1; B in antiphase for XOR. Output: centre amplitude vs one input alone.
export const GATE = { amplitude: 0.3, frequency: 0.5, thresholds: { and: 1.5, or: 0.5, xor: 0.5 } };

export function gateMedium(base, [a, b], op) {
  const m = fromMedium(base, { boundary: 'absorbing' }), { width: W, height: H } = m;
  if (a) for (let y = Math.floor(H / 4); y < Math.floor(3 * H / 4); y++) m.addSource([5, y], GATE.amplitude, GATE.frequency, 0);
  if (b) for (let x = Math.floor(W / 4); x < Math.floor(3 * W / 4); x++) m.addSource([x, 5], GATE.amplitude, GATE.frequency, op === 'xor' ? Math.PI : 0);
  return m;
}

// Largest |mean of the 3 x 3 centre cells| over the last `window` frames
export function gateReadout(m, window = 20) {
  const W = m.width, cx = Math.floor(m.width / 2), cy = Math.floor(m.height / 2), frames = m.history.slice(-window);
  let best = 0;
  for (const f of frames) {
    let s = 0;
    for (let y = cy - 1; y < cy + 2; y++) for (let x = cx - 1; x < cx + 2; x++) s += f[y * W + x];
    best = Math.max(best, Math.abs(s / 9));
  }
  return best;
}

// Full gate evaluation as wave_logic_gates: returns { inputs, output, value, reference, threshold }
export function logicGate(base, bits, op, steps = 100) {
  const m = gateMedium(base, bits, op); m.run(steps);
  const ref = gateMedium(base, [true, false], 'or'); ref.run(steps);
  const value = gateReadout(m), reference = gateReadout(ref), threshold = GATE.thresholds[op] * reference;
  return { inputs: bits, output: value > threshold, value, reference, threshold, medium: m };
}

// WaveComputing.wave_memory set-up; storage then retrieval (retrieval adds pulse sources along the bottom).
export function memoryMedium(base, data) {
  const m = fromMedium(base, { boundary: 'absorbing', damping: 0.995, nonlinearity: 0.05 });
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
  const m = fromMedium(base, { boundary: 'absorbing', damping: 0.99, nonlinearity: 0.1 });
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
