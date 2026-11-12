// Threshold computing engine: a line-by-line port of ThresholdComputingUnit, ThresholdComputingNetwork
// (threshold_computing/threshold_computing.py) and ThresholdLogicComputer (threshold_logic.py). Random numbers come
// from np_random.js in numpy's order, so a seed replays the Python run (threshold_engine.test.mjs, threshold.json).

export class Unit {
  constructor({ threshold = 0.7, activationSlope = 10, refractoryPeriod = 5, integrationRate = 0.1, recoveryRate = 0.05 } = {}) {
    Object.assign(this, { threshold, activationSlope, refractoryPeriod, integrationRate, recoveryRate });
    this.reset();
  }

  reset() { this.potential = 0.0; this.activation = 0.0; this.countdown = 0; this.internal = 0.0; }

  update(input) {
    if (this.countdown > 0) this.countdown -= 1;
    const frac = this.refractoryPeriod > 0 ? this.countdown / this.refractoryPeriod : 0.0;
    const eff = this.threshold + 0.2 * frac;                       // threshold raised while refractory
    const integ = this.integrationRate * (1.0 - 0.8 * frac);      // and input integrated more weakly
    this.potential += integ * input + 0.01 * this.internal;
    this.potential *= 0.95;                                       // leak
    const x = this.activationSlope * (this.potential - eff);
    this.activation = 1.0 / (1.0 + Math.exp(-x));
    this.internal = 0.99 * this.internal + 0.01 * this.activation;  // slow internal state
    if (this.potential > eff && this.countdown === 0) this.countdown = this.refractoryPeriod;
    return this.activation;
  }
}

export class ThresholdNetwork {
  constructor(rng, { height = 10, width = 10, radius = 2, prob = 0.3 } = {}) {
    this.rng = rng; this.H = height; this.W = width;
    this.units = [];
    for (let i = 0; i < height; i++) for (let j = 0; j < width; j++) {
      const threshold = 0.7 + 0.1 * rng.gauss(), slope = 10 + rng.gauss();
      this.units.push(new Unit({ threshold: Math.max(0.4, threshold), activationSlope: Math.max(5, slope) }));
    }
    this.connections = this._connections(radius, prob);   // connections[k] = [[sourceIndex, weight], ...]
    this.input = []; this.output = [];
    for (let i = 0; i < height; i++) for (let j = 0; j < width; j++) {
      if (i < 2) this.input.push(i * width + j);
      if (i >= height - 2) this.output.push(i * width + j);
    }
    this.activationHistory = []; this.potentialHistory = [];
  }

  // Each pair within `radius` is connected with probability `prob`, weight 0.5 + 0.5 N(0, 1) (can be negative)
  _connections(radius, prob) {
    const { H, W, rng } = this, out = [];
    for (let i1 = 0; i1 < H; i1++) for (let j1 = 0; j1 < W; j1++) {
      const list = [];
      for (let i2 = Math.max(0, i1 - radius); i2 < Math.min(H, i1 + radius + 1); i2++) {
        for (let j2 = Math.max(0, j1 - radius); j2 < Math.min(W, j1 + radius + 1); j2++) {
          if (i1 === i2 && j1 === j2) continue;
          const dist = Math.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2);
          if (dist <= radius && rng.random() < prob) list.push([i2 * W + j2, 0.5 + 0.5 * rng.gauss()]);
        }
      }
      out.push(list);
    }
    return out;
  }

  get nConnections() { return this.connections.reduce((s, l) => s + l.length, 0); }

  reset() { this.units.forEach(u => u.reset()); this.activationHistory = []; this.potentialHistory = []; }

  setInput(pattern) {
    if (pattern.length !== this.input.length) throw new Error(`Input pattern length (${pattern.length}) must match input region size (${this.input.length})`);
    this.input.forEach((k, i) => { this.units[k].potential = pattern[i]; });
  }

  getOutput() { return this.output.map(k => this.units[k].activation); }

  update() {
    const n = this.units.length, inputs = new Float64Array(n);
    for (let k = 0; k < n; k++) for (const [s, w] of this.connections[k]) inputs[k] += w * this.units[s].activation;
    const act = new Float64Array(n), pot = new Float64Array(n);
    for (let k = 0; k < n; k++) { act[k] = this.units[k].update(inputs[k]); pot[k] = this.units[k].potential; }
    this.activationHistory.push(act); this.potentialHistory.push(pot);
    if (this.activationHistory.length > 100) { this.activationHistory.shift(); this.potentialHistory.shift(); }
  }

  run(steps, patterns = null) {
    const outputs = [];
    for (let s = 0; s < steps; s++) {
      if (patterns && s < patterns.length) this.setInput(patterns[s]);
      this.update();
      outputs.push(this.getOutput());
    }
    return outputs;
  }
}

const zeros = n => new Array(n).fill(0);

// ThresholdLogicComputer. Each experiment can be run whole (returns results) or stepped: plan() returns the list of
// input patterns per step, so the page can animate it.
export class ThresholdLogic {
  constructor(net) { this.net = net; }

  temporalIntegration(sequence, duration = 50) {
    const pad = sequence.length < duration ? sequence.concat(Array.from({ length: duration - sequence.length }, () => zeros(this.net.input.length))) : sequence.slice(0, duration);
    return this.net.run(duration, pad);
  }

  thresholdDetection(strengths, perSignal = 20) {
    return strengths.map(s => {
      this.net.reset();
      const pat = new Array(this.net.input.length).fill(s);
      const out = this.net.run(perSignal, Array.from({ length: perSignal }, () => pat));
      const mx = Math.max(...out.map(o => Math.max(...o)));
      return [s, mx, mx > 0.5];
    });
  }

  analogMemory(values, retention = 30) {
    return values.map(v => {
      this.net.reset();
      const n = this.net.input.length, seq = [];
      for (let k = 0; k < 5; k++) seq.push(new Array(n).fill(v));
      for (let k = 5; k < retention; k++) seq.push(zeros(n));
      return [v, this.net.run(retention, seq).map(o => Math.max(...o))];
    });
  }
}
