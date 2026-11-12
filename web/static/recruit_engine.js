// Coordinated recruitment engine: a line-by-line port of CoordinatedRecruitmentElement (element.py),
// CoordinatedRecruitmentNetwork (network.py, including sklearn's DBSCAN(eps=1.5, min_samples=3) cluster labelling)
// and QuorumComputer's fault-tolerance and amplification experiments (quorum.py). Random numbers come from
// np_random.js in numpy's order (recruit_engine.test.mjs checks against tests/fixtures/recruit.json).
import { convolve } from './grid.js';

export class Element {
  constructor({ sensitivity = 0.5, threshold = 0.3, coupling = 0.2, recoveryTime = 5, maxActivation = 1.0 } = {}) {
    Object.assign(this, { sensitivity, threshold, coupling, recoveryTime, maxActivation });
    this.activation = 0.0; this.countdown = 0; this.neighbourhood = 0.0; this.fatigue = 0.0;
  }

  update(direct, neighbourhood) {
    this.neighbourhood = neighbourhood;
    let eff = this.sensitivity * (1.0 - 0.5 * this.fatigue);          // fatigue lowers sensitivity
    if (this.countdown > 0) { this.countdown -= 1; eff *= 0.5; }       // and so does recovery
    const total = direct * eff + neighbourhood * this.coupling;
    let signal = 0.0;
    if (total > this.threshold && this.countdown === 0) {
      this.activation = Math.min(this.maxActivation, total);
      signal = this.activation * 0.8;
      this.fatigue = Math.min(1.0, this.fatigue + 0.1);
      this.countdown = this.recoveryTime;
    } else {
      this.activation *= 0.8;
      this.fatigue = Math.max(0.0, this.fatigue - 0.01);
    }
    return signal;
  }
}

// DBSCAN as sklearn labels it: points in order, core points (>= minSamples neighbours within eps, self included)
// seed clusters, a border point joins the first cluster that reaches it. Returns clusters as arrays of point indices.
// Points are integer grid coordinates, so neighbours are found by scanning the offsets within eps.
export function dbscan(points, eps = 1.5, minSamples = 3) {
  const n = points.length, at = new Map(points.map(([a, b], i) => [`${a},${b}`, i])), R = Math.ceil(eps);
  const nb = points.map(([a, b]) => {
    const out = [];
    for (let da = -R; da <= R; da++) for (let db = -R; db <= R; db++) {
      const j = at.get(`${a + da},${b + db}`);
      if (j !== undefined && Math.sqrt(da * da + db * db) <= eps) out.push(j);
    }
    return out;
  });
  const core = nb.map(l => l.length >= minSamples), labels = new Int32Array(n).fill(-1);
  let label = 0;
  for (let i0 = 0; i0 < n; i0++) {
    if (labels[i0] !== -1 || !core[i0]) continue;
    const stack = []; let i = i0;
    for (;;) {
      if (labels[i] === -1) {
        labels[i] = label;
        if (core[i]) for (const j of nb[i]) if (labels[j] === -1) stack.push(j);
      }
      if (!stack.length) break;
      i = stack.pop();
    }
    label++;
  }
  const clusters = new Map();
  for (let k = 0; k < n; k++) if (labels[k] !== -1) { if (!clusters.has(labels[k])) clusters.set(labels[k], []); clusters.get(labels[k]).push(k); }
  return [...clusters.values()];
}

export class RecruitNetwork {
  constructor(rng, { height = 30, width = 30, radius = 3, faultProb = 0.0 } = {}) {
    this.rng = rng; this.H = height; this.W = width; this.radius = radius; this.faultProb = faultProb;
    this.elements = []; this.faulty = new Uint8Array(height * width);
    for (let k = 0; k < height * width; k++) {
      const sensitivity = Math.max(0.2, Math.min(0.8, 0.4 + 0.2 * rng.gauss()));
      const threshold = Math.max(0.1, Math.min(0.5, 0.3 + 0.1 * rng.gauss()));
      this.elements.push(new Element({ sensitivity, threshold }));
      if (rng.random() < faultProb) this.faulty[k] = 1;
    }
    this.ks = 2 * radius + 1; this.kernel = this._kernel(radius);
    this.activationHistory = []; this.clusterHistory = [];
    this.direct = new Float64Array(height * width);
  }

  // exp(-distance / (radius / 2)) inside the radius, zero centre, normalised to sum 1
  _kernel(r) {
    const size = 2 * r + 1, k = new Float64Array(size * size);
    for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) {
      const d = Math.sqrt((i - r) ** 2 + (j - r) ** 2);
      if (d <= r) k[i * size + j] = Math.exp(-d / (r / 2));
    }
    k[r * size + r] = 0;
    let s = 0; for (const v of k) s += v;
    if (s > 0) for (let i = 0; i < k.length; i++) k[i] /= s;
    return k;
  }

  setInputRegion([ci, cj], radius, strength) {
    for (let i = 0; i < this.H; i++) for (let j = 0; j < this.W; j++) {
      const d = Math.sqrt((i - ci) ** 2 + (j - cj) ** 2);
      if (d <= radius) this.direct[i * this.W + j] = strength * (1 - d / radius);
    }
  }

  setInputPattern(p) { this.direct = Float64Array.from(p); }

  activationGrid() {
    return Float64Array.from(this.elements, (e, k) => (this.faulty[k] ? 0 : e.activation));
  }

  update() {
    const neigh = convolve(this.activationGrid(), this.H, this.W, this.kernel, this.ks, 'constant');
    this.elements.forEach((e, k) => { if (!this.faulty[k]) e.update(this.direct[k], neigh[k]); });
    const cur = this.activationGrid();
    this.activationHistory.push(cur);
    this.clusterHistory.push(this.identifyClusters(cur));
    if (this.activationHistory.length > 100) { this.activationHistory.shift(); this.clusterHistory.shift(); }
  }

  // Clusters of elements with activation > 0.3 (arrays of grid indices)
  identifyClusters(grid, thr = 0.3) {
    const idx = [];
    for (let k = 0; k < grid.length; k++) if (grid[k] > thr) idx.push(k);
    if (!idx.length) return [];
    const pts = idx.map(k => [Math.floor(k / this.W), k % this.W]);
    return dbscan(pts).map(c => c.map(p => idx[p]));
  }

  run(steps, patternFn = null) {
    for (let s = 0; s < steps; s++) { if (patternFn) this.setInputPattern(patternFn(s)); this.update(); }
    return this.analyze();
  }

  analyze() {
    const sizes = this.clusterHistory.map(cs => (cs.length ? cs.map(c => c.length) : [0]));
    return {
      avgClusterSizes: sizes.map(s => s.reduce((a, b) => a + b, 0) / s.length),
      numClusters: this.clusterHistory.map(cs => cs.length),
      totalActivation: this.activationHistory.map(a => a.reduce((x, y) => x + y, 0)),
    };
  }
}

// QuorumComputer experiments. Each builds fresh networks from the shared generator, as the Python does.
export function faultTolerance(rng, { height, width, radius }, patterns, faultProbs, trials = 5) {
  return faultProbs.map(fp => {
    const rows = [];
    for (let t = 0; t < trials; t++) {
      const net = new RecruitNetwork(rng, { height, width, radius, faultProb: fp });
      for (const p of patterns) {
        net.setInputPattern(p);
        const a = net.run(20);
        rows.push({ act: a.totalActivation.at(-1) ?? 0, maxCluster: a.avgClusterSizes.length ? Math.max(...a.avgClusterSizes) : 0, n: a.numClusters.at(-1) ?? 0 });
      }
    }
    const avg = f => rows.reduce((s, r) => s + f(r), 0) / rows.length;
    return { faultProb: fp, avgActivation: avg(r => r.act), avgMaxCluster: avg(r => r.maxCluster), avgNumClusters: avg(r => r.n) };
  });
}

export function amplification(rng, { height, width, radius }, strengths, duration = 20) {
  return strengths.map(s => {
    const p = new Float64Array(height * width), ci = Math.floor(height / 2), cj = Math.floor(width / 2), r = Math.floor(Math.min(height, width) / 6);
    for (let i = 0; i < height; i++) for (let j = 0; j < width; j++) if (Math.sqrt((i - ci) ** 2 + (j - cj) ** 2) <= r) p[i * width + j] = s;
    const net = new RecruitNetwork(rng, { height, width, radius });
    net.setInputPattern(p); net.run(duration);
    const input = p.reduce((a, b) => a + b, 0), series = net.activationHistory.map(a => a.reduce((x, y) => x + y, 0));
    const fin = series.at(-1) ?? 0;
    return { strength: s, input, final: fin, ratio: input > 0 ? fin / input : 0, series };
  });
}
