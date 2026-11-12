// Multi-store engine: a line-by-line port of CalciumStore (multistore_computing/store.py) and MultiStoreComputer
// (computer.py), plus HybridComputing.hierarchical_memory (hybrid.py). Random numbers come from np_random.js in
// numpy's order (store_engine.test.mjs checks against tests/fixtures/store.json).

export const STORE_SPECS = {
  primary: { name: 'Primary', size: 100, speed: 0.9, capacity: 1.0, reliability: 0.99, analogRatio: 0.7, persistence: 0.9 },
  secondary: { name: 'Secondary', size: 200, speed: 0.5, capacity: 1.5, reliability: 0.95, analogRatio: 0.5, persistence: 0.97 },
  tertiary: { name: 'Tertiary', size: 500, speed: 0.2, capacity: 3.0, reliability: 0.9, analogRatio: 0.3, persistence: 0.99 },
};
export const CONNECTIONS = [
  ['primary', 'secondary', 0.8], ['secondary', 'primary', 0.5], ['secondary', 'tertiary', 0.7],
  ['tertiary', 'secondary', 0.4], ['primary', 'tertiary', 0.3], ['tertiary', 'primary', 0.2],
];
const clip01 = v => Math.min(1.0, Math.max(0.0, v));
const isSeq = d => Array.isArray(d) || ArrayBuffer.isView(d);

export class Store {
  constructor(rng, spec) {
    Object.assign(this, spec); this.rng = rng;
    this.activation = new Float64Array(this.size);
    this.contents = Array.from({ length: this.size }, () => [0, 0, 0]);
    this.threshold = 0.3;
  }

  // address: index or { start, stop } (a Python slice with step 1)
  write(address, data, level = 1.0) {
    if (typeof address === 'object') {
      const start = Math.max(0, Math.min(address.start ?? 0, this.size - 1)), stop = Math.max(0, Math.min(address.stop ?? this.size, this.size));
      if (stop <= start) return false;
      for (let a = start; a < stop; a++) this._writeOne(a, isSeq(data) && data.length >= stop - start ? data[a - start] : data, level);
      return true;
    }
    return address >= 0 && address < this.size ? this._writeOne(address, data, level) : false;
  }

  _writeOne(a, data, level) {
    if (this.rng.random() > this.reliability) return false;      // unreliable write: nothing happens
    if (data === null || data === undefined) this.contents[a] = [0, 0, 0];
    else if (typeof data === 'number') { const v = clip01(data); this.contents[a] = [v, v * 0.8, v * 0.5]; }
    else if (data.length === 3) this.contents[a] = Array.from(data);
    else { const v = clip01(+data[0]); this.contents[a] = [v, v * 0.8, v * 0.5]; }
    this.activation[a] = Math.min(this.capacity, level);
    return true;
  }

  read(a) {
    if (!(a >= 0 && a < this.size)) return null;
    if (this.rng.random() > this.reliability) return null;       // unreliable read
    return this.contents[a];
  }

  // Activation decays by `persistence`, gets N(0, 0.01) noise and is clipped to [0, capacity]
  update() {
    const noise = this.rng.normalN(0, 0.01, this.size);
    for (let i = 0; i < this.size; i++) {
      this.activation[i] = Math.min(this.capacity, Math.max(0, this.activation[i] * this.persistence + noise[i]));
    }
    for (const c of this.contents) for (let k = 0; k < 3; k++) c[k] = clip01(c[k]);
  }

  // Best match: similarity max(0, 1 - Euclidean distance) weighted by activation
  query(q) {
    const v = q.length !== 3 ? [+q[0], q[0] * 0.8, q[0] * 0.5] : q;
    let best = -1, bestW = -Infinity, mx = -Infinity;
    for (let i = 0; i < this.size; i++) {
      const c = this.contents[i];
      const sim = Math.max(0, 1 - Math.sqrt((c[0] - v[0]) ** 2 + (c[1] - v[1]) ** 2 + (c[2] - v[2]) ** 2));
      const w = sim * this.activation[i];
      if (w > bestW) { bestW = w; best = i; }
      if (w > mx) mx = w;
    }
    return mx > 0 ? { result: this.contents[best].slice(), confidence: bestW, index: best } : { result: [0, 0, 0], confidence: 0, index: -1 };
  }
}

// Start of the first run of `length` quiet cells (activation < 0.1), else of the window with the lowest total activation
function freeSpan(st, length) {
  let run = 0;
  for (let i = 0; i < st.size; i++) { run = st.activation[i] < 0.1 ? run + 1 : 0; if (run === length) return i - length + 1; }
  let best = 0, bestSum = Infinity;
  for (let i = 0; i + length <= st.size; i++) {
    let s = 0; for (let k = i; k < i + length; k++) s += st.activation[k];
    if (s < bestSum) { bestSum = s; best = i; }
  }
  return best;
}

export class MultiStore {
  constructor(rng) {
    this.rng = rng;
    this.stores = Object.fromEntries(Object.entries(STORE_SPECS).map(([k, s]) => [k, new Store(rng, s)]));
    this.timeStep = 0; this.meanHistory = []; this.transfers = []; this.writeLog = [];
  }

  update() {
    for (const s of Object.values(this.stores)) s.update();
    this._crossStore();
    this.meanHistory.push(Object.fromEntries(Object.entries(this.stores).map(([k, s]) => [k, s.activation.reduce((a, b) => a + b, 0) / s.size])));
    this.timeStep++;
  }

  // Strongly active cells (> 0.7) are copied to the proportional address of the connected store, a random
  // fraction `strength` of them per connection, with activation scaled by the strength
  _crossStore() {
    const out = [];
    for (const [sn, tn, strength] of CONNECTIONS) {
      const src = this.stores[sn], tgt = this.stores[tn], high = [];
      let anyActive = false;
      for (let i = 0; i < src.size; i++) if (src.activation[i] > src.threshold) { anyActive = true; if (src.activation[i] > 0.7) high.push(i); }
      if (!anyActive || !high.length) continue;
      const nT = Math.max(1, Math.trunc(high.length * strength));
      const pick = this.rng.choice(high.length, Math.min(nT, high.length));
      for (const p of pick) {
        const idx = high[p], ti = Math.trunc(idx * tgt.size / src.size) % tgt.size;
        const data = src.read(idx), act = src.activation[idx] * strength;
        tgt.write(ti, data, act);
        out.push([sn, tn, idx, ti, act]);
      }
    }
    this.transfers.push(out);
  }

  // priority: 'high' -> primary, 'medium' -> secondary, 'low' -> tertiary; scalars go to a random quiet cell
  // (activation < 0.1), sequences to the first run of quiet cells long enough (writeLog records where)
  writeData(data, priority = 'auto') {
    if (priority === 'auto') priority = isSeq(data) && data.length > 100 ? 'low' : 'medium';
    const name = { high: 'primary', medium: 'secondary', low: 'tertiary' }[priority] || 'secondary', st = this.stores[name];
    if (isSeq(data)) {
      if (data.length > st.size) {
        const ok = st.write({ start: 0, stop: st.size }, data.slice(0, st.size));
        this.writeLog.push({ store: name, start: 0, length: st.size });
        const [ok2] = this.writeData(Array.from(data.slice(st.size)), 'low');
        return [ok && ok2, `${name} + overflow`];
      }
      const start = freeSpan(st, data.length), ok = st.write({ start, stop: start + data.length }, data);
      this.writeLog.push({ store: name, start, length: data.length });
      return [ok, name];
    }
    const empty = [];
    for (let i = 0; i < st.size; i++) if (st.activation[i] < 0.1) empty.push(i);
    let address;
    if (empty.length) address = this.rng.choiceOf(empty);
    else { address = 0; for (let i = 1; i < st.size; i++) if (st.activation[i] < st.activation[address]) address = i; }
    this.writeLog.push({ store: name, start: address, length: 1 });
    return [st.write(address, data), name];
  }

  query(q) {
    const v = isSeq(q) && q.length === 3 ? q : typeof q === 'number' ? [q, q * 0.8, q * 0.5] : [0.5, 0.4, 0.3];
    const res = [];
    for (const name of ['primary', 'secondary', 'tertiary']) {
      const r = this.stores[name].query(v);
      if (r.confidence > 0.4) { res.push({ ...r, source: name }); if (r.confidence > 0.8 && name === 'primary') break; }
    }
    return res.sort((a, b) => b.confidence - a.confidence);
  }

  run(steps) { for (let k = 0; k < steps; k++) this.update(); }
}

// HybridComputing.hierarchical_memory: write a sequence with priorities high, medium, low, high, ...; run; query.
export function hierarchicalMemory(ms, dataSequence, queries, steps = 50) {
  const writes = dataSequence.map((d, i) => { const pr = ['high', 'medium', 'low'][i % 3]; const [ok, store] = ms.writeData(d, pr); return { data: d, priority: pr, success: ok, store }; });
  ms.run(steps);
  return { writes, retrievals: queries.map(q => ({ query: q, results: ms.query(q) })) };
}
