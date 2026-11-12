// Model 5 page: multi-store hybrid computing (MultiStoreComputer + HybridComputing.hierarchical_memory via
// store_engine.js): three stores with different size, capacity, reliability and persistence exchange strong activity.
import { route, esc, list } from './core.js';
import { NumpyRandom } from './np_random.js';
import { MultiStore, STORE_SPECS, CONNECTIONS } from './store_engine.js';
import { lab, fig } from './lab.js';
import { lineChart, fitCanvas, fmt, COLORS } from './chart.js';

const NAMES = Object.keys(STORE_SPECS);
const PRESETS = [['demo', 'Demo: 0.9 (high), [0.5, 0.2, 0.7] (medium), 50 random values (low)'],
  ['hier', 'Hierarchical memory: 0.8, [0.2, 0.4, 0.6], 0.5, [0.1, 0.3, 0.5], 0.3'], ['empty', 'Empty stores']];

route('stores', () => {
  let ms, steps, hier = null, rng;
  const L = lab({
    kicker: 'Model 5 · multi-store hybrid computing', title: 'Three stores that pass activity between them', route: 'stores',
    lead: `Each store is an array of cells holding an RGB-like 3-vector (content) and an activation level. Writes go to
      a store chosen by priority (high → primary, medium → secondary, low → tertiary) and can fail at random
      (reliability). Every step activation decays by the store's persistence factor, gains a little noise, and cells above
      0.7 are copied (with activation scaled by the link strength) to the proportional address of connected stores.
      Queries return the best-matching content weighted by activation.`,
    seed: 42, maxSpeed: 10, speed: 1,
    controls: [
      { id: 'preset', label: 'Initial writes', type: 'select', value: 'demo', options: PRESETS },
      ...NAMES.map(n => ({ id: 'p_' + n, label: `${STORE_SPECS[n].name} persistence`, min: 0.5, max: 0.999, step: 0.001, value: STORE_SPECS[n].persistence, unit: 'per step' })),
      ...NAMES.map(n => ({ id: 'r_' + n, label: `${STORE_SPECS[n].name} reliability`, min: 0.5, max: 1, step: 0.01, value: STORE_SPECS[n].reliability, hint: 'probability that one read or write succeeds' })),
      { id: 'steps', label: 'Steps to run', min: 10, max: 200, step: 10, value: 50, unit: 'steps' },
    ],
    figures: `${list(NAMES, n => fig('s_' + n, 600, 110, `${STORE_SPECS[n].name}: ${STORE_SPECS[n].size} cells, capacity ${STORE_SPECS[n].capacity}. Bars: activation; strip: content colour`))}
      ${fig('chart', 600, 220, 'Mean activation of each store per step')}`,
    onReset(p) {
      rng = new NumpyRandom(p.seed); ms = new MultiStore(rng);
      NAMES.forEach(n => { ms.stores[n].persistence = p['p_' + n]; ms.stores[n].reliability = p['r_' + n]; });
      hier = null;
      if (p.preset === 'demo') [[0.9, 'high'], [[0.5, 0.2, 0.7], 'medium'], [Array.from({ length: 50 }, () => rng.random()), 'low']].forEach(([d, pr]) => ms.writeData(d, pr));
      if (p.preset === 'hier') hier = [0.8, [0.2, 0.4, 0.6], 0.5, [0.1, 0.3, 0.5], 0.3].map((d, i) => { const pr = ['high', 'medium', 'low'][i % 3]; const [ok, st] = ms.writeData(d, pr); return [d, pr, ok, st]; });
      steps = p.steps;
    },
    step: () => ms.update(), done: () => ms.timeStep >= steps, render,
    source: 'multistore_computing/store.py, computer.py and hybrid.py',
    after: `<div class="panel"><h3>Write and query</h3><div class="row">
      <div><label for="wv">Value <b id="wvv">0.6</b></label><input type="range" id="wv" min="0" max="1" step="0.01" value="0.6"></div>
      <div><label for="wp">Priority</label><select id="wp"><option value="high">high → primary</option><option value="medium" selected>medium → secondary</option><option value="low">low → tertiary</option></select></div>
      <div><button id="wb">Write</button><button class="alt" id="qb">Query value</button></div></div>
      <div id="wq" class="small" aria-live="polite"></div></div>`,
    notes: `<h2>Stores and links as coded</h2>
      <div class="scroll"><table><tr><th>Store</th><th>Cells</th><th>Capacity</th><th>Reliability</th><th>Persistence</th><th>Speed (only a latency label)</th></tr>
      ${list(NAMES, n => { const s = STORE_SPECS[n]; return `<tr><td>${s.name}</td><td>${s.size}</td><td>${s.capacity}</td><td>${s.reliability}</td><td>${s.persistence}</td><td>${s.speed}</td></tr>`; })}</table></div>
      <p>Links (strength): ${list(CONNECTIONS, ([a, b, w]) => `${a} → ${b} ${w}`, ' · ')}. A scalar v is stored as (v, 0.8v, 0.5v). A failed read
      during a transfer writes empty content (0, 0, 0) with the transferred activation, as coded. The query keeps matches
      with confidence above 0.4 and stops early if the primary store is above 0.8.</p>`,
  });

  function render() {
    for (const n of NAMES) {
      const st = ms.stores[n], cv = L.canvas('s_' + n), ctx = fitCanvas(cv), W = cv.width, H = cv.height, bw = W / st.size, bh = H * 0.72;
      ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = COLORS[NAMES.indexOf(n)];
      for (let i = 0; i < st.size; i++) { const h = bh * st.activation[i] / st.capacity; ctx.fillRect(i * bw, bh - h, Math.max(1, bw - 0.5), h); }
      for (let i = 0; i < st.size; i++) { const c = st.contents[i]; ctx.fillStyle = `rgb(${c.map(v => Math.round(v * 255)).join(',')})`; ctx.fillRect(i * bw, bh + 4, Math.max(1, bw), H - bh - 4); }
    }
    lineChart(L.canvas('chart'), NAMES.map((n, i) => ({ data: ms.meanHistory.map(h => h[n]), label: n, color: COLORS[i] })), { xLabel: 'Step', yLabel: 'mean activation', x0: 1 });
    const tr = ms.transfers.at(-1) || [];
    const rows = [['Step', `${ms.timeStep} / ${steps}`], ['Transfers this step', tr.length],
      ...NAMES.map(n => [`${n}: cells above 0.3`, ms.stores[n].activation.filter(v => v > 0.3).length])];
    if (hier && ms.timeStep >= steps) {
      for (const q of [0.8, 0.5, 0.3]) { const r = ms.query(q)[0]; rows.push([`query ${q}`, r ? `${r.source} (confidence ${fmt(r.confidence)})` : 'no match']); }
    }
    L.setReadout(rows);
    L.setStatus(ms.timeStep >= steps ? 'Finished.' : ms.timeStep ? 'Running...' : 'Press Run or Step.');
  }

  const $ = id => document.getElementById(id);
  $('wv').oninput = () => { $('wvv').textContent = $('wv').value; };
  $('wb').onclick = () => { const [ok, st] = ms.writeData(+$('wv').value, $('wp').value); $('wq').textContent = `write ${$('wv').value} → ${st}: ${ok ? 'stored' : 'failed (unreliable write)'}`; render(); };
  $('qb').onclick = () => {
    const r = ms.query(+$('wv').value);
    $('wq').innerHTML = r.length ? list(r, x => `${esc(x.source)}: content (${x.result.map(v => fmt(v)).join(', ')}) at cell ${x.index}, confidence ${fmt(x.confidence)}`, '<br>') : 'No store answered with confidence above 0.4.';
  };
});
