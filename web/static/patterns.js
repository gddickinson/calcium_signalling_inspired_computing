// Model 2 page: spatiotemporal computing (SpatiotemporalComputing via st_engine.js) and the Python-only pattern
// recognition run (PatternRecognitionComputer, precomputed through /api/recognition).
import { route, api, esc, list } from './core.js';
import { NumpyRandom } from './np_random.js';
import { Spatiotemporal } from './st_engine.js';
import { lab, fig } from './lab.js';
import { drawGrid, colourBar, CATS } from './heat.js';
import { lineChart, fmt } from './chart.js';

route('patterns', () => {
  let m, steps, means;
  const L = lab({
    kicker: 'Model 2 · spatiotemporal computing', title: 'Reaction-diffusion-style pattern dynamics', route: 'patterns',
    lead: `Three fields ("channels", which the code compares to calcium, IP3 and a slow species) start from a square, a bar
      and a gradient. Each step every field is blurred by its own Gaussian kernel (fast, medium, slow spread), the fields
      act on each other through a fixed 3 x 3 interaction matrix, five random circular regions add a constant drive to one
      channel each, and everything is squashed with tanh into (−1, 1).`,
    seed: 42, maxSpeed: 10, speed: 1,
    controls: [
      { id: 'n', label: 'Grid size', min: 30, max: 100, step: 2, value: 100, unit: 'cells', hint: 'grid_size (Python default 100; the initial shapes sit at fixed cells 10-90)' },
      { id: 'd0', label: 'Channel 0 kernel spread', min: 0.02, max: 2, step: 0.01, value: 0.6, unit: 'cells²', hint: 'diffusion_rates[0]: kernel exp(−d²/rate) on 5 x 5 cells' },
      { id: 'd1', label: 'Channel 1 kernel spread', min: 0.02, max: 2, step: 0.01, value: 0.2, unit: 'cells²' },
      { id: 'd2', label: 'Channel 2 kernel spread', min: 0.02, max: 2, step: 0.01, value: 0.05, unit: 'cells²' },
      { id: 'steps', label: 'Steps to run', min: 10, max: 300, step: 10, value: 100, unit: 'steps', hint: 'the Python demo runs 100' },
    ],
    figures: `<div class="grid3">
        ${fig('c0', 300, 300, 'Channel 0 (fast spread)')}${fig('c1', 300, 300, 'Channel 1 (medium)')}${fig('c2', 300, 300, 'Channel 2 (slow)')}</div>
      ${colourBar('diverge', '−1', '1', 'state (tanh range)')}
      <div class="grid2" style="margin-top:12px">${fig('reg', 300, 300, 'Spatial regions: drive +0.1 per step to channel 0, 1 or 2', `<div class="legend">${[1, 2, 3].map(k => `<span><i style="background:rgb(${CATS[k]})"></i>channel ${k - 1}</span>`).join('')}</div>`)}
        ${fig('chart', 400, 300, 'Mean of each channel over time')}</div>`,
    onReset(p) {
      m = new Spatiotemporal(new NumpyRandom(p.seed), { gridSize: p.n, diffusionRates: [p.d0, p.d1, p.d2] });
      steps = p.steps; means = [[], [], []];
    },
    step: () => { m.update(); m.stats().forEach((s, i) => means[i].push(s.mean)); },
    done: () => m.steps >= steps, render,
    source: 'spatiotemporal_computing/spatiotemporal_computing.py',
    after: '<div id="recog"></div>',
    notes: `<h2>The rules as coded</h2>
      <p class="eq">s_i ← tanh( blur_i(s_i) + Σ_j R[j][i]·s_j + 0.1·[region = i+1] )</p>
      <p>R = [[0, 0.2, −0.1], [0.3, −0.1, 0.2], [−0.2, 0.1, 0]] (row j: effect of channel j). The blur uses wrap-around
      edges; the "spread" parameter is the denominator in exp(−d²/rate), so 0.05 leaves a cell almost unchanged and 0.6
      shares it with its neighbours. Nothing here is a chemical rate: the numbers are dimensionless.</p>`,
  });

  function render() {
    const n = m.n;
    ['c0', 'c1', 'c2'].forEach((id, i) => drawGrid(L.canvas(id), m.state[i], n, n, { map: 'diverge', lo: -1, hi: 1 }));
    drawGrid(L.canvas('reg'), m.regions, n, n, { cats: true });
    lineChart(L.canvas('chart'), means.map((d, i) => ({ data: d, label: `channel ${i}` })), { xLabel: 'Step', yLabel: 'mean state', x0: 1 });
    const c = m.history.length ? m.correlations() : null, st = m.stats();
    L.setReadout([['Step', `${m.steps} / ${steps}`],
      ...st.map((s, i) => [`Channel ${i} mean (min, max)`, `${fmt(s.mean)} (${fmt(s.min)}, ${fmt(s.max)})`]),
      ['Correlation 0-1 / 0-2 / 1-2', c ? `${fmt(c[0][1])} / ${fmt(c[0][2])} / ${fmt(c[1][2])}` : '-']]);
    L.setStatus(m.steps >= steps ? 'Finished.' : m.steps ? 'Running...' : 'Press Run or Step.');
  }

  recognition();
});

async function recognition() {
  const r = await api('/recognition'), host = document.getElementById('recog');
  if (!host) return;
  host.innerHTML = `<h2>Pattern recognition (Python run)</h2>
  <p><code>PatternRecognitionComputer</code> trains on three patterns (a 10 x 10 pulse in channel 0, a horizontal stripe in
  channel 1, a vertical stripe in channel 2) on a ${r.grid} x ${r.grid} grid: it runs each for ${r.steps} steps, reduces the
  frames with PCA, clusters them into 5 states with k-means, and keeps the graph of transitions between cluster labels.
  A noisy copy of each pattern (Gaussian noise, sd ${r.noise_sd}) is then matched by the overlap of transition edges
  (Jaccard similarity). It uses scikit-learn, so this is the real Python result, computed when the site was built
  (seed ${r.seed}).</p>
  <div class="scroll"><table><tr><th>Input (noisy)</th><th>Best match</th>${list(Object.keys(r.graphs), g => `<th>vs ${esc(g)}</th>`)}</tr>
  ${list(r.tests, t => `<tr><td>${esc(t.input)}</td><td class="${t.input === t.match ? 'ok' : 'err'}">${esc(t.match)}</td>${list(Object.keys(r.graphs), g => `<td>${fmt(t.scores[g])}</td>`)}</tr>`)}</table></div>
  <p class="callout">${r.correct} of ${r.tests.length} inputs matched their own pattern in this run. The k-means cluster
  numbers are assigned independently in every run, so the same label in two graphs need not mean the same state; that
  limits what the edge overlap can show (noted for the author in the repository's SESSION_LOG.md).</p>`;
}
