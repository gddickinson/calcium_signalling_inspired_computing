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
  const names = Object.keys(r.radii);
  host.innerHTML = `<h2>Pattern recognition (Python run)</h2>
  <p><code>PatternRecognitionComputer</code> stores three patterns on a ${r.grid} x ${r.grid} grid (a 10 x 10 pulse in
  channel 0, a horizontal stripe in channel 1, a vertical stripe in channel 2). A pattern's signature is the first
  ${r.frames} frames of the model's response, averaged over ${r.block} x ${r.block} cell blocks per channel with each
  channel's mean removed. Each stored pattern gets an acceptance radius of ${r.radius_factor} x the largest distance of
  ${r.n_noisy} noisy training copies. An input is matched to the nearest stored signature if it lies within that radius,
  and rejected otherwise. Every input below has Gaussian noise (sd ${r.noise_sd}) added. Computed in Python when the site was built (seed ${r.seed}).</p>
  <div class="scroll"><table><tr><th>Input (+ noise)</th><th>Stored?</th><th>Result</th><th>Score</th>${list(names, g => `<th>distance to ${esc(g)} (radius ${fmt(r.radii[g])})</th>`)}</tr>
  ${list(r.tests, t => `<tr><td>${esc(t.input)}</td><td>${esc(t.kind)}</td><td class="${t.correct ? 'ok' : 'err'}">${esc(t.match ?? 'rejected')}</td><td>${fmt(t.score)}</td>${list(names, g => `<td>${fmt(t.distances[g])}</td>`)}</tr>`)}</table></div>
  <p class="callout a">${r.correct} of ${r.tests.length} inputs handled correctly. Score = radius / (radius + distance) for
  the nearest pattern, so above 0.5 means inside its radius. Only the first frames are used because the model is
  dissipative: later frames lose the pattern and are dominated by amplified noise. (The earlier version compared k-means
  transition graphs whose cluster labels meant different things in each run, and matched 1 of 3.)</p>`;
}
