// Model 1 page: probabilistic computing on a grid of "puff" sites (CalciumPuffComputing) and stochastic bitstream
// arithmetic (StochasticComputing), both from probabilistic_computing.py via puff_engine.js.
import { route, esc } from './core.js';
import { NumpyRandom } from './np_random.js';
import { CalciumPuff, Stochastic, EXACT } from './puff_engine.js';
import { lab, fig } from './lab.js';
import { drawGrid, colourBar } from './heat.js';
import { lineChart, fmt } from './chart.js';

route('puffs', () => {
  let m, steps;
  const L = lab({
    kicker: 'Model 1 · probabilistic computing', title: 'Calcium-puff-inspired probabilistic computing', route: 'puffs',
    lead: `Each grid cell has a fixed "channel density" (the four quadrants are set to 0.2, 0.4, 0.6 and 0.8 plus a little
      noise). Every step a cell fires with probability equal to its density, unless its signal is already above the
      threshold; firing adds 1 to the cell, then the signal is blurred over neighbours and decays. The question the model
      asks: does the measured firing probability scale linearly with density?`,
    seed: 42, maxSpeed: 20, speed: 2,
    controls: [
      { id: 'n', label: 'Grid size', min: 20, max: 80, step: 2, value: 50, unit: 'cells', hint: 'grid_size (Python default 50)' },
      { id: 'decay', label: 'Decay factor', min: 0.5, max: 0.99, step: 0.01, value: 0.95, unit: 'per step', hint: 'decay_rate: signal is multiplied by this each step' },
      { id: 'thr', label: 'Refractory threshold', min: 0.1, max: 2, step: 0.05, value: 0.7, unit: 'a.u.', hint: 'threshold: a cell cannot fire while its signal is at or above this' },
      { id: 'ks', label: 'Diffusion kernel', type: 'select', value: 3, options: [[3, '3 x 3 cells (default)'], [5, '5 x 5 cells'], [7, '7 x 7 cells']] },
      { id: 'steps', label: 'Steps to run', min: 50, max: 1000, step: 50, value: 500, unit: 'steps', hint: 'the Python demo runs 500' },
    ],
    figures: `<div class="grid3">
        ${fig('dens', 300, 300, 'Channel density (0 to 1)', colourBar('viridis', 0, 1, 'density'))}
        ${fig('ca', 300, 300, 'Calcium-like signal (a.u.)', `<div id="cabar">${colourBar('hot', 0, 1, 'signal')}</div>`)}
        ${fig('act', 300, 300, 'Firing over the last 10 steps (fraction)', colourBar('mint', 0, 1, 'fraction'))}</div>
      ${fig('chart', 600, 260, 'Measured activation probability per density bin (dots, running average from step 11) and the least-squares line')}`,
    onReset(p) { m = new CalciumPuff(new NumpyRandom(p.seed), { gridSize: p.n, decayRate: p.decay, threshold: p.thr, kernelSize: +p.ks }); steps = p.steps; },
    step: () => m.update(), done: () => m.steps >= steps, render,
    source: 'probabilistic_computing/probabilistic_computing.py',
    after: arithmetic(),
    notes: `<h2>The rules as coded</h2>
      <p class="eq">c ← decay · c; &nbsp; fire if u &lt; density and c &lt; threshold (u uniform in [0, 1)); &nbsp; c ← blur(c + fire)</p>
      <p>The blur is a normalised Gaussian kernel exp(−d²) over 3 x 3 cells with wrap-around edges. A cell cannot fire
      again until its signal has decayed below the threshold, and the blur keeps most cells above it, so with the default
      settings the measured firing probability stays far below the density (a few percent per step) and rises only
      weakly with it: the fitted slope is much less than 1, even though the correlation is positive. A faster decay
      (decay factor 0.5) lets it follow the density much more closely (slope about 0.7 for seed 42 after 300 steps;
      about 1 if the threshold is also raised to 2).</p>`,
  });

  function render() {
    const n = m.n, mx = Math.max(1, ...m.calcium);
    drawGrid(L.canvas('dens'), m.density, n, n, { map: 'viridis' });
    drawGrid(L.canvas('ca'), m.calcium, n, n, { map: 'hot', lo: 0, hi: mx });
    document.getElementById('cabar').innerHTML = colourBar('hot', 0, fmt(mx), 'signal');
    drawGrid(L.canvas('act'), m.recentActivation(), n, n, { map: 'mint' });
    const r = m.analyze();
    const series = [{ data: r.probabilities, xs: r.densities, label: 'measured', dotsOnly: true },
      { data: [0, 1], xs: [0, 1], label: 'probability = density', color: '#556', dash: true }];
    if (r.slope !== undefined) series.push({ data: [r.intercept, r.intercept + r.slope], xs: [0, 1], label: 'linear fit', color: '#ffb03d' });
    lineChart(L.canvas('chart'), series, { xLabel: 'Channel density (bin centre)', yLabel: 'P(fire) per step', yMin: 0, yMax: 1 });
    let fired = 0; if (m.history.length) for (const v of m.history.at(-1)) fired += v;
    L.setReadout([['Step', `${m.steps} / ${steps}`], ['Cells fired this step', fired],
      ['Correlation (density, probability)', r.correlation !== undefined ? fmt(r.correlation) : '-'],
      ['Slope / intercept', r.slope !== undefined ? `${fmt(r.slope)} / ${fmt(r.intercept)}` : '-']]);
    L.setStatus(m.steps >= steps ? 'Finished.' : m.steps ? 'Running...' : 'Press Run or Step.');
  }

  wireArithmetic();
});

function arithmetic() {
  return `<h2>Stochastic arithmetic (StochasticComputing)</h2>
  <p>A number p in [0, 1] is encoded as a stream of random bits that are 1 with probability p. Logic gates on the streams
  then compute: AND multiplies, a 50/50 multiplexer averages (doubled and capped at 1 for addition). Subtraction and
  division use <em>correlated</em> streams (one shared random number per bit): a AND NOT b gives max(0, a − b), and the
  fraction of b-bits that are also a-bits gives min(1, a / b).</p>
  <div class="panel"><div class="row">
    <div><label for="sa">a <b id="sav">0.3</b></label><input type="range" id="sa" min="0" max="1" step="0.01" value="0.3"></div>
    <div><label for="sb">b <b id="sbv">0.7</b></label><input type="range" id="sb" min="0" max="1" step="0.01" value="0.7"></div>
    <div><label for="sn">Bitstream length <b id="snv">1000</b> bits</label><input type="range" id="sn" min="10" max="10000" step="10" value="1000"></div>
    <div><label for="sop">Operation</label><select id="sop"><option value="multiply">multiply (AND)</option><option value="add">add (multiplexer)</option>
      <option value="subtract">subtract (correlated AND NOT)</option><option value="divide">divide (correlated conditional)</option></select></div>
    <div><label for="ss">Seed <b id="ssv">42</b></label><input type="range" id="ss" min="0" max="999" value="42"></div>
  </div><div id="sout" aria-live="polite"></div></div>`;
}

function wireArithmetic() {
  const $ = id => document.getElementById(id);
  const show = (bits, label) => `<div class="small mute">${label}</div><div class="bits">${Array.from(bits.slice(0, 160), b => `<span class="${b ? 'one' : 'zero'}">${b}</span>`).join('')}${bits.length > 160 ? ' …' : ''}</div>`;
  const update = () => {
    const a = +$('sa').value, b = +$('sb').value, n = +$('sn').value, op = $('sop').value, seed = +$('ss').value;
    ['sa', 'sb', 'sn', 'ss'].forEach(k => { $(k + 'v').textContent = $(k).value; });
    const r = new Stochastic(new NumpyRandom(seed))[op](a, b, n), exact = EXACT[op](a, b);
    const s = r.streams;
    $('sout').innerHTML = `<div class="op"><span>exact</span><b>${fmt(exact)}</b><span>stochastic</span><b>${fmt(r.value)}</b>
      <span>error</span><b>${fmt(Math.abs(exact - r.value))}</b></div>
      ${s ? show(s.a, 'stream a') + show(s.b, 'stream b') + (s.sel ? show(s.sel, 'selector (p = 0.5)') : '') + show(s.out, 'result stream') : '<p class="small mute">b &lt; 0.01: returns 1.0 without streams (as coded).</p>'}
      <p class="small mute">Same numbers as <code>np.random.seed(${esc(seed)}); sc.stochastic_${esc(op)}(${a}, ${b}, iterations=${n})</code> on an existing <code>sc = StochasticComputing(model)</code>.
      The error shrinks roughly as 1/√length.</p>`;
  };
  ['sa', 'sb', 'sn', 'sop', 'ss'].forEach(k => $(k).addEventListener('input', update));
  update();
}
