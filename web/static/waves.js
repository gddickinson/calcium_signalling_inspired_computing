// Model 6 page: wave-based processing (WaveComputingMedium and the WaveComputing set-ups via wave_engine.js).
import { route } from './core.js';
import { WaveMedium, logicGateMedium, gateReadout, memoryMedium, addRetrievalPulses, memoryReadout, filterMedium, filterReadout } from './wave_engine.js';
import { lab, fig } from './lab.js';
import { drawGrid, colourBar } from './heat.js';
import { lineChart, fmt } from './chart.js';

const SETUPS = [['demo', 'Demo medium: two opposite-phase sources, obstacle, resonant box (100 steps)'],
  ['and', 'Logic gate set-up: AND (50 steps)'], ['or', 'Logic gate set-up: OR (50 steps)'], ['xor', 'Logic gate set-up: XOR (50 steps)'],
  ['memory', 'Wave memory: store 6 values, then retrieval pulses (30 + 20 steps)'], ['filter', 'Wave filter: low-pass a two-tone signal (100 steps)']];
const MEM = [0.8, -0.5, 0.3, 0.0, -0.2, 0.6];
const SIGNAL = Array.from({ length: 50 }, (_, i) => { const t = 2 * Math.PI * i / 49; return 0.5 * Math.sin(t) + 0.3 * Math.sin(5 * t); });

route('waves', () => {
  let m, setup, phaseSteps, ts, total, done, view;
  const L = lab({
    kicker: 'Model 6 · wave-based processing', title: 'Sources, obstacles and resonance in a damped medium', route: 'waves',
    lead: `Point sources inject sinusoids (amplitude, frequency, phase) into a 2-D field; obstacles hold it at zero; resonant
      boxes amplify the field by up to 10% per step when the box's dominant frequency (FFT of its mean over the last 50
      frames) matches their tuning; memory elements add a slowly decaying constant. The WaveComputing set-ups place these
      parts to act as logic gates, a memory and a filter.`,
    maxSpeed: 10, speed: 2,
    controls: [
      { id: 'setup', label: 'Set-up', type: 'select', value: 'demo', options: SETUPS },
      { id: 'n', label: 'Grid size', min: 40, max: 100, step: 10, value: 100, unit: 'cells', hint: 'size=(100, 100) in the demo' },
      { id: 'D', label: 'Diffusion D', min: 0, max: 2, step: 0.05, value: 0.2, unit: 'cells² / time²', hint: 'used as D·dt²·(neighbour mean); above D·dt² ≈ 0.01 the field grows' },
      { id: 'damp', label: 'Damping factor', min: 0.9, max: 1, step: 0.001, value: 0.99, unit: 'per step' },
      { id: 'nl', label: 'Cubic nonlinearity', min: 0, max: 0.5, step: 0.01, value: 0.1, hint: 'subtracts nonlinearity·u³ each step' },
      { id: 'dt', label: 'Time step dt', min: 0.02, max: 0.5, step: 0.01, value: 0.1, unit: 'time units', hint: 'source phase advances 2π·f·dt per step' },
      { id: 'bnd', label: 'Boundary', type: 'select', value: 'absorbing', options: [['absorbing', 'absorbing (5-cell taper)'], ['reflecting', 'reflecting (no change)'], ['periodic', 'periodic (as coded: adds half the opposite edge)']] },
      { id: 'view', label: 'Show', type: 'select', value: 'amp', live: true, options: [['amp', 'amplitude'], ['freq', 'local frequency (sign changes)'], ['phase', 'local phase estimate']] },
    ],
    figures: `<div class="grid2">${fig('field', 400, 400, 'Field (yellow dots: sources, grey: obstacles, green boxes: resonant regions)', '<div id="fbar"></div>')}
        ${fig('chart', 600, 600, 'Total energy Σu² per step')}</div>${fig('chart2', 600, 220, 'Set-up readout')}`,
    onReset(p) {
      const base = new WaveMedium({ width: p.n, height: p.n, boundary: p.bnd, damping: p.damp, diffusion: p.D, nonlinearity: p.nl, timeStep: p.dt });
      setup = p.setup; ts = 0; done = false; view = p.view;
      if (setup === 'demo') {
        m = base; const s = p.n / 100;
        m.addSource([Math.round(20 * s), Math.round(50 * s)], 0.5, 0.8); m.addSource([Math.round(80 * s), Math.round(50 * s)], 0.5, 0.8, Math.PI);
        m.addObstacle(Math.round(45 * s), Math.round(55 * s), Math.round(40 * s), Math.round(60 * s));
        m.addResonantRegion(Math.round(70 * s), Math.round(90 * s), Math.round(70 * s), Math.round(90 * s), 0.8);
        phaseSteps = [100];
      } else if (['and', 'or', 'xor'].includes(setup)) { m = logicGateMedium(base, setup); phaseSteps = [50]; }
      else if (setup === 'memory') { m = memoryMedium(base, MEM); phaseSteps = [30, 20]; }
      else { m = filterMedium(base, SIGNAL, 'lowpass', 0.3); phaseSteps = [100]; }
      m.timeStep = p.dt; total = phaseSteps.reduce((a, b) => a + b, 0);
    },
    onLive(p) { view = p.view; },
    step() {
      m.update(ts * m.timeStep); ts++;
      if (setup === 'memory' && m.steps === phaseSteps[0]) { addRetrievalPulses(m); ts = 0; }   // run_simulation restarts t
      if (m.steps >= total) done = true;
    },
    done: () => done, render,
    source: 'wave_based_computing/wave_medium.py and wave_computing.py',
    notes: `<h2>The update as coded, and a caveat</h2>
      <p class="eq">u ← damping · (u + D·dt²·mean₈(u)) · taper − nonlinearity·u³, then resonance gain, sources, memory</p>
      <p>The Python is written like a leapfrog wave-equation step (2u − u_prev + D·dt²·∇²u), but it sets u_prev = u before
      using it, and its "Laplacian" is the weighted mean of the 8 neighbours. So each step is a first-order damped
      smoothing of a driven field, not a second-order wave: with the defaults (D·dt² = 0.002) activity stays within a
      cell or two of the sources and nothing reaches the centre of the logic-gate set-ups (their readout is essentially zero, below 10⁻⁹,
      and the input patterns passed to <code>wave_logic_gates</code> are not used). Turning it into a true wave
      equation also needs the source and cubic terms rescaled, so that choice is left to the author; this page runs the
      model exactly as it is in the repository.</p>
      <p>Frequencies are in cycles per time unit (t = step·dt). Local frequency is sign changes over the last 10 frames
      / (10·dt) / 2; phase is the position of the current value within the last 10 frames' range, times 2π.</p>`,
  });

  function render() {
    const n = m.width;
    let vals, opt, bar;
    if (view === 'freq') { vals = m.frequency; opt = { map: 'viridis', lo: 0, hi: 2 }; bar = colourBar('viridis', 0, 2, 'cycles / time unit'); }
    else if (view === 'phase') { vals = m.phase; opt = { map: 'mint', lo: 0, hi: 2 * Math.PI }; bar = colourBar('mint', 0, '2π', 'phase'); }
    else { let a = 1e-12; for (const v of m.current) a = Math.max(a, Math.abs(v)); opt = { map: 'diverge', lo: -a, hi: a }; vals = m.current; bar = colourBar('diverge', fmt(-a), fmt(a), 'amplitude (scaled to max |u|)'); }
    drawGrid(L.canvas('field'), vals, n, n, { ...opt, overlay: (ctx, sx, sy) => {
      ctx.fillStyle = 'rgba(140,150,170,.75)'; m.obstacles.forEach((o, k) => { if (o) ctx.fillRect((k % n) * sx, Math.floor(k / n) * sy, sx, sy); });
      ctx.strokeStyle = '#3dffd8'; ctx.lineWidth = 1.5;
      if (m.resonant.length < 30) for (const r of m.resonant) ctx.strokeRect(r.x0 * sx, r.y0 * sy, (r.x1 - r.x0) * sx, (r.y1 - r.y0) * sy);
      ctx.fillStyle = '#ffb03d'; for (const s of m.sources) { ctx.beginPath(); ctx.arc((s.x + 0.5) * sx, (s.y + 0.5) * sy, Math.max(2, sx * 0.6), 0, 7); ctx.fill(); }
    } });
    document.getElementById('fbar').innerHTML = bar;
    lineChart(L.canvas('chart'), [{ data: m.energy, label: 'energy' }], { xLabel: 'Step (last 100)', yLabel: 'Σ u²', x0: Math.max(1, m.steps - m.energy.length + 1) });
    const rows = [['Step', `${m.steps} / ${total}`], ['Sources', m.sources.length], ['D·dt²', fmt(m.diffusion * m.timeStep ** 2)], ['Max |u|', fmt(Math.max(...m.current.map(Math.abs)))]];
    if (['and', 'or', 'xor'].includes(setup)) {
      const g = m.history.length ? gateReadout(m) : 0;
      rows.push([`${setup.toUpperCase()} readout (max |u| in centre 20 x 20)`, fmt(g)]);
      lineChart(L.canvas('chart2'), [{ data: hist(m), label: 'centre max |u|' }], { xLabel: 'Frame (last 100)', yLabel: 'max |u| in centre' });
    } else if (setup === 'memory') {
      const r = memoryReadout(m, MEM.length);
      rows.push(['Retrieved', r.map(fmt).join(', ')]);
      lineChart(L.canvas('chart2'), [{ data: MEM, xs: MEM.map((_, i) => i + 1), label: 'stored', dots: true }, { data: r, xs: MEM.map((_, i) => i + 1), label: 'retrieved', dots: true }], { xLabel: 'Memory slot', yLabel: 'value' });
    } else if (setup === 'filter') {
      const out = m.history.length ? filterReadout(m, SIGNAL.length) : [];
      lineChart(L.canvas('chart2'), [{ data: SIGNAL, label: 'input signal' }, { data: out, label: 'output (scaled to max 1)' }], { xLabel: 'Sample (row)', yLabel: 'value' });
    } else {
      lineChart(L.canvas('chart2'), [{ data: Array.from({ length: n }, (_, x) => m.current[Math.floor(n / 2) * n + x]), label: 'u along the middle row' }], { xLabel: 'x (cells)', yLabel: 'u' });
    }
    L.setReadout(rows);
    L.setStatus(done ? 'Finished.' : m.steps ? 'Running...' : 'Press Run or Step.');
  }
});

// Centre-box max |u| for every stored frame (the gate readout takes the max of the last 10)
function hist(m) {
  const W = m.width, H = m.height, cx = Math.floor(W / 2), cy = Math.floor(H / 2);
  return m.history.map(f => { let b = 0; for (let y = Math.max(0, cy - 10); y < Math.min(H, cy + 10); y++) for (let x = Math.max(0, cx - 10); x < Math.min(W, cx + 10); x++) b = Math.max(b, Math.abs(f[y * W + x])); return b; });
}
