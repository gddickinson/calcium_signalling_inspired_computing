// Model 6 page: wave-based processing (WaveComputingMedium and the WaveComputing set-ups via wave_engine.js).
import { route } from './core.js';
import { WaveMedium, gateMedium, gateReadout, GATE, memoryMedium, addRetrievalPulses, memoryReadout, filterMedium, filterReadout } from './wave_engine.js';
import { lab, fig } from './lab.js';
import { drawGrid, colourBar } from './heat.js';
import { lineChart, fmt } from './chart.js';

const SETUPS = [['demo', 'Demo medium: two opposite-phase sources, obstacle, resonant box (100 steps)'],
  ['and', 'Logic gate: AND (100 steps)'], ['or', 'Logic gate: OR (100 steps)'], ['xor', 'Logic gate: XOR (100 steps)'],
  ['memory', 'Wave memory: store 6 values, then retrieval pulses (30 + 20 steps)'], ['filter', 'Wave filter: low-pass a two-tone signal (100 steps)']];
const MEM = [0.8, -0.5, 0.3, 0.0, -0.2, 0.6];
const SIGNAL = Array.from({ length: 50 }, (_, i) => { const t = 2 * Math.PI * i / 49; return 0.5 * Math.sin(t) + 0.3 * Math.sin(5 * t); });

route('waves', () => {
  let m, setup, phaseSteps, ts, total, done, view, gateRef = 0, bits = [true, true];
  const L = lab({
    kicker: 'Model 6 · wave-based processing', title: 'Sources, obstacles and resonance in a damped medium', route: 'waves',
    lead: `A 2-D medium obeying the damped wave equation. Point sources drive it with sinusoids (amplitude, frequency,
      phase); obstacles hold it at zero; resonant boxes stop losing energy while their dominant frequency (FFT of the box
      mean over the last 50 frames) matches their tuning; memory elements pin a slowly decaying value. The WaveComputing
      set-ups use these parts as logic gates, a memory and a filter.`,
    maxSpeed: 10, speed: 2,
    controls: [
      { id: 'setup', label: 'Set-up', type: 'select', value: 'demo', options: SETUPS },
      { id: 'n', label: 'Grid size', min: 40, max: 100, step: 10, value: 100, unit: 'cells', hint: 'size=(100, 100) in the demo' },
      { id: 'c', label: 'Wave speed c', min: 1, max: 40, step: 0.5, value: 10, unit: 'cells / time unit', hint: 'wave_speed (default 10); the internal time step adapts to keep c·dt within the CFL limit' },
      { id: 'damp', label: 'Damping', min: 0.9, max: 1, step: 0.001, value: 0.99, unit: 'velocity kept per step' },
      { id: 'nl', label: 'Cubic nonlinearity', min: 0, max: 0.5, step: 0.01, value: 0.1, hint: 'restoring term −nonlinearity·u³' },
      { id: 'dt', label: 'Output time step', min: 0.02, max: 0.5, step: 0.01, value: 0.1, unit: 'time units', hint: 'time_step: one frame; source phase advances 2π·f·dt per frame' },
      { id: 'ga', label: 'Gate input A (left line)', type: 'check', value: true },
      { id: 'gb', label: 'Gate input B (top line)', type: 'check', value: true },
      { id: 'bnd', label: 'Boundary', type: 'select', value: 'absorbing', options: [['absorbing', 'absorbing (5-cell sponge)'], ['reflecting', 'reflecting (no change)'], ['periodic', 'periodic (as coded: adds half the opposite edge)']] },
      { id: 'view', label: 'Show', type: 'select', value: 'amp', live: true, options: [['amp', 'amplitude'], ['freq', 'local frequency (sign changes)'], ['phase', 'local phase estimate']] },
    ],
    figures: `<div class="grid2">${fig('field', 400, 400, 'Field (yellow dots: sources, grey: obstacles, green boxes: resonant regions)', '<div id="fbar"></div>')}
        ${fig('chart', 600, 600, 'Total energy Σu² per step')}</div>${fig('chart2', 600, 220, 'Set-up readout')}`,
    onReset(p) {
      const base = new WaveMedium({ width: p.n, height: p.n, boundary: p.bnd, damping: p.damp, waveSpeed: p.c, nonlinearity: p.nl, timeStep: p.dt });
      setup = p.setup; ts = 0; done = false; view = p.view;
      if (setup === 'demo') {
        m = base; const s = p.n / 100;
        m.addSource([Math.round(20 * s), Math.round(50 * s)], 0.5, 0.8); m.addSource([Math.round(80 * s), Math.round(50 * s)], 0.5, 0.8, Math.PI);
        m.addObstacle(Math.round(45 * s), Math.round(55 * s), Math.round(40 * s), Math.round(60 * s));
        m.addResonantRegion(Math.round(70 * s), Math.round(90 * s), Math.round(70 * s), Math.round(90 * s), 0.8);
        phaseSteps = [100];
      } else if (['and', 'or', 'xor'].includes(setup)) {
        bits = [p.ga, p.gb]; m = gateMedium(base, bits, setup); phaseSteps = [100];
        const ref = gateMedium(base, [true, false], 'or'); ref.run(100); gateRef = gateReadout(ref);   // one input alone
      }
      else if (setup === 'memory') { m = memoryMedium(base, MEM); phaseSteps = [30, 20]; }
      else { m = filterMedium(base, SIGNAL, 'lowpass', 0.3); phaseSteps = [100]; }
      total = phaseSteps.reduce((a, b) => a + b, 0);
    },
    onLive(p) { view = p.view; },
    step() {
      m.update(ts * m.timeStep); ts++;
      if (setup === 'memory' && m.steps === phaseSteps[0]) { addRetrievalPulses(m); ts = 0; }   // run_simulation restarts t
      if (m.steps >= total) done = true;
    },
    done: () => done, render,
    source: 'wave_based_computing/wave_medium.py and wave_computing.py',
    notes: `<h2>The model as coded</h2>
      <p class="eq">u_tt = c²·∇²u − γ·u_t − nonlinearity·u³</p>
      <p>Leapfrog in time with a 9-point Laplacian. Each frame (output time step) is split into substeps short enough
      that c·dt/dx stays within 90% of the stencil's stability (CFL) limit, so any wave speed is stable; "damping" is the
      fraction of wave velocity kept per frame. Sources set the field in their 3 x 3 footprint; the absorbing edge is a
      5-cell sponge. With c = 10 cells per time unit, a source at 0.5 cycles per time unit makes waves 20 cells long.</p>
      <p><b>Gates.</b> Each input bit switches a line of sources (A along the left, B along the top), equidistant from the
      centre. The output compares the wave amplitude at the centre with what one input alone gives there: AND needs more
      than 1.5 times that (two waves in phase add), OR more than half, and XOR uses B in antiphase so two active inputs
      cancel at the centre. On the 100 x 100 grid the centre amplitudes are 0, 0.365, 0.365 and 0.734 (0 for XOR with both on).</p>
      <p>Frequencies are in cycles per time unit (t = frame·dt). Local frequency is sign changes over the last 10 frames
      / (10·dt) / 2; phase is the position of the current value within the last 10 frames' range, times 2π. The physics
      was corrected in this branch: the earlier update overwrote its previous state before using it, so it was a
      first-order smoothing in which waves never left the sources.</p>`,
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
    const rows = [['Step', `${m.steps} / ${total} (t = ${fmt(m.steps * m.timeStep)})`], ['Sources', m.sources.length], ['Substeps per step (CFL)', `${m.substeps} (c·dt/dx = ${fmt(m.waveSpeed * m.dtSub)})`], ['Max |u|', fmt(Math.max(...m.current.map(Math.abs)))]];
    if (['and', 'or', 'xor'].includes(setup)) {
      const g = m.history.length ? gateReadout(m) : 0, thr = GATE.thresholds[setup] * gateRef;
      rows.push([`Inputs A, B`, bits.map(Number).join(', ')], ['Centre amplitude (3 x 3, last 20 frames)', fmt(g)],
        [`Threshold (${GATE.thresholds[setup]} x one-input amplitude ${fmt(gateRef)})`, fmt(thr)],
        [`${setup.toUpperCase()} output`, m.steps >= total ? String(+(g > thr)) : '(running)']);
      lineChart(L.canvas('chart2'), [{ data: centre(m), label: 'centre |mean of 3 x 3|' }], { xLabel: 'Frame (last 100)', yLabel: 'centre amplitude', yMin: 0, hlines: [{ y: thr, label: 'threshold' }] });
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

// |mean of the 3 x 3 centre cells| for every stored frame (the gate readout takes the max of the last 20)
function centre(m) {
  const W = m.width, cx = Math.floor(W / 2), cy = Math.floor(m.height / 2);
  return m.history.map(f => { let s = 0; for (let y = cy - 1; y < cy + 2; y++) for (let x = cx - 1; x < cx + 2; x++) s += f[y * W + x]; return Math.abs(s / 9); });
}
