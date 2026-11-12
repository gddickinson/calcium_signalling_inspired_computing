// Model 3 page: analog computing with threshold units (ThresholdComputingNetwork + ThresholdLogicComputer via
// threshold_engine.js). Each experiment is a list of phases run step by step, equivalent to the Python methods.
import { route } from './core.js';
import { NumpyRandom } from './np_random.js';
import { ThresholdNetwork } from './threshold_engine.js';
import { lab, fig } from './lab.js';
import { drawGrid, colourBar } from './heat.js';
import { lineChart, fmt } from './chart.js';

const EXPERIMENTS = [
  ['free', 'Free run: random input once, then 30 steps (demo)'],
  ['integration', 'Temporal integration: 20 weak inputs, 40 steps'],
  ['detection', 'Threshold detection: strengths 0.1 to 1.0'],
  ['memory', 'Analog memory: store 0.2, 0.5, 0.8 for 5 steps'],
];
const PY = {
  free: 'p = np.random.random(len(net.input_region)) * 0.8; net.set_input(p); net.run_simulation(steps=30)',
  integration: 'weak = [np.random.random(len(net.input_region)) * 0.3 for _ in range(20)]; logic.temporal_integration(weak, duration=40)',
  detection: 'logic.threshold_detection(np.linspace(0.1, 1.0, 10))',
  memory: 'logic.analog_memory([0.2, 0.5, 0.8])',
};

// phases: [{ label, reset, steps, inputs(step) -> pattern | null, value }]
function plan(exp, net, rng) {
  const n = net.input.length, fill = v => new Array(n).fill(v);
  if (exp === 'free') {
    const p = Array.from({ length: n }, () => rng.random() * 0.8);
    return [{ label: 'free run', steps: 30, inputs: s => (s === 0 ? p : null), once: true }];
  }
  if (exp === 'integration') {
    const weak = Array.from({ length: 20 }, () => Array.from({ length: n }, () => rng.random() * 0.3));
    return [{ label: 'weak inputs then silence', steps: 40, inputs: s => (s < 20 ? weak[s] : fill(0)) }];
  }
  if (exp === 'detection') {
    return Array.from({ length: 10 }, (_, k) => { const v = 0.1 + 0.1 * k; return { label: `strength ${fmt(v)}`, value: v, reset: true, steps: 20, inputs: () => fill(v) }; });
  }
  return [0.2, 0.5, 0.8].map(v => ({ label: `stored value ${v}`, value: v, reset: true, steps: 30, inputs: s => (s < 5 ? fill(v) : fill(0)) }));
}

route('threshold', () => {
  let net, phases, pi, si, traces, exp, total, done;
  const L = lab({
    kicker: 'Model 3 · analog threshold computing', title: 'Threshold units with refractory periods', route: 'threshold',
    lead: `A grid of leaky units, each with a sigmoid output and a threshold near 0.7 (randomised per unit). Inputs are
      clamped onto the top two rows; the bottom two rows are the output. Each unit integrates the weighted output of its
      random neighbours, leaks 5% per step, and after crossing threshold becomes harder to drive for a few steps
      (refractory period). Experiments from <code>ThresholdLogicComputer</code> probe integration, detection and memory.`,
    seed: 42, maxSpeed: 10, speed: 1,
    controls: [
      { id: 'exp', label: 'Experiment', type: 'select', value: 'free', options: EXPERIMENTS },
      { id: 'H', label: 'Rows', min: 5, max: 20, value: 10, unit: 'units' },
      { id: 'W', label: 'Columns', min: 4, max: 20, value: 10, unit: 'units' },
      { id: 'radius', label: 'Connection radius', min: 1, max: 4, value: 3, unit: 'cells', hint: 'connectivity_radius (demo 3)' },
      { id: 'prob', label: 'Connection probability', min: 0.05, max: 1, step: 0.05, value: 0.3, hint: 'per pair within the radius; weight 0.5 + 0.5·N(0,1)' },
    ],
    figures: `<div class="grid2">${fig('act', 300, 300, 'Unit activation (sigmoid output); blue box: input rows, green box: output rows', colourBar('hot', 0, 1, 'activation'))}
        ${fig('pot', 300, 300, 'Membrane-potential-like state (clipped to 0-1 for colour)', colourBar('viridis', 0, 1, 'potential'))}</div>
      ${fig('chart', 600, 230, 'Largest output-row activation per step (dashed: 0.5 detection level)')}
      ${fig('chart2', 600, 230, 'Experiment summary')}`,
    onReset(p) {
      const rng = new NumpyRandom(p.seed);
      net = new ThresholdNetwork(rng, { height: p.H, width: p.W, radius: p.radius, prob: p.prob });
      exp = p.exp; phases = plan(exp, net, rng); pi = 0; si = 0; traces = phases.map(() => []); done = false;
      total = phases.reduce((s, ph) => s + ph.steps, 0);
      document.getElementById('pycode').textContent = `np.random.seed(${p.seed}); net = ThresholdComputingNetwork(size=(${p.H}, ${p.W}), connectivity_radius=${p.radius}, connectivity_prob=${p.prob}); logic = ThresholdLogicComputer(net); ${PY[exp]}`;
    },
    step() {
      const ph = phases[pi];
      if (si === 0 && ph.reset) net.reset();
      const inp = ph.inputs(si);
      if (inp) net.setInput(inp);
      net.update();
      traces[pi].push(Math.max(...net.getOutput()));
      if (++si >= ph.steps) { si = 0; pi++; if (pi >= phases.length) done = true; }
    },
    done: () => done, render,
    source: 'threshold_computing/threshold_computing.py and threshold_logic.py',
    notes: `<p class="small mute">Equivalent Python: <code id="pycode"></code></p>
      <h2>The unit as coded</h2>
      <p class="eq">v ← 0.95·(v + k·input + 0.01·m); &nbsp; out = 1 / (1 + e^(−slope·(v − θ_eff))); &nbsp; m ← 0.99·m + 0.01·out</p>
      <p>k = 0.1 (integration rate), reduced by up to 80% and θ_eff raised by up to 0.2 while refractory (5 steps after the
      potential crosses θ_eff). Thresholds are 0.7 + 0.1·N(0, 1) (at least 0.4) and slopes 10 + N(0, 1) (at least 5).
      Input cells have their potential <em>set</em> to the input value each step it is applied. Detection and memory
      trials each start from rest (fixed in this branch; before, every trial inherited the previous trial's state).</p>`,
  });

  function render() {
    const { H, W } = net, k = net.units.length;
    const act = Float64Array.from(net.units, u => u.activation), pot = Float64Array.from(net.units, u => Math.min(1, Math.max(0, u.potential)));
    const box = (ctx, sx, sy) => {
      ctx.lineWidth = 3; ctx.strokeStyle = '#7b8cff'; ctx.strokeRect(1, 1, W * sx - 2, 2 * sy - 2);
      ctx.strokeStyle = '#3dffd8'; ctx.strokeRect(1, (H - 2) * sy + 1, W * sx - 2, 2 * sy - 2);
    };
    drawGrid(L.canvas('act'), act, H, W, { map: 'hot', overlay: box });
    drawGrid(L.canvas('pot'), pot, H, W, { map: 'viridis', overlay: box });
    const flat = traces.flat();
    lineChart(L.canvas('chart'), [{ data: flat, label: 'max output' }], { xLabel: 'Step', yLabel: 'activation', yMin: 0, yMax: 1, x0: 1, hlines: [{ y: 0.5, label: 'detection 0.5' }] });
    if (exp === 'detection') {
      const xs = phases.map(p => p.value).slice(0, traces.filter(t => t.length).length), ys = traces.filter(t => t.length).map(t => Math.max(...t));
      lineChart(L.canvas('chart2'), [{ data: ys, xs, label: 'max output', dots: true }], { xLabel: 'Signal strength', yLabel: 'max output', yMin: 0, yMax: 1, hlines: [{ y: 0.5, label: 'detected above' }] });
    } else if (exp === 'memory') {
      lineChart(L.canvas('chart2'), traces.map((t, i) => ({ data: t, label: `input ${phases[i].value}` })), { xLabel: 'Step within trial (input removed after step 5)', yLabel: 'max output', yMin: 0, yMax: 1, x0: 1 });
    } else {
      const meanAct = net.activationHistory.map(a => a.reduce((x, y) => x + y, 0) / k);
      lineChart(L.canvas('chart2'), [{ data: meanAct, label: 'mean activation (all units)' }], { xLabel: 'Step', yLabel: 'mean activation', yMin: 0, x0: 1 });
    }
    const ph = phases[Math.min(pi, phases.length - 1)], step = traces.reduce((s, t) => s + t.length, 0);
    L.setReadout([['Step', `${step} / ${total}`], ['Phase', done ? 'finished' : ph.label], ['Connections', net.nConnections],
      ['Max output now', fmt(Math.max(...net.getOutput()))], ['Refractory units', net.units.filter(u => u.countdown > 0).length]]);
    L.setStatus(done ? 'Finished.' : step ? 'Running...' : 'Press Run or Step.');
  }
});
