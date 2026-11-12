// Model 4 page: coordinated recruitment (CoordinatedRecruitmentNetwork) with DBSCAN clusters, plus the
// QuorumComputer fault-tolerance and amplification experiments (recruit_engine.js).
import { route, esc } from './core.js';
import { NumpyRandom } from './np_random.js';
import { RecruitNetwork, faultTolerance, amplification } from './recruit_engine.js';
import { lab, fig } from './lab.js';
import { drawGrid, colourBar } from './heat.js';
import { lineChart, fmt } from './chart.js';

const squarePattern = (H, W) => { const p = new Float64Array(H * W); for (let i = 10; i < Math.min(20, H); i++) for (let j = 10; j < Math.min(20, W); j++) p[i * W + j] = 0.8; return p; };

route('recruitment', () => {
  let net, steps, p0;
  const L = lab({
    kicker: 'Model 4 · coordinated recruitment', title: 'Quorum-like recruitment on a grid', route: 'recruitment',
    lead: `Each element fires when its direct input (times its sensitivity) plus the weighted activity of its neighbours
      passes its own threshold. Firing sets its activation to that total (at most 1), makes it fatigued (less
      sensitive) and puts it into a 5-step recovery; otherwise its activation decays by 20% per step. Clusters of
      elements above 0.3 are found with DBSCAN, and a fraction of elements can be marked faulty (always silent).`,
    seed: 42, maxSpeed: 10, speed: 1,
    controls: [
      { id: 'n', label: 'Grid size', min: 12, max: 50, value: 30, unit: 'cells', hint: 'the demo uses 30 x 30' },
      { id: 'radius', label: 'Coupling radius', min: 1, max: 6, value: 3, unit: 'cells', hint: 'kernel exp(−d / (radius/2)), normalised' },
      { id: 'fault', label: 'Fault probability', min: 0, max: 0.6, step: 0.01, value: 0.05, hint: 'fraction of elements that never fire (demo 0.05)' },
      { id: 'input', label: 'Input', type: 'select', value: 'circle', options: [['circle', 'Circle in the centre, strength falling to 0 at the edge (demo)'], ['square', 'Uniform square, rows/cols 10-19 (fault-test pattern)']] },
      { id: 'strength', label: 'Input strength (circle centre)', min: 0.1, max: 2, step: 0.05, value: 0.7, unit: 'a.u.' },
      { id: 'rin', label: 'Input radius', min: 2, max: 12, value: 5, unit: 'cells' },
      { id: 'steps', label: 'Steps to run', min: 10, max: 100, step: 5, value: 30, unit: 'steps' },
    ],
    figures: `<div class="grid2">${fig('act', 300, 300, 'Element activation (blue squares: faulty elements)', colourBar('hot', 0, 1, 'activation'))}
        ${fig('clu', 300, 300, 'DBSCAN clusters of elements above 0.3 (one colour per cluster)')}</div>
      ${fig('chart', 600, 230, 'Total activation and number of clusters per step')}`,
    onReset(p) {
      net = new RecruitNetwork(new NumpyRandom(p.seed), { height: p.n, width: p.n, radius: p.radius, faultProb: p.fault });
      if (p.input === 'circle') net.setInputRegion([Math.floor(p.n / 2), Math.floor(p.n / 2)], p.rin, p.strength);
      else net.setInputPattern(squarePattern(p.n, p.n));
      steps = p.steps; p0 = p;
    },
    step: () => net.update(), done: () => net.activationHistory.length >= steps, render,
    source: 'coordinated_recruitment/element.py, network.py and quorum.py',
    after: `<h2>Experiments from QuorumComputer</h2>
      <p>Both build fresh networks (same grid size, coupling radius and seed as above) and run 20 steps each, as the Python does.</p>
      <div class="grid2"><div class="panel"><h3>Fault tolerance</h3><p class="small">Square input (0.8), fault probability 0 to 0.5 in 6 steps, 5 networks each.</p>
        <button id="ftb">Run sweep</button><span class="small mute" id="fts"></span>${fig('ft', 400, 240, 'Mean final total activation and mean of the largest average cluster size vs fault probability')}</div>
      <div class="panel"><h3>Signal amplification</h3><p class="small">Disc of radius min(H, W)/6 at strengths 0.1 to 1.0; ratio = final total activation / total input.</p>
        <button id="ampb">Run</button><span class="small mute" id="amps"></span>${fig('amp', 400, 240, 'Total activation over time for each input strength')}<div id="ampt" class="small"></div></div></div>`,
    notes: `<h2>The element as coded</h2>
      <p class="eq">total = input·sensitivity·(1 − 0.5·fatigue)·[0.5 if recovering] + 0.2·neighbourhood</p>
      <p>Sensitivities are 0.4 + 0.2·N(0, 1) clipped to [0.2, 0.8]; thresholds 0.3 + 0.1·N(0, 1) clipped to [0.1, 0.5];
      coupling 0.2. Fatigue rises by 0.1 per firing and falls by 0.01 per quiet step. With the demo input (0.7 at the
      centre) many elements stay just below threshold, so few clusters form; stronger or uniform input recruits more.</p>`,
  });

  function render() {
    const n = net.H, last = net.activationHistory.at(-1) || net.activationGrid();
    drawGrid(L.canvas('act'), last, n, n, { map: 'hot', overlay: (ctx, sx, sy) => {
      ctx.strokeStyle = '#7b8cff'; ctx.lineWidth = 1.5;
      net.faulty.forEach((f, k) => { if (f) ctx.strokeRect((k % n) * sx + 1, Math.floor(k / n) * sy + 1, sx - 2, sy - 2); });
    } });
    const cm = new Float64Array(n * n), cs = net.clusterHistory.at(-1) || [];
    cs.forEach((c, i) => c.forEach(k => { cm[k] = 1 + (i % 4); }));
    drawGrid(L.canvas('clu'), cm, n, n, { cats: true });
    const a = net.analyze();
    lineChart(L.canvas('chart'), [{ data: a.totalActivation, label: 'total activation' }, { data: a.numClusters, label: 'clusters' }], { xLabel: 'Step', yLabel: 'value', x0: 1 });
    L.setReadout([['Step', `${net.activationHistory.length} / ${steps}`], ['Faulty elements', net.faulty.reduce((s, v) => s + v, 0)],
      ['Elements above 0.3', last.filter(v => v > 0.3).length], ['Clusters (sizes)', cs.length ? cs.map(c => c.length).join(', ') : '0'],
      ['Total activation', fmt(a.totalActivation.at(-1) ?? 0)]]);
    L.setStatus(net.activationHistory.length >= steps ? 'Finished.' : net.activationHistory.length ? 'Running...' : 'Press Run or Step.');
  }

  const later = (status, fn) => { document.getElementById(status).textContent = ' running...'; setTimeout(() => { fn(); document.getElementById(status).textContent = ' done.'; }, 30); };
  const base = () => { const rng = new NumpyRandom(p0.seed); new RecruitNetwork(rng, { height: p0.n, width: p0.n, radius: p0.radius, faultProb: p0.fault }); return rng; };
  document.getElementById('ftb').onclick = () => later('fts', () => {
    const probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
    const r = faultTolerance(base(), { height: p0.n, width: p0.n, radius: p0.radius }, [squarePattern(p0.n, p0.n)], probs, 5);
    lineChart(document.getElementById('ft'), [{ data: r.map(x => x.avgActivation), xs: probs, label: 'activation', dots: true },
      { data: r.map(x => x.avgMaxCluster), xs: probs, label: 'max cluster size', dots: true }], { xLabel: 'Fault probability', yLabel: 'mean over 5 networks' });
  });
  document.getElementById('ampb').onclick = () => later('amps', () => {
    const s = [0.1, 0.325, 0.55, 0.775, 1.0];
    const r = amplification(base(), { height: p0.n, width: p0.n, radius: p0.radius }, s, 20);
    lineChart(document.getElementById('amp'), r.map(x => ({ data: x.series, label: `input ${fmt(x.strength)}` })), { xLabel: 'Step', yLabel: 'total activation', x0: 1 });
    document.getElementById('ampt').innerHTML = `Amplification ratio: ${r.map(x => `${fmt(x.strength)} → ${esc(fmt(x.ratio))}`).join(' · ')}`;
  });
});
