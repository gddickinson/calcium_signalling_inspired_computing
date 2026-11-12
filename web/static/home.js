// Overview page: the idea in the repository's own words, the six models as cards, a live puff grid in the hero.
import { route, api, app, esc, list, animate } from './core.js';
import { NumpyRandom } from './np_random.js';
import { CalciumPuff } from './puff_engine.js';
import { drawGrid } from './heat.js';

const BLURB = {
  puffs: 'Cells fire at random with a probability set by their local "channel density"; stochastic bitstreams do arithmetic.',
  patterns: 'Three coupled fields blur at different rates, interact through a fixed matrix and saturate with tanh.',
  threshold: 'Leaky sigmoid units with a refractory period integrate weak inputs until they cross threshold.',
  recruitment: 'Elements fire only when direct input plus neighbour activity passes their threshold, forming clusters.',
  stores: 'Three memory stores with different size, reliability and persistence pass strong activity between them.',
  waves: 'Oscillating sources drive a damped 2-D medium with obstacles, resonant regions and memory elements.',
};

route('home', async () => {
  const [info, models, ov] = await Promise.all([api('/info'), api('/models'), api('/overview')]);
  app().innerHTML = `
  <section class="hero two" style="align-items:center">
    <div><p class="kicker">Calcium-signalling-inspired computing · live in your browser</p>
      <h1 class="glow">${esc(info.subtitle)}</h1>
      <p class="lead">${esc(info.overview)}</p>
      <a class="btn" href="#/puffs">Start with model 1</a><a class="btn alt" href="#/about">About the port</a></div>
    <figure class="fig"><canvas id="hero" width="400" height="400" role="img" aria-label="Live run of the probabilistic puff model: bright cells are recently activated; the four quadrants have channel densities 0.2, 0.4, 0.6 and 0.8"></canvas>
      <figcaption class="small mute" id="herocap">Model 1 running live: calcium-like signal on a 50 x 50 grid.</figcaption></figure>
  </section>
  <div class="tiles">
    <div class="tile"><b>${info.models}</b><span>models in the repository</span></div>
    <div class="tile"><b>${info.classes}</b><span>Python classes ported</span></div>
    <div class="tile"><b>${info.engines}</b><span>JS engines checked against Python</span></div>
    <div class="tile"><b>0</b><span>installs needed</span></div>
  </div>
  <div class="callout a"><b>Inspired by, not a model of.</b> These are abstract computing schemes whose rules borrow ideas
    from calcium signalling (stochastic channel opening, waves, thresholds, coordinated recruitment, separate stores).
    They run on dimensionless grids in time steps with arbitrary units; they are not quantitative models of cells, and the
    repository does not cite specific studies for the analogies. Each page shows the rules exactly as coded.</div>
  <h2>The models</h2>
  <div class="grid wide">${list(models, m => `<a class="card door" href="#/${m.route}"><span class="num">MODEL ${m.number}</span>
    <b>${esc(m.title)}</b><span>${esc(BLURB[m.route] || m.intro)}</span>
    <span>${list(m.features, f => `<span class="tag c">${esc(f)}</span>`)}</span></a>`)}</div>
  <div class="two">
    <div><h2>Research background</h2><p class="mute small">From the README: the project is inspired by research on calcium signalling dynamics, particularly</p>
      <ul class="feat">${list(ov.background, b => `<li>${esc(b)}</li>`)}</ul></div>
    <div><h2>Possible applications</h2><p class="mute small">The README lists these as <em>potential</em> applications; the repository contains demonstrations, not benchmarks.</p>
      <ul class="feat">${list(ov.applications, b => `<li>${esc(b)}</li>`)}</ul></div>
  </div>`;
  hero();
});

function hero() {
  const cv = document.getElementById('hero');
  let m, n = 0;
  const reset = () => { m = new CalciumPuff(new NumpyRandom(Math.floor(Math.random() * 1000)), { gridSize: 50 }); n = 0; };
  const draw = () => drawGrid(cv, m.calcium, 50, 50, { map: 'hot', lo: 0, hi: 1 });
  reset();
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { m.run(60); draw(); return; }
  let tick = 0;
  animate(() => {
    if (++tick % 3) return;
    m.update(); if (++n > 600) reset();
    draw();
  });
}
