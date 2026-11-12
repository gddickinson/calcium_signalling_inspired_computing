// About page: what the site is, how to run the desktop demos and the web app, how the JS ports are checked, credits.
import { route, api, app, esc, list } from './core.js';

route('about', async () => {
  const [models, checks, ov] = await Promise.all([api('/models'), api('/checks'), api('/overview')]);
  app().innerHTML = `
  <section class="hero"><p class="kicker">About</p><h1 class="glow">About this site</h1>
    <p class="lead">A browser front end for the repository <code>calcium_signalling_inspired_computing</code>: six
    computational models inspired by calcium signalling dynamics. Every model page runs a JavaScript port of the Python
    classes, so the static site is the working collection; the text on each page comes from the README and the code.</p></section>
  <h2>Run it yourself</h2>
  <div class="two"><div class="panel"><h3>Desktop demos (matplotlib)</h3>
    <pre>pip install -r requirements.txt
${esc(models.map(m => m.run).join('\n'))}</pre></div>
  <div class="panel"><h3>Web app (local)</h3>
    <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8144
python scripts/build_static.py   # static site in site/</pre>
    <p class="small">The local app serves the same pages; the API supplies the README text, class defaults and the
    Python-only pattern-recognition run (scikit-learn), which the static site reads from JSON exported at build time.</p></div></div>
  <h2>How the ports are checked</h2>
  <p>Each engine replays seeded runs of the Python classes. Random numbers come from a JavaScript copy of numpy's legacy
  MT19937 generator (<code>np.random.seed</code>, <code>random</code>, <code>randn</code>, <code>normal</code>,
  <code>randint</code>, <code>choice</code>), so a seed gives the same numbers in both. Reference values are written by
  <code>scripts/make_*_fixture.py</code> and compared to a relative tolerance of 10⁻⁹ by node tests; pytest checks that
  the fixtures still match the current Python.</p>
  <div class="scroll"><table><tr><th>JS engine</th><th>Reproduces</th><th>Fixture</th></tr>
  ${list(checks, c => `<tr><td><code>${esc(c.engine)}</code></td><td>${esc(c.python)}</td><td><code>${esc(c.fixture)}</code>${c.runs ? ` (${c.runs} runs)` : ''}</td></tr>`)}</table></div>
  <p class="small mute">Not ported: the PCA-based analyses (spatiotemporal PCA summaries, wave pattern recognition by PCA
  signatures) and the matplotlib figures. Pattern recognition for model 2 is shown from a Python run.</p>
  <h2>Scientific scope</h2>
  <p>These are abstract computing schemes with rules borrowed from calcium signalling; all quantities are dimensionless
  (grid cells, time steps, arbitrary units). The README names the research themes that inspired them
  (${esc(ov.background.join('; '))}) but cites no specific studies, so the pages make no claims about biology.</p>
  <h2>Credits and licence</h2>
  <p>Models by George Dickinson. ${esc(ov.license)}</p>
  <ul class="feat">${list(ov.acknowledgments, a => `<li>${esc(a)}</li>`)}</ul>
  <p class="small mute">The code was written with AI assistance from Claude (Anthropic), as the README notes; this web
  interface was added the same way.</p>`;
});
