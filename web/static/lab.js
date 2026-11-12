// Shared layout for the model pages (adapted from the creative-genetic-problem-solver web port): parameter controls
// with units built from a spec list, a run/pause/step/reset loop driven by requestAnimationFrame, figure canvases,
// a read-out table, and the "from the repository" panel with the README text for the model.
import { app, esc, list, animate, onLeave, api } from './core.js';

// spec: { id, label, type: 'range'|'select'|'check', min, max, step, value, options: [[value, text]], unit, live,
//         hint (shown under the control) }
function controlHtml(c) {
  const hint = c.hint ? `<div class="small mute">${esc(c.hint)}</div>` : '';
  if (c.type === 'select') {
    return `<label for="p-${c.id}">${esc(c.label)}</label><select id="p-${c.id}">${list(c.options, ([v, t]) =>
      `<option value="${esc(v)}"${String(v) === String(c.value) ? ' selected' : ''}>${esc(t)}</option>`)}</select>${hint}`;
  }
  if (c.type === 'check') return `<label><input type="checkbox" id="p-${c.id}"${c.value ? ' checked' : ''}> ${esc(c.label)}</label>${hint}`;
  const u = c.unit ? ` ${c.unit}` : '';
  return `<label for="p-${c.id}">${esc(c.label)} <b id="v-${c.id}">${c.value}${esc(u)}</b></label>
    <input type="range" id="p-${c.id}" min="${c.min}" max="${c.max}" step="${c.step || 1}" value="${c.value}">${hint}`;
}

// opts: { kicker, title, lead, route (README model), controls, seed (default seed; omit for deterministic models),
//         figures: HTML for the view column, onReset(params), step(), render(), done(), notes (HTML), source,
//         maxSpeed, speed, after (HTML below the lab) }
export function lab(opts) {
  const controls = [...opts.controls];
  if (opts.seed !== undefined) controls.push({ id: 'seed', label: 'Random seed (np.random.seed)', min: 0, max: 999, value: opts.seed });
  controls.push({ id: 'speed', label: 'Steps per animation frame', min: 1, max: opts.maxSpeed || 20, value: opts.speed ?? 1, live: true });
  app().innerHTML = `
  <section class="hero"><p class="kicker">${esc(opts.kicker)}</p><h1 class="glow">${esc(opts.title)}</h1>
    <p class="lead">${opts.lead}</p></section>
  <div class="lab">
    <aside class="controls panel" aria-label="Parameters">
      ${list(controls, c => `<div class="ctl">${controlHtml(c)}</div>`)}
      <div class="btns"><button id="run">Run</button><button class="alt" id="stepb">Step</button><button class="alt" id="reset">Reset</button></div>
      <p class="small mute" id="status" aria-live="polite"></p>
    </aside>
    <div class="view">${opts.figures}<div class="readout" id="readout"></div></div>
  </div>
  ${opts.after || ''}
  <div class="notes">${opts.notes || ''}
  <div id="repo-text"></div>
  <p class="small mute">Ported from <code>${esc(opts.source)}</code>. ${opts.seed !== undefined
    ? 'Random numbers come from a JavaScript copy of numpy\'s legacy generator, so a seed replays the seeded Python run number for number.'
    : 'The model uses no random numbers.'} The port is checked against the Python with <code>node web/static/*.test.mjs</code>.</p></div>`;

  const val = c => { const el = document.getElementById('p-' + c.id);
    return c.type === 'select' ? el.value : c.type === 'check' ? el.checked : +el.value; };
  const params = () => Object.fromEntries(controls.map(c => [c.id, val(c)]));
  let running = false, stop = null, ready = false;
  const runBtn = document.getElementById('run'), status = document.getElementById('status');
  const draw = () => opts.render();
  const reset = () => { pause(); opts.onReset(params()); ready = true; runBtn.textContent = 'Run'; draw(); };
  const tick = () => {
    const n = val(controls[controls.length - 1]);
    for (let i = 0; i < n && !opts.done(); i++) opts.step();
    draw();
    if (opts.done()) pause();
  };
  function pause() { running = false; if (stop) stop(); stop = null; runBtn.textContent = ready && opts.done() ? 'Run again' : 'Run'; }
  runBtn.onclick = () => {
    if (running) return pause();
    if (opts.done()) reset();
    running = true; runBtn.textContent = 'Pause'; stop = animate(tick);
  };
  document.getElementById('stepb').onclick = () => { pause(); if (!opts.done()) { opts.step(); draw(); } };
  document.getElementById('reset').onclick = reset;
  for (const c of controls) {
    const el = document.getElementById('p-' + c.id);
    el.addEventListener('input', () => {
      const b = document.getElementById('v-' + c.id); if (b) b.textContent = el.value + (c.unit ? ' ' + c.unit : '');
      if (!c.live) reset(); else if (opts.onLive) { opts.onLive(params()); draw(); }
    });
  }
  onLeave(() => { running = false; if (stop) stop(); });
  if (opts.route) repoText(opts.route);
  queueMicrotask(reset);
  return {
    params,
    setStatus: t => { status.textContent = t; },
    setReadout: rows => { document.getElementById('readout').innerHTML = list(rows, ([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`); },
    canvas: id => document.getElementById(id),
  };
}

// The README section and the demo script's docstring for a model, plus its class defaults (from /api/models).
export async function repoText(route) {
  const host = document.getElementById('repo-text');
  const m = (await api('/models')).find(x => x.route === route);
  if (!m || !host) return;
  host.innerHTML = `<h2>From the repository</h2>
  <div class="panel"><h3>README · ${esc(m.number)}. ${esc(m.title)}</h3><p>${esc(m.intro)}</p>
    <ul class="feat">${list(m.features, f => `<li>${esc(f)}</li>`)}</ul>
    <p class="small">Desktop demo: <code>${esc(m.run)}</code></p>
    <details class="src"><summary>Module docstring and class defaults</summary>
      <pre>${esc(m.docstring)}</pre>
      ${list(m.classes, c => `<p class="defaults"><b>${esc(c.name)}</b> (${esc(c.module)}): ${esc(Object.entries(c.defaults).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ') || 'no keyword defaults')}</p>`)}
    </details></div>`;
}

// One figure (canvas + caption)
export const fig = (id, w, h, caption, extra = '') =>
  `<figure class="fig"><canvas id="${id}" width="${w}" height="${h}" role="img" aria-label="${esc(caption)}"></canvas>
   <figcaption class="small mute">${esc(caption)}</figcaption>${extra}</figure>`;
