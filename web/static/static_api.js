// Static-site data layer: answers api() calls from the JSON files exported by scripts/build_static.py (every GET in
// STATIC_ENDPOINTS of web/app.py, produced by the real Python code at build time). The models need no API: they
// run in the browser (*_engine.js).
const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path) {
  return file(path.split('?')[0].replace(/^\/+/, ''));
}
