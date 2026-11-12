// Entry point: registers the lazily loaded pages, sets the footer mode text, starts the router and the background.
import { start, route, lazy, app, isStatic, $ } from './core.js';
import { startBackground } from './bg.js';

const PAGES = {
  'home.js': ['home'], 'about.js': ['about'], 'puffs.js': ['puffs'], 'patterns.js': ['patterns'],
  'threshold.js': ['threshold'], 'recruitment.js': ['recruitment'], 'stores.js': ['stores'], 'waves.js': ['waves'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));

route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: all six models run in your browser (JS ports of the Python code)'
  : 'local app: Python server on this machine';
startBackground($('#bg'));
start();
