// Node test: store_engine.js reproduces multistore_computing (tests/fixtures/store.json).
// Run: node web/static/store_engine.test.mjs
import { NumpyRandom } from './np_random.js';
import { MultiStore, hierarchicalMemory } from './store_engine.js';
import { fixture, near } from './util.test.mjs';

const pick = rs => rs.map(r => ({ source: r.source, confidence: r.confidence, result: r.result }));
for (const f of fixture('store').runs) {
  const rng = new NumpyRandom(f.seed), c = new MultiStore(rng);
  const writes = [[0.9, 'high'], [[0.5, 0.2, 0.7], 'medium'], [Array.from({ length: 50 }, () => rng.random()), 'low'], [0.95, 'high']]
    .map(([d, p]) => c.writeData(d, p));
  for (const seq of [[0.2, 0.4, 0.6, 0.8], new Array(6).fill(0.3), Array.from({ length: 5 }, () => rng.random())]) writes.push(c.writeData(seq, 'medium'));
  near(writes, f.writes, 'writes');
  near(c.writeLog.map(w => [w.store, w.start, w.length]), f.write_log, 'write log');
  c.run(25);
  near(c.transfers.map(t => t.length), f.transfers, 'transfers');
  near(pick(c.query(0.9)), f.query, 'query');
  for (const [k, s] of Object.entries(c.stores)) {
    near(Array.from(s.activation), f.state[k].activation, `${k} activation`);
    near([0, 1, 2].map(j => s.contents.reduce((a, row) => a + row[j], 0)), f.state[k].contents_sum, `${k} contents`);
  }
  const h = hierarchicalMemory(new MultiStore(rng), [0.8, [0.2, 0.4, 0.6], 0.5, [0.1, 0.3, 0.5], 0.3], [0.8, 0.5, 0.3], 20);
  near(h.writes.map(w => [w.priority, w.success, w.store]), f.hm.writes, 'hm writes');
  near(h.retrievals.map(r => pick(r.results)), f.hm.retrievals, 'hm retrievals');
}
console.log('store_engine: ok');
