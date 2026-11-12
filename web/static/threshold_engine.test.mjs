// Node test: threshold_engine.js reproduces threshold_computing (tests/fixtures/threshold.json).
// Run: node web/static/threshold_engine.test.mjs
import { NumpyRandom } from './np_random.js';
import { ThresholdNetwork, ThresholdLogic, Unit } from './threshold_engine.js';
import { fixture, near } from './util.test.mjs';

for (const f of fixture('threshold').runs) {
  const rng = new NumpyRandom(f.seed);
  const net = new ThresholdNetwork(rng, { height: f.size[0], width: f.size[1], radius: f.radius, prob: f.prob });
  near(net.nConnections, f.n_conn, 'connections');
  near(net.units.map(u => u.threshold), f.thresholds, 'thresholds');
  net.setInput(Array.from({ length: net.input.length }, () => rng.random() * 0.8));
  near(net.run(15), f.free, 'free run');
  const logic = new ThresholdLogic(net);
  const weak = Array.from({ length: 8 }, () => Array.from({ length: net.input.length }, () => rng.random() * 0.3));
  near(logic.temporalIntegration(weak, 20), f.integ, 'temporal integration');
  near(logic.thresholdDetection([0.2, 0.6, 1.0], 10), f.detect, 'threshold detection');
  near(logic.analogMemory([0.5, 0.9], 15), f.memory, 'analog memory');
  near(Array.from(net.potentialHistory.at(-1)), f.potentials.flat(), 'potentials');
}
const u = new Unit({ refractoryPeriod: 0 });
for (let k = 0; k < 3; k++) u.update(2.0);
if (!(u.activation >= 0 && u.activation <= 1)) throw new Error('refractory 0');
console.log('threshold_engine: ok');
