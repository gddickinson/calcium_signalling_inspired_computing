// Node test: wave_engine.js reproduces wave_based_computing (tests/fixtures/wave.json).
// Run: node web/static/wave_engine.test.mjs
import { WaveMedium, logicGate, memoryMedium, addRetrievalPulses, memoryReadout, filterMedium, filterReadout } from './wave_engine.js';
import { fixture, near } from './util.test.mjs';

const f = fixture('wave');
for (const r of f.media) {
  const m = new WaveMedium({ width: 40, height: 30, boundary: r.boundary, waveSpeed: r.wave_speed });
  near(m.substeps, r.substeps, 'substeps');
  m.addSource([8, 15], 0.5, 0.8); m.addSource([32, 15], 0.5, 0.8, Math.PI);
  m.addObstacle(18, 22, 10, 20); m.addResonantRegion(28, 36, 20, 28, 0.8); m.setMemory([5, 5], 0.6);
  m.run(40);
  const W = 40, tag = `${r.boundary} c=${r.wave_speed}`;
  near(Array.from(m.current.slice(15 * W, 16 * W)), r.row, `${tag} row`);
  near(Array.from({ length: 30 }, (_, y) => m.current[y * W + 30]), r.col, `${tag} col`);
  near(m.energy, r.energy, `${tag} energy`);
  near(m.waveEnergy(), r.wave_energy, `${tag} wave energy`, 1e-8);
  near(Array.from(m.frequency.slice(15 * W, 16 * W)), r.freq_row, `${tag} frequency`);
  near(Array.from(m.phase.slice(16 * W, 17 * W)), r.phase_row, `${tag} phase`, 1e-7);
  near(m.readMemory([5, 5]), r.memory, 'memory');
}
const base = new WaveMedium({ width: 40, height: 40 });
for (const [key, g] of Object.entries(f.gates)) {
  const op = key.slice(0, -2), bits = [key.at(-2) === '1', key.at(-1) === '1'];
  const r = logicGate(base, bits, op, 60);
  near({ output: r.output, value: r.value, reference: r.reference, threshold: r.threshold }, g, `gate ${key}`, 1e-8);
}
const mm = memoryMedium(base, [0.8, -0.5, 0.3]); mm.run(20); addRetrievalPulses(mm); mm.run(10);
near(memoryReadout(mm, 3), f.memory, 'memory readout');
const fm = filterMedium(base, f.signal, 'lowpass', 0.3); fm.run(25);
near(filterReadout(fm, f.signal.length), f.filtered, 'filter', 1e-7);
console.log('wave_engine: ok');
