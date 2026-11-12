// Node test: recruit_engine.js reproduces coordinated_recruitment (tests/fixtures/recruit.json).
// Run: node web/static/recruit_engine.test.mjs
import { NumpyRandom } from './np_random.js';
import { RecruitNetwork, faultTolerance, amplification } from './recruit_engine.js';
import { fixture, near } from './util.test.mjs';

for (const f of fixture('recruit').runs) {
  const rng = new NumpyRandom(f.seed), [H, W] = f.size;
  const net = new RecruitNetwork(rng, { height: H, width: W, radius: f.radius, faultProb: f.fault });
  near([...net.faulty.keys()].filter(k => net.faulty[k]), f.faulty, 'faulty');
  net.setInputRegion([Math.floor(H / 2), Math.floor(W / 2)], 5, 0.7);
  const a = net.run(f.steps);
  near(a.avgClusterSizes, f.avg_cluster_sizes, 'avg cluster sizes');
  near(a.numClusters, f.num_clusters, 'num clusters');
  near(a.totalActivation, f.total_activation, 'total activation');
  near(net.clusterHistory.slice(-3), f.clusters, 'clusters');
  near(Array.from(net.activationHistory.at(-1)), f.last.flat(), 'last activation');
  const pat = new Float64Array(H * W);
  for (let i = 3; i < 9; i++) for (let j = 4; j < 10; j++) pat[i * W + j] = 0.8;
  const ft = faultTolerance(rng, { height: H, width: W, radius: f.radius }, [pat], [0.0, 0.3], 2);
  near(ft.map(r => ({ fault_prob: r.faultProb, avg_activation: r.avgActivation, avg_max_cluster: r.avgMaxCluster, avg_num_clusters: r.avgNumClusters })), f.ft, 'fault tolerance');
  const amp = amplification(rng, { height: H, width: W, radius: f.radius }, [0.3, 0.9], 8);
  near(amp.map(r => ({ input_strength: r.strength, initial_input: r.input, final_activation: r.final, amplification_ratio: r.ratio, time_series: r.series })), f.amp, 'amplification');
}
const net = new RecruitNetwork(new NumpyRandom(1), { height: 16, width: 16, radius: 2 });
for (const [k, c] of fixture('recruit').dbscan.entries()) near(net.identifyClusters(Float64Array.from(c.grid.flat()), 0.5), c.clusters, `dbscan case ${k}`);
console.log('recruit_engine: ok');
