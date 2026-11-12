// Node test: np_random.js reproduces numpy's legacy RNG (tests/fixtures/np_random.json).
// Run: node web/static/np_random.test.mjs
import { NumpyRandom } from './np_random.js';
import { fixture, near } from './util.test.mjs';

const f = fixture('np_random');
const r = new NumpyRandom(f.seed);
near(Array.from({ length: 5 }, () => r.random()), f.random, 'random');
near(Array.from({ length: 5 }, () => r.gauss()), f.randn, 'randn');
near(Array.from(r.normalN(0.5, 0.05, 7)), f.normal, 'normal');
near(Array.from({ length: 6 }, () => r.randint(5, 20)), f.randint, 'randint');
near(r.random(), f.random_after, 'random after gaussians');
near(Array.from({ length: 5 }, () => r.choiceOf([3, 8, 13, 21, 34])), f.choice_arr, 'choice(arr)');
const pool = [4, 9, 16, 25, 36, 49];
near(Array.from(r.choice(pool.length, 3), i => pool[i]), f.choice_noreplace, 'choice(replace=False)');
console.log('np_random: ok');
