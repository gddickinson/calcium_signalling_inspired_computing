// numpy's legacy global RNG (np.random.seed / random / randint / choice / normal / randn / poisson) in JavaScript:
// MT19937 with the same seeding, double construction, masked-rejection integers, Fisher-Yates permutation, polar
// Box-Muller Gaussians (legacy_gauss, with the cached second value) and Poisson samplers as numpy/random/src.
// With the same seed it returns the same numbers as numpy, so each JS engine reproduces a seeded Python run
// (np_random.test.mjs checks this against tests/fixtures/np_random.json).
// Taken from the calcium_simulator web port (same author); gauss/normal/choiceOf added here.

const N = 624, M = 397;

export class NumpyRandom {
  constructor(seed = 0) { this.mt = new Uint32Array(N); this.seed(seed); }

  seed(seed) {   // mt19937_seed (init_genrand), used by np.random.seed(int)
    const mt = this.mt;
    mt[0] = seed >>> 0;
    for (let i = 1; i < N; i++) {
      const s = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, s) + i) >>> 0;
    }
    this.pos = N;
    this.hasGauss = false; this.gaussNext = 0;
  }

  _gen() {
    const mt = this.mt;
    let y, i;
    for (i = 0; i < N - M; i++) {
      y = (mt[i] & 0x80000000) | (mt[i + 1] & 0x7fffffff);
      mt[i] = mt[i + M] ^ (y >>> 1) ^ (-(y & 1) & 0x9908b0df);
    }
    for (; i < N - 1; i++) {
      y = (mt[i] & 0x80000000) | (mt[i + 1] & 0x7fffffff);
      mt[i] = mt[i + (M - N)] ^ (y >>> 1) ^ (-(y & 1) & 0x9908b0df);
    }
    y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff);
    mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ (-(y & 1) & 0x9908b0df);
    this.pos = 0;
  }

  uint32() {
    if (this.pos === N) this._gen();
    let y = this.mt[this.pos++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  // np.random.rand(): 53-bit double in [0, 1)
  random() {
    const a = this.uint32() >>> 5, b = this.uint32() >>> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  // np.random.randn() / standard_normal(): legacy polar method, second value cached
  gauss() {
    if (this.hasGauss) { this.hasGauss = false; return this.gaussNext; }
    let x1, x2, r2;
    do {
      x1 = 2.0 * this.random() - 1.0; x2 = 2.0 * this.random() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 === 0.0);
    const f = Math.sqrt(-2.0 * Math.log(r2) / r2);
    this.gaussNext = f * x1; this.hasGauss = true;
    return f * x2;
  }

  // np.random.normal(loc, scale) (one value), and n values in order
  normal(loc = 0, scale = 1) { return loc + scale * this.gauss(); }
  normalN(loc, scale, n) { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = loc + scale * this.gauss(); return a; }

  // np.random.choice(arr) for a 1-D array: arr[randint(0, len)]
  choiceOf(arr) { return arr[this.randint(0, arr.length)]; }

  // Fill a Float64Array with np.random.rand(n) (row-major order for a 2-D shape)
  fill(out) { for (let i = 0; i < out.length; i++) out[i] = this.random(); return out; }

  // integer in [0, max] by masked rejection (random_interval / buffered_bounded_masked_uint32), max < 2^32
  interval(max) {
    if (max === 0) return 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
    mask >>>= 0;
    let v;
    while ((v = (this.uint32() & mask) >>> 0) > max);
    return v;
  }

  // np.random.randint(low, high): integer in [low, high)
  randint(low, high) { return low + this.interval(high - low - 1); }

  // np.random.permutation(n): Fisher-Yates from the end, as RandomState.shuffle
  permutation(n) {
    const a = new Int32Array(n);
    for (let i = 0; i < n; i++) a[i] = i;
    for (let i = n - 1; i >= 1; i--) {
      const j = this.interval(i);
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  // np.random.choice(n, k, replace=False) = permutation(n)[:k]
  choice(n, k) { return this.permutation(n).subarray(0, k); }

  // np.random.poisson(lam)
  poisson(lam) {
    if (lam >= 10) return this._poissonPtrs(lam);
    if (lam === 0) return 0;
    const enlam = Math.exp(-lam);
    let x = 0, prod = 1.0;
    for (;;) {
      prod *= this.random();
      if (prod > enlam) x += 1; else return x;
    }
  }

  _poissonPtrs(lam) {
    const slam = Math.sqrt(lam), loglam = Math.log(lam);
    const b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
    const invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
      const U = this.random() - 0.5, V = this.random();
      const us = 0.5 - Math.abs(U);
      const k = Math.floor((2 * a / us + b) * U + lam + 0.43);
      if (us >= 0.07 && V <= vr) return k;
      if (k < 0 || (us < 0.013 && V > us)) continue;
      if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + k * loglam - loggam(k + 1)) return k;
    }
  }
}

const LOGGAM_A = [8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04, -5.952380952380952e-04,
  8.417508417508418e-04, -1.917526917526918e-03, 6.410256410256410e-03, -2.955065359477124e-02,
  1.796443723688307e-01, -1.39243221690590e+00];

// log(Gamma(x)) as numpy's random_loggam
export function loggam(x) {
  if (x === 1.0 || x === 2.0) return 0.0;
  const n = x < 7.0 ? Math.trunc(7 - x) : 0;
  let x0 = x + n;
  const x2 = (1.0 / x0) * (1.0 / x0);
  let gl0 = LOGGAM_A[9];
  for (let k = 8; k >= 0; k--) { gl0 *= x2; gl0 += LOGGAM_A[k]; }
  let gl = gl0 / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * Math.log(x0) - x0;
  if (x < 7.0) for (let k = 1; k <= n; k++) { gl -= Math.log(x0 - 1.0); x0 -= 1.0; }
  return gl;
}
