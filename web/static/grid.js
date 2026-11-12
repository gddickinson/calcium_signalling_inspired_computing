// Grid helpers shared by the engines: 2-D arrays are Float64Array in row-major order (index r * W + c), as numpy
// arrays of shape (H, W). The convolutions reproduce scipy.signal.convolve2d(mode='same', boundary='wrap'|'symm')
// and scipy.ndimage.convolve(mode='constant', cval=0) for the odd, symmetric kernels the models use (so
// convolution and correlation coincide).

// k: Float64Array of size ks * ks (ks odd). boundary: 'wrap' (periodic), 'symm' (mirror including the edge cell,
// a b c | c b a) or 'constant' (zeros outside).
export function convolve(a, H, W, k, ks, boundary = 'wrap') {
  const out = new Float64Array(H * W), c = (ks - 1) / 2;
  const idx = (i, n) => {
    if (i >= 0 && i < n) return i;
    if (boundary === 'wrap') return ((i % n) + n) % n;
    if (boundary === 'symm') { const p = 2 * n; i = ((i % p) + p) % p; return i < n ? i : p - 1 - i; }
    return -1;
  };
  for (let r = 0; r < H; r++) {
    for (let q = 0; q < W; q++) {
      let s = 0;
      for (let u = 0; u < ks; u++) {
        const rr = idx(r + u - c, H);
        if (rr < 0) continue;
        for (let v = 0; v < ks; v++) {
          const qq = idx(q + v - c, W);
          if (qq < 0) continue;
          s += a[rr * W + qq] * k[u * ks + v];
        }
      }
      out[r * W + q] = s;
    }
  }
  return out;
}

// Normalise a kernel to sum 1 (in place) and return it.
export function normalise(k) {
  let s = 0;
  for (const v of k) s += v;
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return k;
}

export const sum = a => { let s = 0; for (const v of a) s += v; return s; };
export const mean = a => sum(a) / a.length;
export function std(a) {
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) ** 2;
  return Math.sqrt(s / a.length);
}

// Pearson correlation of two equal-length arrays (np.corrcoef(x, y)[0, 1]).
export function corr(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) { const dx = x[i] - mx, dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  return sxy / Math.sqrt(sxx * syy);
}

// Least-squares straight line (np.polyfit(x, y, 1)): [slope, intercept].
export function linfit(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const slope = sxy / sxx;
  return [slope, my - slope * mx];
}

// Linear-interpolated percentile (np.percentile default) of an array.
export function percentile(a, p) {
  const s = Float64Array.from(a).sort(), pos = (s.length - 1) * p / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
