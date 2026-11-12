// Draw a 2-D grid (Float64Array, row-major H x W) onto a canvas with a colour map; nearest-neighbour scaling.
const MAPS = {
  hot: [[0, 0, 0], [140, 10, 0], [255, 80, 0], [255, 200, 40], [255, 255, 230]],
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  diverge: [[61, 140, 255], [20, 40, 90], [6, 8, 20], [110, 50, 10], [255, 176, 61]],
  mint: [[6, 8, 20], [10, 60, 70], [20, 150, 140], [61, 255, 216], [230, 255, 250]],
};
export const CATS = [[12, 16, 32], [255, 176, 61], [123, 140, 255], [255, 79, 160], [61, 255, 216]];

export function colour(map, t) {
  const S = MAPS[map]; t = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (S.length - 1);
  const i = Math.min(S.length - 2, Math.floor(t)), f = t - i, a = S[i], b = S[i + 1];
  return [0, 1, 2].map(k => Math.round(a[k] + (b[k] - a[k]) * f));
}

export const cssGradient = map => `linear-gradient(90deg, ${MAPS[map].map(c => `rgb(${c})`).join(',')})`;

// opts: { map, lo, hi, cats (integer values -> CATS), overlay(ctx, sx, sy) }
export function drawGrid(cv, vals, H, W, { map = 'hot', lo = 0, hi = 1, cats = false, overlay = null } = {}) {
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const octx = off.getContext('2d'), img = octx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const c = cats ? CATS[vals[i] | 0] || CATS[0] : colour(map, (vals[i] - lo) / (hi - lo || 1));
    img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  const w = Math.max(200, Math.round(cv.clientWidth || cv.width)), dpr = Math.min(2, devicePixelRatio || 1);
  const ratio = cv.height / cv.width;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(w * ratio * dpr); }
  const ctx = cv.getContext('2d'); ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, cv.width, cv.height);
  if (overlay) overlay(ctx, cv.width / W, cv.height / H);
}

// Small HTML colour bar with end labels
export const colourBar = (map, lo, hi, label = '') =>
  `<div class="cbar" style="background:${cssGradient(map)}"></div><div class="cbar-l"><span>${lo}</span><span>${label}</span><span>${hi}</span></div>`;
