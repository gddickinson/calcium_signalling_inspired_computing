// Small canvas charts shared by the model pages: a multi-series line/scatter chart and helpers for mapping data
// coordinates to a canvas (adapted from the creative-genetic-problem-solver web port).

export const COLORS = ['#3dffd8', '#ffb03d', '#7b8cff', '#ff4fa0', '#b6ff3d', '#ff6b6b', '#c9d4ff'];

// Fit a canvas's backing store to its displayed width (keeps the aspect ratio from its width/height attributes).
export function fitCanvas(cv) {
  const ratio = cv.height / cv.width, w = Math.max(200, Math.round(cv.clientWidth || cv.width));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(w * ratio * dpr); }
  return cv.getContext('2d');
}

// series: [{ data: number[], xs?: number[] (x values; default index + x0), label, color, dots, dash }]
// opts: { xLabel, yLabel, x0, yMin, yMax, hlines: [{ y, label, color }] }
export function lineChart(cv, series, { xLabel = 'Step', yLabel = '', x0 = 0, yMin, yMax, hlines = [] } = {}) {
  const ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600;
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
  const pad = { l: 58 * s, r: 12 * s, t: 14 * s, b: 34 * s };
  const vals = series.flatMap(d => Array.from(d.data).filter(Number.isFinite)).concat(hlines.map(h => h.y));
  const xsAll = series.flatMap(d => (d.xs ? d.xs : Array.from(d.data, (_, i) => i + x0)));
  let lo = yMin ?? (vals.length ? Math.min(...vals) : 0), hi = yMax ?? (vals.length ? Math.max(...vals) : 1);
  if (hi - lo < 1e-12) { hi += Math.abs(hi) * 0.05 + 1e-6; lo -= Math.abs(lo) * 0.05 + 1e-6; }
  let xa = xsAll.length ? Math.min(...xsAll) : x0, xb = xsAll.length ? Math.max(...xsAll) : x0 + 1;
  if (xb - xa < 1e-12) xb = xa + 1;
  const X = x => pad.l + (W - pad.l - pad.r) * (x - xa) / (xb - xa), Y = v => H - pad.b - (H - pad.t - pad.b) * (v - lo) / (hi - lo);
  ctx.strokeStyle = '#212c4c'; ctx.lineWidth = 1; ctx.font = `${11 * s}px ui-monospace,Menlo,monospace`; ctx.fillStyle = '#94a3c6';
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(fmt(Math.abs(v) < (hi - lo) * 1e-9 ? 0 : v), pad.l - 6 * s, y + 4 * s);
  }
  ctx.textAlign = 'center';
  for (let k = 0; k <= 4; k++) { const x = xa + (xb - xa) * k / 4; ctx.fillText(fmt(x), X(x), H - pad.b + 15 * s); }
  ctx.fillText(xLabel, (pad.l + W - pad.r) / 2, H - 4 * s);
  ctx.save(); ctx.translate(12 * s, (pad.t + H - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
  for (const h of hlines) {
    ctx.strokeStyle = h.color || '#ff5c7a'; ctx.setLineDash([6 * s, 5 * s]); ctx.beginPath(); ctx.moveTo(pad.l, Y(h.y)); ctx.lineTo(W - pad.r, Y(h.y)); ctx.stroke();
    ctx.setLineDash([]); if (h.label) { ctx.fillStyle = h.color || '#ff5c7a'; ctx.textAlign = 'right'; ctx.fillText(h.label, W - pad.r - 4 * s, Y(h.y) - 4 * s); }
  }
  series.forEach((d, j) => {
    const col = d.color || COLORS[j % COLORS.length], xs = d.xs || Array.from(d.data, (_, i) => i + x0);
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2 * s; ctx.setLineDash(d.dash ? [5 * s, 4 * s] : []);
    if (!d.dotsOnly) {
      ctx.beginPath(); let pen = false;
      Array.from(d.data).forEach((v, i) => { if (!Number.isFinite(v)) { pen = false; return; } pen ? ctx.lineTo(X(xs[i]), Y(v)) : ctx.moveTo(X(xs[i]), Y(v)); pen = true; });
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (d.dots || d.dotsOnly) Array.from(d.data).forEach((v, i) => { if (Number.isFinite(v)) { ctx.beginPath(); ctx.arc(X(xs[i]), Y(v), 3.5 * s, 0, 7); ctx.fill(); } });
  });
  if (series.length > 1) {
    ctx.textAlign = 'left'; let x = pad.l + 6 * s;
    series.forEach((d, j) => { ctx.fillStyle = d.color || COLORS[j % COLORS.length]; ctx.fillText('■ ' + d.label, x, pad.t + 10 * s); x += ctx.measureText('■ ' + d.label).width + 14 * s; });
  }
}

export function fmt(v) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 0.01 || a >= 1e5) ? v.toExponential(2) : (+v.toFixed(a < 1 ? 3 : a < 100 ? 2 : 1)).toString();
}

// Map a data box [x0,x1]x[y0,y1] onto the canvas with a margin; y grows upwards. Returns {X, Y, scale}.
export function frame(cv, x0, x1, y0, y1, margin = 0.06) {
  const W = cv.width, H = cv.height, m = margin * Math.min(W, H);
  const k = Math.min((W - 2 * m) / (x1 - x0), (H - 2 * m) / (y1 - y0));
  const ox = (W - k * (x1 - x0)) / 2, oy = (H - k * (y1 - y0)) / 2;
  return { X: x => ox + k * (x - x0), Y: y => H - oy - k * (y - y0), scale: k };
}
