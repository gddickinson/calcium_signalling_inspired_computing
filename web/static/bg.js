// Subtle animated background: a few slow, fading rings (decorative, not a model). Off under prefers-reduced-motion.
export function startBackground(canvas) {
  if (!canvas || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const ctx = canvas.getContext('2d'), rings = [];
  let w = 0, h = 0, last = 0;
  const resize = () => { w = canvas.width = innerWidth; h = canvas.height = innerHeight; };
  addEventListener('resize', resize); resize();
  const frame = t => {
    if (t - last > 50) {
      last = t;
      if (rings.length < 5 && Math.random() < 0.03) rings.push({ x: Math.random() * w, y: Math.random() * h, r: 2, a: 0.7 });
      ctx.clearRect(0, 0, w, h);
      for (const p of rings) {
        p.r += 1.1; p.a *= 0.975;
        ctx.strokeStyle = `rgba(61,255,216,${p.a})`; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 7); ctx.stroke();
      }
      for (let i = rings.length - 1; i >= 0; i--) if (rings[i].a < 0.02) rings.splice(i, 1);
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}
