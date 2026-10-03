import { useMemo, useState } from "react";

const PADDING = { top: 12, right: 16, bottom: 28, left: 46 };
const HEIGHT = 200;

function niceTicks(min, max, count = 4) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const rawStep = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const residual = rawStep / magnitude;
  const step = (residual > 5 ? 10 : residual > 2 ? 5 : residual > 1 ? 2 : 1) * magnitude;
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = niceMin; v <= niceMax + step / 2; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

export default function LineChart({ title, data, xKey = "epoch", yKey = "loss", color = "var(--accent)" }) {
  const [showTable, setShowTable] = useState(false);
  if (!data || data.length === 0) {
    return (
      <figure className="card p-4 flex flex-col items-center justify-center text-center gap-1" style={{ minHeight: HEIGHT + 40 }}>
        <span className="text-sm font-medium text-secondary font-display">{title}</span>
        <span className="text-xs text-muted">Waiting for data…</span>
      </figure>
    );
  }

  const xValues = data.map((d) => d[xKey]);
  const xMin = Math.min(...xValues), xMax = Math.max(...xValues);
  const yValues = data.map((d) => d[yKey]);
  const yMin = Math.min(0, ...yValues), yMax = Math.max(...yValues);
  const yTicks = useMemo(() => niceTicks(yMin, yMax), [yMin, yMax]);
  const niceYMin = yTicks[0], niceYMax = yTicks[yTicks.length - 1];

  const width = 480;
  const plotW = width - PADDING.left - PADDING.right;
  const plotH = HEIGHT - PADDING.top - PADDING.bottom;
  const xScale = (x) => (xMax === xMin ? plotW / 2 : ((x - xMin) / (xMax - xMin)) * plotW);
  const yScale = (y) => plotH - ((y - niceYMin) / (niceYMax - niceYMin || 1)) * plotH;

  const points = data.map((d) => [xScale(d[xKey]), yScale(d[yKey])]);
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ");

  return (
    <figure className="card p-4">
      <div className="flex items-center justify-between mb-2">
        <figcaption className="text-sm font-medium text-secondary font-display">{title}</figcaption>
        <button type="button" onClick={() => setShowTable((v) => !v)} className="text-xs text-secondary border border-[var(--border)] rounded px-2 py-0.5 hover:bg-[var(--surface-2)]">
          {showTable ? "Chart view" : "Table view"}
        </button>
      </div>
      {showTable ? (
        <table className="w-full text-xs mono">
          <thead><tr className="text-left text-muted"><th className="pr-4 py-1">{xKey}</th><th className="pr-4 py-1">{yKey}</th></tr></thead>
          <tbody>
            {data.map((d, i) => (
              <tr key={i} className="border-t border-[var(--gridline)]">
                <td className="pr-4 py-1">{d[xKey]}</td><td className="pr-4 py-1">{d[yKey].toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <svg viewBox={`0 0 ${width} ${HEIGHT}`} className="w-full" role="img" aria-label={title}>
          <g transform={`translate(${PADDING.left},${PADDING.top})`}>
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={0} x2={plotW} y1={yScale(t)} y2={yScale(t)} stroke="var(--gridline)" strokeWidth={1} />
                <text x={-8} y={yScale(t)} dy="0.32em" textAnchor="end" fontSize={10} fontFamily="var(--font-display)" fill="var(--text-muted)">{t}</text>
              </g>
            ))}
            <line x1={0} x2={0} y1={0} y2={plotH} stroke="var(--baseline)" strokeWidth={1} />
            <line x1={0} x2={plotW} y1={plotH} y2={plotH} stroke="var(--baseline)" strokeWidth={1} />
            {data.map((d) => (
              <text key={d[xKey]} x={xScale(d[xKey])} y={plotH + 18} textAnchor="middle" fontSize={10} fontFamily="var(--font-display)" fill="var(--text-muted)">{d[xKey]}</text>
            ))}
            <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            {points.map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r={3} fill={color} stroke="var(--surface-1)" strokeWidth={1.5} />
            ))}
          </g>
        </svg>
      )}
    </figure>
  );
}
