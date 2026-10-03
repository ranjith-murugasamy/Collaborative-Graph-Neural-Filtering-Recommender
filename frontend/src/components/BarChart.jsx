import { useState } from "react";

/** Horizontal bar chart comparing the 5 methods on one metric — the
 * "emphasis" form: CGNF (the proposed method) is drawn in the accent
 * color, every baseline in a per-method hue from index.css's --m-* tokens
 * so each method keeps the same color across every chart in the app. */
export default function BarChart({ title, items, emphasisKey = "cgnf", valueFormat = (v) => v.toFixed(4), height }) {
  const [showTable, setShowTable] = useState(false);
  const max = Math.max(...items.map((i) => i.value), 1e-9);
  const rowH = 32;
  const chartHeight = height || items.length * rowH + 20;

  return (
    <figure className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <figcaption className="text-sm font-medium text-secondary font-display">{title}</figcaption>
        <button type="button" onClick={() => setShowTable((v) => !v)} className="text-xs text-secondary border border-[var(--border)] rounded px-2 py-0.5 hover:bg-[var(--surface-2)]">
          {showTable ? "Chart view" : "Table view"}
        </button>
      </div>

      {showTable ? (
        <table className="w-full text-xs mono">
          <thead>
            <tr className="text-left text-muted">
              <th className="pr-4 py-1">Method</th>
              <th className="pr-4 py-1">Value</th>
            </tr>
          </thead>
          <tbody>
            {items.map((i) => (
              <tr key={i.key} className="border-t border-[var(--gridline)]">
                <td className="pr-4 py-1">{i.fullLabel || i.label}</td>
                <td className="pr-4 py-1">{valueFormat(i.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <svg viewBox={`0 0 560 ${chartHeight}`} className="w-full" role="img" aria-label={title}>
          {items.map((item, idx) => {
            const y = idx * rowH;
            const barMaxW = 560 - 150 - 64;
            const w = Math.max(2, (item.value / max) * barMaxW);
            const isEmphasis = item.key === emphasisKey;
            const color = item.color || "var(--baseline)";
            return (
              <g key={item.key}>
                <text x={0} y={y + 19} fontSize={11} fontFamily="var(--font-display)" fill={isEmphasis ? "var(--text-primary)" : "var(--text-secondary)"} fontWeight={isEmphasis ? 700 : 500}>
                  {item.label}
                </text>
                <rect x={150} y={y + 7} width={barMaxW} height={15} rx={2} fill="var(--gridline)" />
                <rect x={150} y={y + 7} width={w} height={15} rx={2} fill={color} opacity={isEmphasis ? 1 : 0.75} />
                {isEmphasis && <rect x={150} y={y + 7} width={2} height={15} fill="var(--text-primary)" opacity={0.3} />}
                <text x={150 + barMaxW + 8} y={y + 19} fontSize={11} fontFamily="var(--font-display)" fill={isEmphasis ? "var(--accent)" : "var(--text-secondary)"} fontWeight={isEmphasis ? 700 : 400}>
                  {valueFormat(item.value)}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </figure>
  );
}
