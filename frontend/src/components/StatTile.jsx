function formatValue(value, decimals) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (decimals !== undefined) return value.toFixed(decimals);
  if (Math.abs(value) >= 1000) return value.toLocaleString();
  return String(value);
}

export default function StatTile({ label, value, decimals, sublabel, accent }) {
  return (
    <div className="card px-4 py-3 flex flex-col gap-1 min-w-[9rem]" style={accent ? { borderTopColor: accent, borderTopWidth: 3 } : undefined}>
      <span className="text-xs text-muted uppercase tracking-wide font-display">{label}</span>
      <span className="text-2xl font-semibold mono">{formatValue(value, decimals)}</span>
      {sublabel && <span className="text-xs text-secondary">{sublabel}</span>}
    </div>
  );
}
