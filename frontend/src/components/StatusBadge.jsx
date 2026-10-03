const STATUS = {
  queued: { color: "var(--text-muted)", icon: "○", label: "Queued" },
  running: { color: "var(--status-warning)", icon: "●", label: "Running" },
  finished: { color: "var(--status-good)", icon: "✓", label: "Finished" },
  error: { color: "var(--status-critical)", icon: "✕", label: "Error" },
};

export default function StatusBadge({ status }) {
  const s = STATUS[status] || STATUS.queued;
  return (
    <span
      className="inline-flex items-center gap-1.5 text-[11px] font-semibold font-display px-2.5 py-1 rounded"
      style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
    >
      <span style={{ color: s.color }}>{s.icon}</span>
      <span className="text-secondary">{s.label}</span>
    </span>
  );
}
