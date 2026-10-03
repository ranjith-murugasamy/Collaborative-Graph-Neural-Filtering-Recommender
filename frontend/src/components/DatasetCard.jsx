import { DatabaseIcon } from "./icons";
import { compactNumber } from "../utils/format";

export default function DatasetCard({ dataset, selected, onSelect }) {
  return (
    <button
      type="button"
      onClick={() => onSelect(dataset)}
      className="card hover-card p-4 text-left flex flex-col gap-3 w-full"
      style={selected ? { borderColor: "var(--accent)", boxShadow: "0 0 0 1px var(--accent)" } : undefined}
    >
      <div className="flex items-center justify-between">
        <span className="node-badge" style={{ background: "var(--surface-2)", color: "var(--accent)" }}>
          <DatabaseIcon width={16} height={16} />
        </span>
        <span className="chip">{dataset.source === "custom" ? "Custom" : "Default"}</span>
      </div>
      <div>
        <h3 className="text-sm font-semibold font-display">{dataset.label}</h3>
        <p className="text-xs text-muted mt-0.5">{dataset.id}</p>
      </div>
      <dl className="grid grid-cols-3 gap-2 text-center">
        <div>
          <dt className="text-[10px] uppercase tracking-wide text-muted">Users</dt>
          <dd className="text-sm font-medium mono">{compactNumber(dataset.num_users)}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-wide text-muted">Items</dt>
          <dd className="text-sm font-medium mono">{compactNumber(dataset.num_items)}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-wide text-muted">Ratings</dt>
          <dd className="text-sm font-medium mono">{compactNumber(dataset.num_ratings)}</dd>
        </div>
      </dl>
    </button>
  );
}
