import StatusBadge from "./StatusBadge";
import { relativeTime } from "../utils/format";
import { ArrowRightIcon } from "./icons";

export default function RunsHistory({ runs, selectedId, onSelect }) {
  if (!runs || runs.length === 0) {
    return <p className="text-sm text-muted card p-4 text-center">No training runs yet — start one from the Train tab.</p>;
  }
  return (
    <div className="card divide-y divide-[var(--gridline)] overflow-hidden">
      {runs.map((run) => (
        <button
          key={run.id}
          type="button"
          onClick={() => onSelect(run.id)}
          className="hover-card w-full flex items-center justify-between gap-3 px-4 py-3 text-left"
          style={run.id === selectedId ? { background: "var(--surface-hover)" } : undefined}
        >
          <div className="flex flex-col gap-0.5 min-w-0">
            <span className="text-sm font-medium mono truncate">{run.id}</span>
            <span className="text-xs text-muted">
              {run.dataset_info?.num_users ? `${run.dataset_info.num_users.toLocaleString()} users` : run.config?.dataset?.id} &middot; {relativeTime(run.created_at)}
            </span>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <StatusBadge status={run.status} />
            <ArrowRightIcon width={14} height={14} className="text-muted" />
          </div>
        </button>
      ))}
    </div>
  );
}
