import { useEffect, useState } from "react";
import { api } from "../api";
import { GithubIcon, PulseDotIcon } from "./icons";

export default function Footer() {
  const [connected, setConnected] = useState(null);

  useEffect(() => {
    let cancelled = false;
    function check() {
      api.health().then(() => !cancelled && setConnected(true)).catch(() => !cancelled && setConnected(false));
    }
    check();
    const id = setInterval(check, 15000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const statusColor = connected === null ? "var(--text-muted)" : connected ? "var(--status-good)" : "var(--status-critical)";
  const statusLabel = connected === null ? "checking…" : connected ? "connected" : "unreachable";

  return (
    <footer className="mt-10 border-t border-[var(--border)]">
      <div className="px-6 py-4 flex flex-wrap items-center justify-between gap-4 text-xs">
        <span className="flex items-center gap-2 text-muted font-display">
          <span className="node-badge" style={{ background: "var(--accent)", color: "#fff" }}>N</span>
          CGNF Recommender — 5-way benchmark, reproduced from the paper's own equations.
        </span>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-muted" title={`${api.base} — ${statusLabel}`}>
            <PulseDotIcon style={{ color: statusColor }} />
            Backend {statusLabel}
          </span>
          <span className="flex items-center gap-1.5 text-secondary">
            <GithubIcon width={14} height={14} />
            specs/spec.md
          </span>
        </div>
      </div>
    </footer>
  );
}
