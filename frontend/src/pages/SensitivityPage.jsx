import { useEffect, useState } from "react";
import { api } from "../api";
import { GridIcon } from "../components/icons";
import { num } from "../utils/format";

export default function SensitivityPage({ activeRunId }) {
  const [run, setRun] = useState(null);
  const [H, setH] = useState(16);
  const [tau, setTau] = useState(0.7);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cells, setCells] = useState([]);   // computed cells this session, most recent first

  useEffect(() => {
    if (!activeRunId) { setRun(null); return; }
    api.getRun(activeRunId).then(setRun).catch(() => {});
  }, [activeRunId]);

  async function compute(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const cell = await api.sensitivity(activeRunId, H, tau);
      setCells((prev) => [{ ...cell, key: `${H}:${tau}:${Date.now()}` }, ...prev].slice(0, 12));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (!activeRunId) {
    return <p className="text-sm text-muted card p-4 max-w-md">Select a finished run on the Results tab first.</p>;
  }
  if (run && run.status !== "finished") {
    return <p className="text-sm text-muted card p-4 max-w-md">Run <span className="mono">{activeRunId}</span> is still {run.status} — the sensitivity sweep needs a finished run (it reuses that run's dataset split).</p>;
  }

  return (
    <div className="flex flex-col gap-6 max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold flex items-center gap-2"><GridIcon width={18} height={18} style={{ color: "var(--accent)" }} /> Sensitivity sweep</h2>
        <p className="text-sm text-secondary mt-1">
          CGNF's two main hyperparameters — target community count <span className="mono">H</span> and the
          strong-link threshold <span className="mono">τ</span> — re-run community detection for exactly the
          cell you pick (not instant like a fusion weight: community structure genuinely changes with these inputs,
          see specs/plan.md &sect;3). Each cell is cached by run.
        </p>
      </div>

      <form onSubmit={compute} className="card p-5 flex flex-col gap-4">
        <div className="grid sm:grid-cols-2 gap-5">
          <label className="flex flex-col gap-2 text-xs text-secondary">
            <span className="flex justify-between"><span>Community count H</span><span className="mono text-primary">{H}</span></span>
            <input type="range" min={2} max={128} value={H} onChange={(e) => setH(parseInt(e.target.value, 10))} />
          </label>
          <label className="flex flex-col gap-2 text-xs text-secondary">
            <span className="flex justify-between"><span>Strong-link threshold τ</span><span className="mono text-primary">{tau.toFixed(2)}</span></span>
            <input type="range" min={0.1} max={1} step={0.05} value={tau} onChange={(e) => setTau(parseFloat(e.target.value))} />
          </label>
        </div>
        <button type="submit" disabled={loading} className="btn btn-primary justify-center">
          {loading ? "Computing…" : "Compute this cell"}
        </button>
        {error && <p className="text-xs" style={{ color: "var(--status-critical)" }}>{error}</p>}
      </form>

      {cells.length > 0 && (
        <div className="card p-4 overflow-x-auto">
          <h3 className="text-sm font-semibold font-display mb-3">Computed cells (this session, most recent first)</h3>
          <table className="w-full text-xs mono">
            <thead>
              <tr className="text-left text-muted">
                <th className="pr-4 py-1">H</th><th className="pr-4 py-1">&tau;</th>
                <th className="pr-4 py-1">Modularity</th><th className="pr-4 py-1">NDCG@20</th>
                <th className="pr-4 py-1">Recall@20</th><th className="pr-4 py-1">Edges kept</th>
              </tr>
            </thead>
            <tbody>
              {cells.map((c) => (
                <tr key={c.key} className="border-t border-[var(--gridline)]">
                  <td className="pr-4 py-1.5">{c.H}</td>
                  <td className="pr-4 py-1.5">{c.tau.toFixed(2)}</td>
                  <td className="pr-4 py-1.5">{num(c.modularity, 4)}</td>
                  <td className="pr-4 py-1.5" style={{ color: "var(--accent)" }}>{num(c.metrics["ndcg@20"], 4)}</td>
                  <td className="pr-4 py-1.5">{num(c.metrics["recall@20"], 4)}</td>
                  <td className="pr-4 py-1.5">{c.edges_after}/{c.edges_before}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
