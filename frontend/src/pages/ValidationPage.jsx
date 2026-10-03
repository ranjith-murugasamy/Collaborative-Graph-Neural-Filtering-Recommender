import { useState } from "react";
import { api } from "../api";
import { useRunStream } from "../hooks/useRunStream";
import { DistributionIcon } from "../components/icons";
import LogsConsole from "../components/LogsConsole";
import { num } from "../utils/format";

export default function ValidationPage({ activeRunId }) {
  const { run, logs } = useRunStream(activeRunId);
  const [seeds, setSeeds] = useState(5);
  const [metric, setMetric] = useState("ndcg@20");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState(null);
  const [started, setStarted] = useState(false);

  async function start(e) {
    e.preventDefault();
    setStarting(true);
    setError(null);
    try {
      await api.startMultiseed(activeRunId, seeds, metric);
      setStarted(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setStarting(false);
    }
  }

  if (!activeRunId) {
    return <p className="text-sm text-muted card p-4 max-w-md">Select a finished run on the Results tab first.</p>;
  }
  if (run && run.status !== "finished") {
    return <p className="text-sm text-muted card p-4 max-w-md">Run <span className="mono">{activeRunId}</span> is still {run.status} — statistical validation needs a finished run.</p>;
  }

  const result = run?.multiseed;
  const running = started && !result;

  return (
    <div className="flex flex-col gap-6 max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <DistributionIcon width={18} height={18} style={{ color: "var(--accent)" }} /> Statistical validation
        </h2>
        <p className="text-sm text-secondary mt-1">
          Re-trains CGNF and the strongest baseline from this run across multiple random seeds, reports
          mean &plusmn; standard deviation, and a paired significance test (paired t-test, plus Wilcoxon
          signed-rank when there are enough seeds) — not a single point estimate.
        </p>
      </div>

      {!result && (
        <form onSubmit={start} className="card p-5 flex flex-col gap-4 max-w-md">
          <label className="flex flex-col gap-1 text-xs text-secondary">
            Number of seeds
            <input className="field-input" type="number" min={2} max={10} value={seeds}
                   onChange={(e) => setSeeds(parseInt(e.target.value, 10) || 5)} />
          </label>
          <label className="flex flex-col gap-1 text-xs text-secondary">
            Metric
            <select className="field-input" value={metric} onChange={(e) => setMetric(e.target.value)}>
              <option value="ndcg@20">NDCG@20</option>
              <option value="recall@20">Recall@20</option>
              <option value="ndcg@50">NDCG@50</option>
              <option value="recall@50">Recall@50</option>
            </select>
          </label>
          <button type="submit" disabled={starting || running} className="btn btn-primary justify-center">
            {starting ? "Starting…" : running ? "Running…" : "Run multi-seed validation"}
          </button>
          {error && <p className="text-xs" style={{ color: "var(--status-critical)" }}>{error}</p>}
          <p className="text-xs text-muted">
            Re-trains both methods once per seed — this takes a few minutes per seed for CGNF vs. a
            neural baseline. Progress streams in the log below.
          </p>
        </form>
      )}

      {running && <LogsConsole logs={logs.filter((l) => l.message.includes("multiseed"))} />}

      {result && (
        <div className="card p-5 flex flex-col gap-4">
          <h3 className="text-sm font-semibold font-display">
            CGNF vs. {result.best_baseline} — {result.metric}, {result.aggregate.cgnf.seeds} seeds
          </h3>
          <div className="grid sm:grid-cols-2 gap-4">
            {["cgnf", result.best_baseline].map((key) => (
              <div key={key} className="rounded border border-[var(--border)] p-3">
                <div className="text-xs uppercase tracking-wide text-muted font-display">{key}</div>
                <div className="text-2xl font-semibold mono mt-1">
                  {num(result.aggregate[key].mean, 4)} <span className="text-sm text-muted">&plusmn; {num(result.aggregate[key].std, 4)}</span>
                </div>
                <div className="text-xs text-muted mt-1 mono">
                  {result.aggregate[key].values.map((v) => v.toFixed(3)).join(", ")}
                </div>
              </div>
            ))}
          </div>

          {result.significance?.available && (
            <div className="rounded border border-[var(--border)] p-3 text-xs mono flex flex-col gap-1">
              <div>mean difference (cgnf − baseline): {num(result.significance.mean_diff, 4)}</div>
              {result.significance.paired_t_test?.p_value !== undefined && (
                <div>paired t-test: t={num(result.significance.paired_t_test.statistic, 3)}, p={num(result.significance.paired_t_test.p_value, 4)}</div>
              )}
              {result.significance.wilcoxon?.p_value !== undefined ? (
                <div>Wilcoxon signed-rank: W={num(result.significance.wilcoxon.statistic, 3)}, p={num(result.significance.wilcoxon.p_value, 4)}</div>
              ) : (
                <div className="text-muted">Wilcoxon: {result.significance.wilcoxon?.reason}</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
