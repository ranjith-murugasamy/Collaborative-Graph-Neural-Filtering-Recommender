import { useState } from "react";
import { useRunStream } from "../hooks/useRunStream";
import StatusBadge from "./StatusBadge";
import StatTile from "./StatTile";
import LineChart from "./LineChart";
import BarChart from "./BarChart";
import LogsConsole from "./LogsConsole";
import RecommendPanel from "./RecommendPanel";
import { AlertCircleIcon, LayersIcon } from "./icons";
import { compactNumber, pct, num } from "../utils/format";

const METHOD_COLOR = {
  mostpop: "var(--m-mostpop)", itemknn: "var(--m-itemknn)", bprmf: "var(--m-bprmf)",
  lightgcn: "var(--m-lightgcn)", cgnf: "var(--m-cgnf)",
};
const METHOD_ORDER = ["mostpop", "itemknn", "bprmf", "lightgcn", "cgnf"];
const METRIC_TABS = [
  { key: "ndcg@20", label: "NDCG@20" },
  { key: "recall@20", label: "Recall@20" },
  { key: "ndcg@50", label: "NDCG@50" },
  { key: "recall@50", label: "Recall@50" },
];
const ABLATION_ORDER = ["neither", "unweighted_modularity", "no_edge_reduction", "full"];
// Short chart-bar labels — the full labels ("CGNF − both (plain community
// CF)") overflow BarChart's fixed label column; the long form still shows
// in the legend strip below the chart and in Table view.
const ABLATION_SHORT_LABEL = {
  neither: "− both",
  unweighted_modularity: "− weighted mod.",
  no_edge_reduction: "− edge reduction",
  full: "Full CGNF",
};

export default function RunDetail({ runId }) {
  const { run, logs } = useRunStream(runId);
  const [metricTab, setMetricTab] = useState("ndcg@20");

  if (!run) return <p className="text-sm text-muted">Loading run…</p>;
  const isTerminal = run.status === "finished" || run.status === "error";

  const barItems = run.results
    ? METHOD_ORDER.filter((k) => run.results[k]).map((k) => ({
        key: k, label: run.results[k].label, value: run.results[k].metrics[metricTab],
        color: METHOD_COLOR[k],
      }))
    : [];

  const ablationItems = run.cgnf_ablation
    ? ABLATION_ORDER.filter((k) => run.cgnf_ablation[k]).map((k) => ({
        key: k, label: ABLATION_SHORT_LABEL[k], fullLabel: run.cgnf_ablation[k].label,
        value: run.cgnf_ablation[k].metrics[metricTab],
        color: k === "full" ? "var(--m-cgnf)" : "var(--baseline)",
      }))
    : [];

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold mono">{run.id}</h2>
          <StatusBadge status={run.status} />
        </div>
        <span className="text-xs text-muted font-display">{run.current_stage}</span>
      </div>

      {run.error && (
        <div className="card p-4 flex items-start gap-2" style={{ borderColor: "var(--status-critical)" }}>
          <AlertCircleIcon width={16} height={16} style={{ color: "var(--status-critical)" }} />
          <p className="text-sm">{run.error}</p>
        </div>
      )}

      {run.dataset_info?.num_users && (
        <div className="flex flex-wrap gap-3">
          <StatTile label="Users" value={run.dataset_info.num_users} />
          <StatTile label="Items" value={run.dataset_info.num_items} />
          <StatTile label="Ratings" value={run.dataset_info.num_ratings} />
          <StatTile label="Eval users" value={run.dataset_info.num_test_users} />
          <StatTile label="Sparsity" value={run.dataset_info.sparsity} decimals={4} sublabel={pct(run.dataset_info.sparsity)} />
        </div>
      )}

      {!isTerminal && (
        <div className="card p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Training in progress…</span>
          </div>
          <LogsConsole logs={logs} />
        </div>
      )}

      {(run.loss_history?.bprmf?.length > 0 || run.loss_history?.lightgcn?.length > 0) && (
        <div className="grid sm:grid-cols-2 gap-4">
          <LineChart title="BPR-MF — training loss" data={run.loss_history.bprmf} color="var(--m-bprmf)" />
          <LineChart title="LightGCN — training loss" data={run.loss_history.lightgcn} color="var(--m-lightgcn)" />
        </div>
      )}

      {run.loss_history?.cgnf_h_sweep?.length > 0 && (
        <LineChart title="CGNF — modularity vs. community count H" data={run.loss_history.cgnf_h_sweep}
                    xKey="H" yKey="modularity" color="var(--m-cgnf)" />
      )}

      {run.results && (
        <>
          <div className="card p-4">
            <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
              <h3 className="text-sm font-semibold font-display">5-way comparison</h3>
              <div className="flex gap-1">
                {METRIC_TABS.map((t) => (
                  <button
                    key={t.key} type="button" onClick={() => setMetricTab(t.key)}
                    className="btn btn-ghost text-xs px-2 py-1"
                    style={metricTab === t.key ? { background: "var(--surface-2)", color: "var(--text-primary)" } : undefined}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>
            <BarChart title={METRIC_TABS.find((t) => t.key === metricTab)?.label} items={barItems} emphasisKey="cgnf" />
          </div>

          {ablationItems.length > 0 && (
            <div className="card p-4">
              <div className="flex items-center gap-2 mb-3">
                <LayersIcon width={16} height={16} style={{ color: "var(--accent)" }} />
                <h3 className="text-sm font-semibold font-display">CGNF internal ablation</h3>
              </div>
              <BarChart title={`CGNF pipeline — ${METRIC_TABS.find((t) => t.key === metricTab)?.label}`} items={ablationItems} emphasisKey="full" />
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 pt-3 border-t border-[var(--gridline)]">
                {ABLATION_ORDER.map((k) => run.cgnf_ablation[k] && (
                  <div key={k} className="text-center">
                    <div className="text-[10px] uppercase tracking-wide text-muted">{run.cgnf_ablation[k].label}</div>
                    <div className="text-xs mono mt-1">H*={run.cgnf_ablation[k].H} &middot; {run.cgnf_ablation[k].seconds.toFixed(1)}s</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="grid sm:grid-cols-2 gap-4">
            <div className="card p-4">
              <h3 className="text-sm font-semibold font-display mb-3">Diversity</h3>
              <table className="w-full text-xs mono">
                <thead>
                  <tr className="text-left text-muted">
                    <th className="pr-3 py-1">Method</th><th className="pr-3 py-1">Coverage</th>
                    <th className="pr-3 py-1">Novelty</th><th className="pr-3 py-1">ILD</th>
                  </tr>
                </thead>
                <tbody>
                  {METHOD_ORDER.filter((k) => run.results[k]?.diversity).map((k) => (
                    <tr key={k} className="border-t border-[var(--gridline)]">
                      <td className="pr-3 py-1.5" style={{ color: METHOD_COLOR[k] }}>{run.results[k].label}</td>
                      <td className="pr-3 py-1.5">{pct(run.results[k].diversity.coverage)}</td>
                      <td className="pr-3 py-1.5">{num(run.results[k].diversity.novelty, 2)}</td>
                      <td className="pr-3 py-1.5">{num(run.results[k].diversity.intra_list_diversity, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {run.cgnf_ablation?.full && (
              <div className="card p-4">
                <h3 className="text-sm font-semibold font-display mb-3">Edge reduction efficiency</h3>
                <div className="flex items-center justify-between gap-4">
                  <StatTile label="Edges before" value={run.cgnf_ablation.full.edges_before} />
                  <StatTile label="Edges after" value={run.cgnf_ablation.full.edges_after} />
                  <StatTile label="Reduction"
                    value={1 - run.cgnf_ablation.full.edges_after / run.cgnf_ablation.full.edges_before}
                    decimals={4}
                    sublabel={pct(1 - run.cgnf_ablation.full.edges_after / run.cgnf_ablation.full.edges_before)} />
                </div>
                <p className="text-xs text-muted mt-3">
                  Full CGNF (with reduction) took {run.cgnf_ablation.full.seconds.toFixed(1)}s;
                  {" "}without it, {run.cgnf_ablation.no_edge_reduction?.seconds.toFixed(1)}s —
                  measured wall-clock, not asserted.
                </p>
              </div>
            )}
          </div>

          <RecommendPanel runId={run.id} numUsers={run.dataset_info?.num_users} />
        </>
      )}

      {isTerminal && (
        <details className="card p-4">
          <summary className="text-sm font-medium cursor-pointer font-display">Run logs ({compactNumber(run.log_count)})</summary>
          <div className="mt-3"><LogsConsole logs={logs} maxHeight={320} /></div>
        </details>
      )}
    </div>
  );
}
