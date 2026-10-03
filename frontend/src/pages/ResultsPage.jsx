import { useEffect, useState } from "react";
import { api } from "../api";
import RunsHistory from "../components/RunsHistory";
import RunDetail from "../components/RunDetail";

export default function ResultsPage({ activeRunId, onSelectRun }) {
  const [runs, setRuns] = useState([]);

  useEffect(() => {
    function refresh() {
      api.listRuns().then((data) => setRuns(data.sort((a, b) => b.created_at - a.created_at))).catch(() => {});
    }
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!activeRunId && runs.length > 0) onSelectRun(runs[0].id);
  }, [runs, activeRunId, onSelectRun]);

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      <aside className="lg:w-72 shrink-0">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted mb-2 px-1 font-display">History</h3>
        <RunsHistory runs={runs} selectedId={activeRunId} onSelect={onSelectRun} />
      </aside>
      <main className="flex-1 min-w-0">
        {activeRunId ? <RunDetail runId={activeRunId} /> : <p className="text-sm text-muted">Select a run to see its results.</p>}
      </main>
    </div>
  );
}
