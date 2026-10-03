import { useEffect, useState } from "react";
import { api } from "../api";
import RunForm from "../components/RunForm";
import { DatabaseIcon } from "../components/icons";

export default function TrainPage({ dataset, onRunCreated }) {
  const [defaults, setDefaults] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getDefaults().then(setDefaults).catch((e) => setError(e.message));
  }, []);

  async function handleSubmit(config) {
    setSubmitting(true);
    setError(null);
    try {
      const run = await api.createRun(config);
      onRunCreated(run.run_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex flex-col gap-6 max-w-2xl">
      <div>
        <h2 className="text-xl font-semibold">Train</h2>
        <p className="text-sm text-secondary mt-1">
          Trains all 5 methods — MostPopular, Item-kNN, BPR-MF, LightGCN, and CGNF (with its 4-way internal
          ablation) — on one identical held-out split, then evaluates Recall/NDCG@{"{20,50}"} and diversity for
          every method. A full run takes a few minutes on CPU.
        </p>
      </div>

      {!dataset && (
        <div className="card p-4 flex items-center gap-3">
          <DatabaseIcon width={18} height={18} style={{ color: "var(--accent)" }} />
          <p className="text-sm">Pick a dataset on the <strong>Datasets</strong> tab first.</p>
        </div>
      )}

      {dataset && (
        <div className="chip w-fit">
          Training on <strong className="mx-1">{dataset.label}</strong>
        </div>
      )}

      {error && <p className="text-sm" style={{ color: "var(--status-critical)" }}>{error}</p>}

      {defaults && <RunForm defaults={defaults} dataset={dataset} onSubmit={handleSubmit} submitting={submitting} />}
    </div>
  );
}
