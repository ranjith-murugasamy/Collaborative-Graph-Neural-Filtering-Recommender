import { useState } from "react";
import { PlayCircleIcon, SlidersIcon } from "./icons";

const FIELD_GROUPS = [
  {
    title: "Split & relevance",
    fields: [
      { key: "holdout_k", label: "Holdout items / user", type: "int", min: 1, max: 5 },
      { key: "min_interactions", label: "Min. interactions / user", type: "int", min: 1, max: 50 },
      { key: "relevance_threshold", label: "Relevance threshold (rating ≥)", type: "float", min: 1, max: 5, step: 0.5 },
    ],
  },
  {
    title: "CGNF (proposed method)",
    fields: [
      { key: "cgnf_tau", label: "Strong-link threshold τ", type: "float", min: 0.1, max: 1, step: 0.05 },
      { key: "cgnf_iters", label: "Community-detection iterations", type: "int", min: 2, max: 40 },
      { key: "cgnf_max_h", label: "Max community count (grid cap)", type: "int", min: 4, max: 128 },
      { key: "cgnf_top_k_neighbors", label: "Neighbors used per prediction", type: "int", min: 5, max: 200 },
    ],
  },
  {
    title: "Item-kNN baseline",
    fields: [
      { key: "itemknn_k", label: "Neighbors per item", type: "int", min: 5, max: 200 },
    ],
  },
  {
    title: "BPR-MF baseline",
    fields: [
      { key: "bprmf_embedding_dim", label: "Embedding dim", type: "int", min: 4, max: 128 },
      { key: "bprmf_epochs", label: "Epochs", type: "int", min: 1, max: 40 },
      { key: "bprmf_lr", label: "Learning rate", type: "float", min: 0.001, max: 0.5, step: 0.001 },
    ],
  },
  {
    title: "LightGCN baseline",
    fields: [
      { key: "lightgcn_embedding_dim", label: "Embedding dim", type: "int", min: 4, max: 128 },
      { key: "lightgcn_num_layers", label: "Propagation layers K", type: "int", min: 1, max: 6 },
      { key: "lightgcn_epochs", label: "Epochs", type: "int", min: 1, max: 40 },
      { key: "lightgcn_lr", label: "Learning rate", type: "float", min: 0.001, max: 0.5, step: 0.001 },
    ],
  },
];

export default function RunForm({ defaults, dataset, onSubmit, submitting }) {
  const [config, setConfig] = useState(() => ({ ...defaults }));

  function setField(key, type, raw) {
    const value = type === "int" ? parseInt(raw, 10) : parseFloat(raw);
    setConfig((prev) => ({ ...prev, [key]: Number.isNaN(value) ? prev[key] : value }));
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!dataset) return;
    const { method_labels, cgnf_ablation_presets, ...rest } = config;
    onSubmit({ ...rest, dataset: { source: dataset.source === "custom" ? "custom" : "default", id: dataset.id } });
  }

  return (
    <form onSubmit={handleSubmit} className="card p-5 flex flex-col gap-5">
      <div className="flex items-center gap-2">
        <SlidersIcon width={16} height={16} style={{ color: "var(--accent)" }} />
        <h2 className="text-sm font-semibold font-display">Training configuration</h2>
      </div>

      {FIELD_GROUPS.map((group) => (
        <fieldset key={group.title} className="flex flex-col gap-2">
          <legend className="text-xs font-medium text-secondary uppercase tracking-wide mb-1 font-display">{group.title}</legend>
          <div className="grid grid-cols-2 gap-3">
            {group.fields.map((f) => (
              <label key={f.key} className="flex flex-col gap-1 text-xs text-secondary">
                {f.label}
                <input
                  className="field-input"
                  type="number"
                  min={f.min}
                  max={f.max}
                  step={f.step || 1}
                  value={config[f.key]}
                  onChange={(e) => setField(f.key, f.type, e.target.value)}
                />
              </label>
            ))}
          </div>
        </fieldset>
      ))}

      <button type="submit" disabled={!dataset || submitting} className="btn btn-primary justify-center">
        <PlayCircleIcon width={16} height={16} />
        {submitting ? "Starting…" : "Start training run"}
      </button>
      {!dataset && <p className="text-xs text-muted text-center">Select a dataset above to enable training.</p>}
    </form>
  );
}
