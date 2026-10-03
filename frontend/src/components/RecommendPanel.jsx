import { useState } from "react";
import { api } from "../api";
import { SearchUserIcon, SparkleIcon, HexNodeIcon } from "./icons";

const METHODS = [
  { key: "cgnf", label: "CGNF" },
  { key: "mostpop", label: "MostPopular" },
  { key: "itemknn", label: "Item-kNN" },
  { key: "bprmf", label: "BPR-MF" },
  { key: "lightgcn", label: "LightGCN" },
];

export default function RecommendPanel({ runId, numUsers }) {
  const [userId, setUserId] = useState(1);
  const [method, setMethod] = useState("cgnf");
  const [recs, setRecs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleRecommend(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await api.recommend(runId, { userId, topK: 10, method });
      setRecs(data.items);
    } catch (err) {
      setError(err.message);
      setRecs(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card p-4 flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <SparkleIcon width={16} height={16} style={{ color: "var(--accent)" }} />
        <h3 className="text-sm font-semibold font-display">Recommend for a user</h3>
      </div>

      <form onSubmit={handleRecommend} className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1 text-xs text-secondary">
            <span className="flex items-center gap-1"><SearchUserIcon width={13} height={13} /> User ID (1–{numUsers || "?"})</span>
            <input
              className="field-input" type="number" min={1} max={numUsers || undefined}
              value={userId} onChange={(e) => setUserId(parseInt(e.target.value, 10) || 1)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-secondary">
            Method
            <select className="field-input" value={method} onChange={(e) => setMethod(e.target.value)}>
              {METHODS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </label>
        </div>

        <button type="submit" disabled={loading} className="btn btn-primary justify-center">
          {loading ? "Scoring…" : "Get recommendations"}
        </button>
      </form>

      {error && <p className="text-xs" style={{ color: "var(--status-critical)" }}>{error}</p>}

      {recs && (
        <ul className="flex flex-col gap-2">
          {recs.map((item, idx) => (
            <li key={item.item_index} className="hover-card rounded p-2.5 flex items-center gap-3">
              <span className="w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-semibold shrink-0 mono" style={{ background: "var(--surface-2)", color: "var(--accent)" }}>
                {idx + 1}
              </span>
              <HexNodeIcon width={15} height={15} className="text-muted shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium truncate">{item.title}</p>
                <p className="text-xs text-muted truncate">{item.genres.join(", ")}</p>
              </div>
              <span className="text-[11px] mono text-secondary shrink-0">{item.score.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
