import { useRef, useState } from "react";
import { api } from "../api";
import { UploadIcon } from "./icons";

export default function UploadDataset({ onUploaded }) {
  const [name, setName] = useState("");
  const [movies, setMovies] = useState(null);
  const [ratings, setRatings] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const moviesRef = useRef(null);
  const ratingsRef = useRef(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!movies || !ratings) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.uploadCustomDataset(movies, ratings, name || undefined);
      setName(""); setMovies(null); setRatings(null);
      if (moviesRef.current) moviesRef.current.value = "";
      if (ratingsRef.current) ratingsRef.current.value = "";
      onUploaded?.(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="card p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2 text-sm font-medium font-display">
        <UploadIcon width={16} height={16} style={{ color: "var(--accent)" }} />
        Upload a custom dataset
      </div>
      <p className="text-xs text-muted">
        Two CSV files: movies (<code className="mono">movie_id, title, genres</code>) and ratings
        (<code className="mono">user_id, movie_id, rating[, timestamp]</code>).
      </p>
      <label className="flex flex-col gap-1 text-xs text-secondary">
        Display name (optional)
        <input className="field-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="My catalog" />
      </label>
      <label className="flex flex-col gap-1 text-xs text-secondary">
        Movies CSV
        <input ref={moviesRef} className="field-input" type="file" accept=".csv" onChange={(e) => setMovies(e.target.files?.[0] || null)} />
      </label>
      <label className="flex flex-col gap-1 text-xs text-secondary">
        Ratings CSV
        <input ref={ratingsRef} className="field-input" type="file" accept=".csv" onChange={(e) => setRatings(e.target.files?.[0] || null)} />
      </label>
      {error && <p className="text-xs" style={{ color: "var(--status-critical)" }}>{error}</p>}
      <button type="submit" disabled={!movies || !ratings || busy} className="btn btn-secondary justify-center">
        {busy ? "Uploading…" : "Upload"}
      </button>
    </form>
  );
}
