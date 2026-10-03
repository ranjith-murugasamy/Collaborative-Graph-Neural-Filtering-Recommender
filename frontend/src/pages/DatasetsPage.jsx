import { useEffect, useState } from "react";
import { api } from "../api";
import DatasetCard from "../components/DatasetCard";
import UploadDataset from "../components/UploadDataset";

export default function DatasetsPage({ selectedDataset, onSelectDataset }) {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  function refresh() {
    setLoading(true);
    api.listDatasets().then((d) => { setDatasets(d); setError(null); }).catch((e) => setError(e.message)).finally(() => setLoading(false));
  }

  useEffect(refresh, []);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-xl font-semibold">Datasets</h2>
        <p className="text-sm text-secondary mt-1 max-w-2xl">
          MovieLens 1M ships as the default sample (6,040 users &middot; 3,883 movies &middot; 1,000,209 ratings) —
          the same dataset the paper itself evaluates on — or upload your own movies/ratings CSV pair.
        </p>
      </div>

      {loading && <p className="text-sm text-muted">Loading datasets…</p>}
      {error && <p className="text-sm" style={{ color: "var(--status-critical)" }}>{error}</p>}

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {datasets.map((d) => (
          <DatasetCard key={d.id} dataset={d} selected={selectedDataset?.id === d.id} onSelect={onSelectDataset} />
        ))}
      </div>

      <div className="max-w-md">
        <UploadDataset onUploaded={refresh} />
      </div>
    </div>
  );
}
