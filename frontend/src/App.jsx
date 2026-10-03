import { useState } from "react";
import DatasetsPage from "./pages/DatasetsPage";
import TrainPage from "./pages/TrainPage";
import ResultsPage from "./pages/ResultsPage";
import SensitivityPage from "./pages/SensitivityPage";
import ValidationPage from "./pages/ValidationPage";
import ThemeToggle from "./components/ThemeToggle";
import Footer from "./components/Footer";
import { HexNodeIcon, DatabaseIcon, PlayCircleIcon, ActivityIcon, GridIcon, DistributionIcon } from "./components/icons";

const NAV = [
  { key: "datasets", label: "Datasets", icon: DatabaseIcon },
  { key: "train", label: "Train", icon: PlayCircleIcon },
  { key: "results", label: "Results", icon: ActivityIcon },
  { key: "sensitivity", label: "Sensitivity", icon: GridIcon },
  { key: "validation", label: "Validation", icon: DistributionIcon },
];

export default function App() {
  const [tab, setTab] = useState("datasets");
  const [dataset, setDataset] = useState(null);
  const [activeRunId, setActiveRunId] = useState(null);

  function handleRunCreated(runId) {
    setActiveRunId(runId);
    setTab("results");
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-[var(--border)]" style={{ background: "var(--surface-1)" }}>
        <div className="px-6 pt-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <span className="node-badge" style={{ background: "var(--accent)", color: "#fff" }}>
              <HexNodeIcon width={18} height={18} />
            </span>
            <div>
              <h1 className="text-lg leading-none">CGNF Recommender</h1>
              <p className="text-xs text-muted mt-1">Collaborative Graph Neural Filtering — reproduced &amp; benchmarked</p>
            </div>
          </div>
          <ThemeToggle />
        </div>
        <nav className="px-6 flex gap-6 overflow-x-auto mt-3">
          {NAV.map(({ key, label, icon: Icon }) => (
            <button
              key={key} type="button" onClick={() => setTab(key)}
              className={`tab-link flex items-center gap-1.5 shrink-0 ${tab === key ? "active" : ""}`}
            >
              <Icon width={15} height={15} />
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="px-6 py-6 flex-1">
        {tab === "datasets" && <DatasetsPage selectedDataset={dataset} onSelectDataset={(d) => { setDataset(d); setTab("train"); }} />}
        {tab === "train" && <TrainPage dataset={dataset} onRunCreated={handleRunCreated} />}
        {tab === "results" && <ResultsPage activeRunId={activeRunId} onSelectRun={setActiveRunId} />}
        {tab === "sensitivity" && <SensitivityPage activeRunId={activeRunId} />}
        {tab === "validation" && <ValidationPage activeRunId={activeRunId} />}
      </main>

      <Footer />
    </div>
  );
}
