# CGNF Recommender — Frontend

React + Vite + Tailwind CSS v4 UI for the [CGNF Recommender](../README.md) project — the
"Network Lab" visual identity (light-first indigo/cyan palette, monospace display type,
graph-paper dot-grid texture, hexagonal node badges, top underline-tab nav).

Not meant to be run standalone — use [`../scripts/app.sh`](../scripts/app.sh) from the project
root, which starts this alongside the Flask backend it talks to:

```bash
../scripts/app.sh start
```

For frontend-only development (backend already running elsewhere):

```bash
npm install
npm run dev -- --port 5174
```

`VITE_API_BASE` (default `http://127.0.0.1:5003`) points the UI at the backend API.

## Structure

```
src/
├── api.js              REST client matching backend/app/api.py's routes
├── App.jsx              Shell: top nav, theme toggle, page routing
├── components/          icons.jsx, charts (BarChart/LineChart), StatTile, RunDetail, RecommendPanel, ...
├── pages/                DatasetsPage, TrainPage, ResultsPage, SensitivityPage, ValidationPage
├── hooks/useRunStream.js SSE-backed live run state
└── index.css             Design tokens + "Network Lab" theme (light/dark)
```
