const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5003";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.error) message = body.error;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(message);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  base: API_BASE,
  health: () => request("/api/health"),
  getDefaults: () => request("/api/config/defaults"),

  listDatasets: () => request("/api/datasets"),
  uploadCustomDataset: (moviesFile, ratingsFile, name) => {
    const form = new FormData();
    form.append("movies", moviesFile);
    form.append("ratings", ratingsFile);
    if (name) form.append("name", name);
    return request("/api/datasets/custom", { method: "POST", body: form });
  },

  createRun: (config) =>
    request("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  listRuns: () => request("/api/runs"),
  getRun: (runId) => request(`/api/runs/${runId}`),
  getLogs: (runId, after = 0) => request(`/api/runs/${runId}/logs?after=${after}`),
  streamUrl: (runId) => `${API_BASE}/api/runs/${runId}/stream`,

  sensitivity: (runId, H, tau) => request(`/api/runs/${runId}/sensitivity?H=${H}&tau=${tau}`),

  startMultiseed: (runId, seeds = 5, metric = "ndcg@20") =>
    request(`/api/runs/${runId}/multiseed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seeds, metric }),
    }),

  searchMovies: (runId, q) => request(`/api/runs/${runId}/movies/search?q=${encodeURIComponent(q)}`),

  recommend: (runId, { userId, topK = 10, method = "cgnf" }) => {
    const params = new URLSearchParams({ user_id: userId, top_k: topK, method });
    return request(`/api/runs/${runId}/recommend?${params.toString()}`);
  },
};
