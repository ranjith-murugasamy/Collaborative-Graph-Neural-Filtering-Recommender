import { useEffect, useState } from "react";
import { api } from "../api";

/** Merges new log entries by `seq`, de-duplicating — the SSE stream
 * replays the full backlog from seq 0 on connect, which would otherwise
 * double every line for an already-finished run. */
function mergeLogs(prev, incoming) {
  const bySeq = new Map(prev.map((e) => [e.seq, e]));
  for (const e of incoming) bySeq.set(e.seq, e);
  return [...bySeq.values()].sort((a, b) => a.seq - b.seq);
}

export function useRunStream(runId) {
  const [run, setRun] = useState(null);
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    setRun(null);
    setLogs([]);
    if (!runId) return undefined;
    let cancelled = false;

    api.getRun(runId).then((d) => !cancelled && setRun(d)).catch(() => {});
    api.getLogs(runId, 0).then((d) => !cancelled && setLogs((prev) => mergeLogs(prev, d.logs))).catch(() => {});

    const source = new EventSource(api.streamUrl(runId));
    source.addEventListener("logs", (event) => {
      setLogs((prev) => mergeLogs(prev, JSON.parse(event.data)));
    });
    source.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      setRun(data);
      if (data.status === "finished" || data.status === "error") source.close();
    });
    source.onerror = () => {
      if (run && (run.status === "finished" || run.status === "error")) source.close();
    };

    return () => {
      cancelled = true;
      source.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  return { run, logs };
}
