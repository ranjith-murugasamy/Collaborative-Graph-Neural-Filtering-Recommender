import { useEffect, useRef } from "react";

export default function LogsConsole({ logs, maxHeight = 240 }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [logs.length]);

  return (
    <div
      ref={scrollRef}
      className="rounded px-3 py-2 text-xs mono overflow-y-auto"
      style={{ background: "var(--surface-2)", maxHeight, border: "1px solid var(--border)" }}
    >
      {logs.length === 0 ? (
        <p className="text-muted">Waiting for logs…</p>
      ) : (
        logs.map((entry) => (
          <div key={entry.seq} className="text-secondary whitespace-pre-wrap leading-relaxed">
            <span className="text-muted">[{entry.seq}]</span> {entry.message}
          </div>
        ))
      )}
    </div>
  );
}
