import { useEffect, useState } from "react";
import { MoonIcon, SunIcon, SystemIcon } from "./icons";

const OPTIONS = [
  { key: "light", icon: SunIcon, title: "Light" },
  { key: "system", icon: SystemIcon, title: "Match system" },
  { key: "dark", icon: MoonIcon, title: "Dark" },
];

export default function ThemeToggle() {
  const [theme, setTheme] = useState(() => localStorage.getItem("cgnf-theme") || "system");

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("cgnf-theme", theme);
    } catch {
      /* ignore storage errors */
    }
  }, [theme]);

  return (
    <div className="inline-flex items-center gap-0.5 p-0.5 rounded" style={{ background: "var(--surface-2)" }}>
      {OPTIONS.map(({ key, icon: Icon, title }) => (
        <button
          key={key} type="button" onClick={() => setTheme(key)} title={title} aria-label={title}
          className="p-1.5 rounded"
          style={theme === key ? { background: "var(--surface-1)", color: "var(--accent)", boxShadow: "var(--shadow-sm)" } : { color: "var(--text-muted)" }}
        >
          <Icon width={14} height={14} />
        </button>
      ))}
    </div>
  );
}
