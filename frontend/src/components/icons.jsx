// A hand-written, "network diagram" icon set — nodes-and-edges motifs for
// graph-related concepts, plain geometric outlines otherwise. Distinct
// stroke weight and geometry from both sibling projects' icon sets.
const base = {
  width: 18, height: 18, viewBox: "0 0 24 24", fill: "none",
  stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round", strokeLinejoin: "round",
};

// Brand mark: a hexagonal node with a filled center.
export const HexNodeIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3l7.5 4.3v9.4L12 21l-7.5-4.3V7.3z" />
    <circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none" />
  </svg>
);

// Datasets: a small point-cloud / data-grid, not a cylinder database.
export const DatabaseIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="5.5" cy="5.5" r="1.8" />
    <circle cx="12" cy="5.5" r="1.8" />
    <circle cx="18.5" cy="5.5" r="1.8" />
    <circle cx="5.5" cy="12" r="1.8" />
    <circle cx="12" cy="12" r="1.8" fill="currentColor" stroke="none" />
    <circle cx="18.5" cy="12" r="1.8" />
    <circle cx="5.5" cy="18.5" r="1.8" />
    <circle cx="12" cy="18.5" r="1.8" />
    <circle cx="18.5" cy="18.5" r="1.8" />
  </svg>
);

export const PlayCircleIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="M10 8.5l6 3.5-6 3.5v-7z" fill="currentColor" stroke="none" />
  </svg>
);

// Results / network comparison: three nodes, fully connected.
export const ActivityIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M6 18L12 6l6 12z" />
    <circle cx="12" cy="6" r="2" />
    <circle cx="6" cy="18" r="2" />
    <circle cx="18" cy="18" r="2" />
  </svg>
);

export const UploadIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 15V4" />
    <path d="M7 9l5-5 5 5" />
    <path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
  </svg>
);

export const ChevronDownIcon = (props) => (
  <svg {...base} {...props}><path d="M6 9l6 6 6-6" /></svg>
);

export const SunIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" />
  </svg>
);

export const MoonIcon = (props) => (
  <svg {...base} {...props}><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" /></svg>
);

export const SystemIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="3" y="4" width="18" height="12" rx="1.5" /><path d="M8 20h8M12 16v4" />
  </svg>
);

export const ArrowRightIcon = (props) => (
  <svg {...base} {...props}><path d="M5 12h14M13 6l6 6-6 6" /></svg>
);

export const CheckCircleIcon = (props) => (
  <svg {...base} {...props}><circle cx="12" cy="12" r="9" /><path d="M8 12.5l2.5 2.5L16 9.5" /></svg>
);

export const AlertCircleIcon = (props) => (
  <svg {...base} {...props}><circle cx="12" cy="12" r="9" /><path d="M12 8v5M12 16h.01" /></svg>
);

// Sensitivity sweep: literal sliders (the control this icon actually
// labels is two real sliders — community count, edge threshold).
export const SlidersIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
    <circle cx="16" cy="7" r="2" />
    <circle cx="8" cy="17" r="2" />
  </svg>
);

export const SearchUserIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="10" cy="9.3" r="3.2" />
    <path d="M4.5 19c.8-3 3-4.5 5.5-4.5s4.7 1.5 5.5 4.5" />
    <circle cx="18" cy="17" r="3" />
    <path d="M22 21l-1.6-1.6" />
  </svg>
);

// Ablation: stacked layers, one highlighted — matches "which layer of the
// pipeline matters" framing.
export const LayersIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3l8 4.5-8 4.5-8-4.5z" />
    <path d="M4 12l8 4.5 8-4.5" />
    <path d="M4 16.5l8 4.5 8-4.5" />
  </svg>
);

// Grid: sensitivity-sweep page glyph.
export const GridIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="3" y="3" width="6" height="6" rx="1" />
    <rect x="15" y="3" width="6" height="6" rx="1" />
    <rect x="3" y="15" width="6" height="6" rx="1" />
    <rect x="15" y="15" width="6" height="6" rx="1" />
  </svg>
);

// Statistical validation: a simple normal-distribution curve.
export const DistributionIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M3 19h18" />
    <path d="M3 19c3-0.2 4.5-13 9-13s6 12.8 9 13" />
  </svg>
);

export const SparkleIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3l1.8 4.9L19 9.5l-4.9 1.8L12 16l-1.8-4.7L5 9.5l5.2-1.6L12 3z" />
  </svg>
);

export const GithubIcon = (props) => (
  <svg width={18} height={18} viewBox="0 0 24 24" fill="currentColor" {...props}>
    <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.89 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.63-1.33-2.22-.25-4.56-1.11-4.56-4.95 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02a9.6 9.6 0 0 1 5 0c1.91-1.29 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.59 1.03 2.68 0 3.85-2.34 4.7-4.57 4.94.36.31.68.92.68 1.85v2.75c0 .26.18.58.69.48A10 10 0 0 0 12 2z" />
  </svg>
);

export const PulseDotIcon = (props) => (
  <svg width={8} height={8} viewBox="0 0 8 8" fill="currentColor" {...props}><circle cx="4" cy="4" r="4" /></svg>
);
