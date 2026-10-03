export function relativeTime(epochSeconds) {
  if (!epochSeconds) return "";
  const seconds = Math.round((Date.now() - epochSeconds * 1000) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function compactNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function pct(value, decimals = 1) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function num(value, decimals = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(decimals);
}
