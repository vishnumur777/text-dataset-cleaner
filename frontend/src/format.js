export function bytes(n, empty = "—") {
  if (!n) return empty;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function filenameOf(path) {
  const text = String(path ?? "");
  const index = text.lastIndexOf("/");
  return index === -1 ? text : text.slice(index + 1);
}
