/** Colour themes. The palette lives in styles.css; this module only records the
 *  user's choice and puts it on <html data-theme="...">. */

export const THEMES = [
  { id: "light", label: "Light", hint: "Paper white" },
  { id: "tokyo-night", label: "Tokyo Night", hint: "Deep indigo" },
  { id: "synthwave", label: "Synthwave", hint: "Neon dusk" },
  { id: "catppuccin", label: "Catppuccin", hint: "Mocha" },
];

const STORAGE_KEY = "tdc.theme";
const DEFAULT_DARK = "tokyo-night";

export function isTheme(id) {
  return THEMES.some((theme) => theme.id === id);
}

export function getTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && isTheme(stored)) return stored;
  } catch {
    /* private mode: fall through to the system preference */
  }
  const prefersDark =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? DEFAULT_DARK : "light";
}

export function applyTheme(id) {
  const theme = isTheme(id) ? id : "light";
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme === "light" ? "light" : "dark";
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* ignore storage failures */
  }
  return theme;
}

export function initTheme() {
  // `?theme=tokyo-night` makes a themed view shareable (and lets screenshots
  // pick a palette without touching localStorage first).
  let requested = null;
  try {
    requested = new URLSearchParams(window.location.search).get("theme");
  } catch {
    requested = null;
  }
  return applyTheme(isTheme(requested) ? requested : getTheme());
}
