import { useState } from "react";
import { NavLink } from "react-router-dom";
import { useSession } from "../state";
import { applyTheme, getTheme, THEMES } from "../theme";

function Pill({ tone = "neutral", children, title }) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      {children}
    </span>
  );
}

function ThemePicker() {
  const [theme, setTheme] = useState(() => getTheme());

  return (
    <label className="theme-picker" title="Colour theme">
      <span aria-hidden="true">◐</span>
      <select
        value={theme}
        aria-label="Colour theme"
        onChange={(event) => setTheme(applyTheme(event.target.value))}
      >
        {THEMES.map((item) => (
          <option key={item.id} value={item.id}>
            {item.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Header() {
  const { summary, health } = useSession();
  const model = health?.llm;
  const connected = summary?.source?.label;

  return (
    <header className="app-header">
      <div className="header-left">
        {connected ? (
          <Pill tone="source" title={summary.source.kind}>
            {summary.source.kind === "huggingface" ? "🤗" : summary.source.kind === "kaggle" ? " Kaggle" : "📄"}{" "}
            {connected}
          </Pill>
        ) : (
          <Pill tone="muted">No dataset connected</Pill>
        )}
      </div>

      <div className="header-center">
        <h1>Text Dataset Cleaner</h1>
        <p className="tagline">Diff-first cleaning for messy Tamil &amp; English datasets</p>
      </div>

      <div className="header-right">
        <nav className="stepper" aria-label="Workflow">
          <NavLink to="/file-select" className="step">
            <span className="step-num">1</span> File
          </NavLink>
          <NavLink to="/clean-data" className={({ isActive }) => `step ${isActive ? "active" : ""}`}>
            <span className="step-num">2</span> Clean
          </NavLink>
          <NavLink to="/upload" className={({ isActive }) => `step ${isActive ? "active" : ""}`}>
            <span className="step-num">3</span> Export
          </NavLink>
        </nav>
        <ThemePicker />
        <Pill
          tone={model?.available ? "ok" : "warn"}
          title={model?.available ? `Serving ${model.model}` : "Rule engine only"}
        >
          {model?.available ? `Gemma · ${model.provider}` : "Rules only"}
        </Pill>
      </div>
    </header>
  );
}
