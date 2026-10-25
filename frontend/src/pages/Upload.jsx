import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import { useSession, useToast } from "../state";

const FORMATS = [
  { id: "csv", label: "CSV (.csv)" },
  { id: "tsv", label: "TSV (.tsv)" },
  { id: "json", label: "JSON (.json)" },
  { id: "jsonl", label: "JSON Lines (.jsonl)" },
  { id: "parquet", label: "Parquet (.parquet)" },
  { id: "avro", label: "Avro (.avro)" },
  { id: "xlsx", label: "Excel (.xlsx)" },
  { id: "txt", label: "Plain text (.txt) — one row per line" },
  { id: "zip", label: "Zip bundle (.zip) — cleaned.txt + manifest" },
];

function slugify(text) {
  return String(text ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

export default function Upload() {
  const { summary, refresh, clear, health } = useSession();
  const notify = useToast();
  const navigate = useNavigate();
  const [format, setFormat] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [publishing, setPublishing] = useState("");
  const [published, setPublished] = useState(null);
  const [hfRepo, setHfRepo] = useState("");
  const [kaggleRef, setKaggleRef] = useState("");
  const initializedFor = useRef("");

  const activeFormat = format ?? summary?.format ?? "csv";
  const baseName = useMemo(() => {
    const file = summary?.source?.file ?? "dataset";
    return file.replace(/\.[^.]+$/, "") || "dataset";
  }, [summary?.source?.file]);

  const kind = summary?.source?.kind ?? "";
  const sourceRepo = summary?.source?.repo ?? "";
  const isText = (summary?.format || "").toLowerCase() === "txt";
  const kaggleUser = health?.sources?.kaggle?.username ?? "";
  const kaggleConnected = Boolean(health?.sources?.kaggle?.connected);
  const hfToken = Boolean(health?.sources?.huggingface?.token);

  useEffect(() => {
    if (!summary?.id) return;
    // Re-initialise when the session changes *or* when Kaggle finishes loading:
    // the first render often has no username yet, and an empty default target
    // would leave the publish button disabled for no reason.
    const key = `${summary.id}:${kaggleUser}`;
    if (initializedFor.current === key) return;
    initializedFor.current = key;
    // Kaggle dataset slugs are capped at 50 characters, and the companion adds
    // "-cleaned" (8), so keep the base at 42 or the default target is invalid.
    const slug = (slugify(baseName) || "dataset").slice(0, 42).replace(/-+$/, "") || "dataset";
    setHfRepo(summary.source?.repo || "");
    setKaggleRef(kaggleUser ? `${kaggleUser}/${slug}-cleaned` : "");
    setFormat(null);
    setPublished(null);
  }, [summary?.id, summary?.source?.repo, baseName, kaggleUser]);

  if (!summary) {
    return (
      <main className="page">
        <div className="card empty">
          <h2>Nothing to export yet</h2>
          <p className="muted">Load a dataset and save at least one row before exporting.</p>
          <Link className="btn btn-primary" to="/file-select">
            Open the file picker
          </Link>
        </div>
      </main>
    );
  }

  const download = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.download(summary.id, activeFormat, `${baseName}.cleaned`);
      notify(`Downloaded ${baseName}.cleaned.${activeFormat}`, "ok");
      await refresh();
    } catch (err) {
      setError(err);
      notify(err.message, "error", err.hint);
    } finally {
      setBusy(false);
    }
  };

  const publish = async (target) => {
    setPublishing(target);
    setError(null);
    setPublished(null);
    try {
      const ref = target === "huggingface" ? hfRepo.trim() : kaggleRef.trim();
      const result = await api.publish(summary.id, { target, ref: ref || null, format: activeFormat });
      setPublished(result);
      notify(`Published ${result.path} to ${target === "huggingface" ? "Hugging Face" : "Kaggle"}.`, "ok");
    } catch (err) {
      setError(err);
      notify(err.message, "error", err.hint);
    } finally {
      setPublishing("");
    }
  };

  const canPublish = !isText && ["csv", "tsv", "json", "jsonl", "parquet", "avro", "xlsx", "txt", "zip"].includes(activeFormat);

  return (
    <main className="page">
      <div className="page-intro">
        <h2>Export the cleaned dataset</h2>
        <p className="muted">
          Your saved rows are written back into a copy of the original dataset, keeping every other column and the
          original row order.
        </p>
      </div>

      {error ? (
        <div className="banner banner-error" role="alert">
          <div>
            <strong>{error.message}</strong>
            {error.hint ? <p>{error.hint}</p> : null}
          </div>
          <button className="banner-close" onClick={() => setError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ) : null}

      <div className="grid-2">
        <section className="card">
          <h3>Session</h3>
          <dl className="stats">
            <div>
              <dt>Source</dt>
              <dd>{summary.source?.label ?? "—"}</dd>
            </div>
            <div>
              <dt>File</dt>
              <dd>{summary.source?.file ?? "—"}</dd>
            </div>
            <div>
              <dt>Rows</dt>
              <dd>{summary.rows.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Text column</dt>
              <dd>{summary.text_column ?? "—"}</dd>
            </div>
            <div>
              <dt>Rows saved</dt>
              <dd>
                <strong>{summary.saved}</strong>
              </dd>
            </div>
            <div>
              <dt>Rows changed</dt>
              <dd>{summary.changed}</dd>
            </div>
          </dl>
          <div className="row-actions">
            <Link className="btn btn-ghost" to="/clean-data">
              ← Keep cleaning
            </Link>
            <button
              className="btn btn-ghost"
              onClick={() => {
                clear();
                navigate("/file-select");
              }}
            >
              Finish &amp; disconnect
            </button>
          </div>
        </section>

        <section className="card">
          <h3>Download</h3>
          <label className="field">
            <span>Output format</span>
            <select value={activeFormat} onChange={(e) => setFormat(e.target.value)}>
              {FORMATS.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>

          <p className="muted small">
            Defaults to the format you imported ({summary.format?.toUpperCase()}). The download contains{" "}
            {summary.changed} edited row{summary.changed === 1 ? "" : "s"} out of {summary.rows.toLocaleString()}.
          </p>

          {activeFormat === "zip" ? (
            <p className="muted small">
              The zip holds <code>{baseName}.cleaned.txt</code> (one row per line), a{" "}
              <code>manifest.json</code> with row counts and the source, and — for tabular sources — a copy of the
              cleaned table in its original format.
            </p>
          ) : null}

          <button className="btn btn-primary btn-lg" onClick={download} disabled={busy}>
            {busy ? <span className="spinner" /> : "⬇"} Download {baseName}.cleaned.{activeFormat}
          </button>
        </section>
      </div>

      <section className="card" style={{ marginTop: 18 }}>
        <div className="card-head">
          <div>
            <h3>Publish the cleaned data</h3>
            <p className="muted small" style={{ margin: 0 }}>
              Writes the same file you would download, without leaving this machine&apos;s credentials behind.
            </p>
          </div>
          {canPublish ? <span className="pill pill-muted">{summary.changed} saved rows</span> : null}
        </div>

        {isText ? (
          <div className="manual-note">
            <strong>Text-file datasets are uploaded by hand.</strong>
            <p>
              A <code>.txt</code> source is a list of lines, not a repository file, so there is nothing to write back
              to. Download the zip export, then upload <code>{baseName}.cleaned.txt</code> and{" "}
              <code>manifest.json</code> wherever the dataset lives.
            </p>
            <pre className="code">
{`# Hugging Face (web: Files → Add file, or CLI)
huggingface-cli upload <owner>/<dataset> ./${baseName}.cleaned.zip cleaned_datasets/${baseName}.cleaned.zip --repo-type dataset`}
            </pre>
          </div>
        ) : (
          <div className="publish-grid">
            {kind === "huggingface" && sourceRepo ? (
              <div className="publish-block">
                <h4>🤗 Hugging Face</h4>
                <p className="muted small" style={{ margin: 0 }}>
                  Adds <code>cleaned_datasets/{baseName}.cleaned.{activeFormat}</code> to the same repository. Your
                  original files are untouched.
                </p>
                <label className="field">
                  <span>Repository</span>
                  <input
                    value={hfRepo}
                    onChange={(e) => setHfRepo(e.target.value)}
                    placeholder="owner/dataset-name"
                  />
                </label>
                {!hfToken ? (
                  <p className="muted small">
                    No <code>HF_TOKEN</code> is set on the server. Add a token with write access and restart it; the
                    token is never sent to this page.
                  </p>
                ) : null}
                <div className="publish-actions">
                  <button
                    className="btn btn-primary"
                    onClick={() => publish("huggingface")}
                    disabled={!canPublish || !hfRepo.trim() || publishing === "huggingface"}
                  >
                    {publishing === "huggingface" ? <span className="spinner" /> : "⬆"} Publish to Hugging Face
                  </button>
                  <span className="muted small">folder: cleaned_datasets/</span>
                </div>
                {published?.target === "huggingface" ? (
                  <div className="publish-result">
                    Uploaded <code>{published.path}</code> to{" "}
                    <a href={published.url} target="_blank" rel="noreferrer">
                      {published.ref}
                    </a>
                    .
                  </div>
                ) : null}
              </div>
            ) : null}

            {kind === "kaggle" && sourceRepo ? (
              <div className="publish-block">
                <h4>Kaggle</h4>
                <p className="muted small" style={{ margin: 0 }}>
                  Kaggle versions replace a dataset&apos;s whole file list, so cleaned rows go to a companion dataset
                  instead. Your source dataset is never modified.
                </p>
                <label className="field">
                  <span>Target dataset</span>
                  <input
                    value={kaggleRef}
                    onChange={(e) => setKaggleRef(e.target.value)}
                    placeholder={kaggleConnected ? "owner/my-dataset-cleaned" : "connect Kaggle first"}
                    disabled={!kaggleConnected}
                  />
                </label>
                {!kaggleConnected ? (
                  <p className="muted small">
                    Connect Kaggle on the <Link to="/file-select">File</Link> tab to publish.
                  </p>
                ) : null}
                <div className="publish-actions">
                  <button
                    className="btn btn-primary"
                    onClick={() => publish("kaggle")}
                    disabled={!canPublish || !kaggleRef.trim() || !kaggleConnected || publishing === "kaggle"}
                  >
                    {publishing === "kaggle" ? <span className="spinner" /> : "⬆"} Publish to Kaggle
                  </button>
                  <span className="muted small">file: {baseName}.cleaned.{activeFormat}</span>
                </div>
                {published?.target === "kaggle" ? (
                  <div className="publish-result">
                    Uploaded <code>{published.path}</code> to{" "}
                    <a href={published.url} target="_blank" rel="noreferrer">
                      {published.ref}
                    </a>
                    .
                  </div>
                ) : null}
              </div>
            ) : null}

            {kind !== "huggingface" && kind !== "kaggle" ? (
              <div className="manual-note">
                <strong>This session came from a local upload or a sample.</strong>
                <p>
                  There is no source repository to write back to. Download the cleaned file and upload it to Hugging
                  Face or Kaggle yourself — the export is byte-for-byte the same file the publish buttons would send.
                </p>
              </div>
            ) : null}

            {kind === "huggingface" && !sourceRepo ? (
              <div className="manual-note">
                <strong>This Hugging Face session has no repository attached.</strong>
                <p>Reload it from the file picker so the publish target can be filled in.</p>
              </div>
            ) : null}
          </div>
        )}
      </section>
    </main>
  );
}
