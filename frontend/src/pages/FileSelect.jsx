import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api";
import { LoadProgress } from "../components/LoadProgress";
import { bytes, filenameOf, sleep } from "../format";
import { useSession, useToast } from "../state";

function Banner({ error, onClose }) {
  if (!error) return null;
  return (
    <div className="banner banner-error" role="alert">
      <div>
        <strong>{error.message}</strong>
        {error.hint ? <p>{error.hint}</p> : null}
      </div>
      {onClose ? (
        <button className="banner-close" onClick={onClose} aria-label="Dismiss">
          ×
        </button>
      ) : null}
    </div>
  );
}

const TABS = [
  { id: "upload", label: "Upload a file" },
  { id: "huggingface", label: "Hugging Face" },
  { id: "kaggle", label: "Kaggle" },
  { id: "sample", label: "Try a sample" },
];

function ColumnPicker({ loaded, onRestart }) {
  const navigate = useNavigate();
  const { adopt, refresh } = useSession();
  const notify = useToast();
  const [column, setColumn] = useState(loaded.suggested || loaded.session.columns[0]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const session = loaded.session;

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const summary = await api.setColumn(session.id, column);
      adopt({ session: summary });
      await refresh();
      navigate("/clean-data");
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card column-card">
      <div className="card-head">
        <div>
          <h2>Choose the text column</h2>
          <p className="muted">
            {session.rows.toLocaleString()} rows · {session.columns.length} columns · {session.format?.toUpperCase()} ·{" "}
            {session.source?.label}
          </p>
        </div>
        <button className="btn btn-ghost" onClick={onRestart}>
          Load something else
        </button>
      </div>

      {session.truncated ? (
        <div className="banner banner-warn">
          <div>
            <strong>Large file: only the first {session.rows.toLocaleString()} rows were loaded.</strong>
            <p>Cleaning and export apply to the loaded rows. Export keeps the original column order.</p>
          </div>
        </div>
      ) : null}

      <Banner error={error} onClose={() => setError(null)} />

      <label className="field">
        <span>Column to clean</span>
        <select value={column} onChange={(e) => setColumn(e.target.value)}>
          {session.columns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>

      <div className="preview-table">
        <table>
          <thead>
            <tr>
              {session.columns.slice(0, 4).map((c) => (
                <th key={c} className={c === column ? "picked" : ""}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loaded.preview.map((row, i) => (
              <tr key={i}>
                {session.columns.slice(0, 4).map((c) => (
                  <td key={c} className={c === column ? "picked" : ""}>
                    {String(row[c] ?? "").slice(0, 120)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <button className="btn btn-primary btn-lg" onClick={start} disabled={busy}>
        {busy ? <span className="spinner" /> : null} Start cleaning
      </button>
    </div>
  );
}

export default function FileSelect() {
  const [tab, setTab] = useState("upload");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);
  const [loadJob, setLoadJob] = useState(null);
  const [progress, setProgress] = useState(0);
  const [loaded, setLoaded] = useState(null);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [repo, setRepo] = useState(null);
  const [files, setFiles] = useState([]);

  const [kaggleQuery, setKaggleQuery] = useState("");
  const [kaggleResults, setKaggleResults] = useState([]);
  const [kaggleRepo, setKaggleRepo] = useState(null);
  const [kaggleFiles, setKaggleFiles] = useState([]);
  const [creds, setCreds] = useState({ username: "", key: "" });
  const [connecting, setConnecting] = useState(false);
  const [samples, setSamples] = useState([]);

  const { health, loadHealth, adopt } = useSession();
  const notify = useToast();
  const inputRef = useRef(null);
  const jobToken = useRef(0);
  const kaggleConnected = health?.sources?.kaggle?.connected;

  useEffect(() => {
    if (tab === "sample" && !samples.length) {
      api.samples().then((d) => setSamples(d.samples || [])).catch(() => setSamples([]));
    }
  }, [tab, samples.length]);

  const finish = async (payload, previewRows) => {
    adopt(payload);
    setLoaded({
      session: payload.session,
      suggested: payload.suggested_column,
      format: payload.format,
      preview: previewRows,
    });
    await loadHealth();
  };

  /** Start a server-side download job and follow it until it finishes.
   *
   * The server returns immediately with a job id, so the browser stays
   * responsive and can paint real byte progress instead of waiting on one long
   * request. A token guards against a stale loop overwriting a newer job. */
  const runLoadJob = useCallback(async (kind, body, label) => {
    const token = jobToken.current + 1;
    jobToken.current = token;
    setLoadJob({
      status: "running",
      phase: "starting",
      message: label,
      done: 0,
      total: 0,
      percent: null,
    });
    try {
      const started = kind === "huggingface" ? await api.hfOpenJob(body.repo, body.path) : await api.kaggleOpenJob(body.ref, body.path);
      let current = started.job;
      if (!current) throw new ApiError({ message: "The server did not start the download." });
      setLoadJob(current);
      while (current.status === "running") {
        await sleep(current.total > 0 ? 320 : 520);
        if (jobToken.current !== token) return null;
        current = await api.job(current.id);
        if (jobToken.current !== token) return null;
        setLoadJob(current);
      }
      if (current.status !== "done") {
        const failure = current.error || {};
        throw new ApiError({
          code: failure.code || "load_failed",
          message: failure.message || "The dataset could not be loaded.",
          hint: failure.hint || "",
          status: 400,
        });
      }
      // Hold the finished bar for a beat: a cached file completes in well under
      // one poll, and clearing immediately makes the progress look like it
      // never ran.
      setLoadJob(current);
      await sleep(450);
      if (jobToken.current !== token) return null;
      return current.result;
    } finally {
      if (jobToken.current === token) setLoadJob(null);
    }
  }, []);

  const handleFile = async (file) => {
    if (!file) return;
    setError(null);
    setProgress(0);
    setBusy("Reading file…");
    try {
      const payload = await api.upload(file, (percent) => {
        setProgress(percent);
        if (percent >= 100) setBusy("Parsing the table…");
      });
      const preview = await api.row(payload.session.id, 0).catch(() => null);
      await finish(payload, preview ? [{ [preview.column]: preview.original }] : []);
      notify("File loaded. Pick the text column next.", "ok");
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
      setProgress(0);
    }
  };

  const searchHf = async (e) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setBusy("Searching Hugging Face…");
    setError(null);
    setResults([]);
    setRepo(null);
    try {
      const data = await api.hfSearch(query.trim());
      setResults(data.results || []);
      if (!data.results?.length) notify("No datasets matched that search.", "info");
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  };

  const openRepo = async (item) => {
    setBusy("Listing files…");
    setError(null);
    setRepo(item);
    setFiles([]);
    try {
      const data = await api.hfFiles(item.id);
      setFiles(data.files || []);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  };

  const openHfFile = async (file) => {
    const name = filenameOf(file.path);
    setBusy(`Downloading ${name}…`);
    setError(null);
    try {
      const payload = await runLoadJob("huggingface", { repo: repo.id, path: file.path }, `Downloading ${name}…`);
      if (!payload) return;
      const preview = await api.row(payload.session.id, 0).catch(() => null);
      await finish(payload, preview ? [{ [preview.column]: preview.original }] : []);
      notify("Dataset loaded from Hugging Face.", "ok");
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  };

  const searchKaggle = async (e) => {
    e?.preventDefault();
    if (!kaggleQuery.trim()) return;
    setBusy("Searching Kaggle…");
    setError(null);
    setKaggleResults([]);
    setKaggleRepo(null);
    try {
      const data = await api.kaggleSearch(kaggleQuery.trim());
      setKaggleResults(data.results || []);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  };

  const connectKaggle = async (e) => {
    e.preventDefault();
    setConnecting(true);
    setError(null);
    try {
      await api.kaggleConnect(creds.username, creds.key);
      notify("Kaggle connected. Your key is stored on this machine only.", "ok");
      setCreds({ username: "", key: "" });
      await loadHealth();
    } catch (err) {
      setError(err);
    } finally {
      setConnecting(false);
    }
  };

  const openKaggleRepo = async (item) => {
    setBusy("Listing files…");
    setError(null);
    setKaggleRepo(item);
    setKaggleFiles([]);
    try {
      const data = await api.kaggleFiles(item.id);
      setKaggleFiles(data.files || []);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  };

  const openKaggleFile = async (file) => {
    const name = filenameOf(file.path);
    setBusy(`Downloading ${name}…`);
    setError(null);
    try {
      const payload = await runLoadJob("kaggle", { ref: kaggleRepo.id, path: file.path }, `Downloading ${name}…`);
      if (!payload) return;
      const preview = await api.row(payload.session.id, 0).catch(() => null);
      await finish(payload, preview ? [{ [preview.column]: preview.original }] : []);
      notify("Dataset loaded from Kaggle.", "ok");
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  };

  const openSample = async (name) => {
    setBusy(`Opening ${name}…`);
    setError(null);
    try {
      const payload = await api.openSample(name);
      const preview = await api.row(payload.session.id, 0).catch(() => null);
      await finish(payload, preview ? [{ [preview.column]: preview.original }] : []);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  };

  if (loaded) {
    return (
      <main className="page">
        <ColumnPicker loaded={loaded} onRestart={() => setLoaded(null)} />
      </main>
    );
  }

  return (
    <main className="page">
      <div className="page-intro">
        <h2>Bring in a dataset</h2>
        <p className="muted">
          Upload a file, pull one from Hugging Face or Kaggle, or start from a sample. Supported import formats: CSV,
          TSV, JSON, JSONL, Parquet, Avro, Excel, and TXT (one sample per line).
        </p>
      </div>

      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={`tab ${tab === t.id ? "active" : ""}`}
            onClick={() => {
              setTab(t.id);
              setError(null);
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <Banner error={error} onClose={() => setError(null)} />
      {loadJob ? <LoadProgress job={loadJob} fallback={busy || ""} /> : null}
      {!loadJob && busy ? (
        <div className="inline-busy">
          <span className="spinner" /> {busy}
          {progress ? <span className="muted"> {progress}%</span> : null}
        </div>
      ) : null}

      {tab === "upload" ? (
        <section className="card">
          <div
            className="dropzone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFile(e.dataTransfer.files?.[0]);
            }}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
            }}
          >
            <div className="dropzone-icon">⬆</div>
            <h3>Drop a dataset file here</h3>
            <p className="muted">or click to browse — max size is limited by your disk, not by the browser</p>
            <input
              ref={inputRef}
              type="file"
              hidden
              accept=".csv,.tsv,.json,.jsonl,.ndjson,.parquet,.avro,.xlsx,.xls,.txt"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>
          {progress > 0 ? (
            <div className="progress">
              <div className="bar" style={{ width: `${progress}%` }} />
            </div>
          ) : null}
          <p className="fineprint">
            Files are stored in a temporary session on this machine. Nothing is uploaded to a third party.
          </p>
        </section>
      ) : null}

      {tab === "huggingface" ? (
        <section className="card">
          <form className="search-row" onSubmit={searchHf}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search datasets, e.g. tamil corpus"
              aria-label="Search Hugging Face datasets"
            />
            <button className="btn btn-primary" type="submit" disabled={!!busy}>
              Search
            </button>
          </form>

          {results.length ? (
            <ul className="result-list">
              {results.map((item) => (
                <li key={item.id} className={repo?.id === item.id ? "selected" : ""}>
                  <button className="result" onClick={() => openRepo(item)}>
                    <span className="result-title">{item.id}</span>
                    <span className="result-meta">
                      ⬇ {item.downloads?.toLocaleString?.() ?? item.downloads} · ♥ {item.likes}
                      {item.gated ? " · gated" : ""}
                      {item.private ? " · private" : ""}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}

          {repo ? (
            <div className="file-panel">
              <h3>{repo.id}</h3>
              <p className="muted">Pick a tabular file to download and clean.</p>
              <ul className="file-list">
                {files.map((f) => (
                  <li key={f.path}>
                    <button
                      className={`file ${f.supported ? "" : "unsupported"}`}
                      onClick={() => f.supported && openHfFile(f)}
                      disabled={!f.supported}
                      title={f.supported ? "Open this file" : "Not a supported tabular format"}
                    >
                      <span className="file-name">{f.path}</span>
                      <span className="file-meta">
                        {f.ext} · {bytes(f.size)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
              {!files.some((f) => f.supported) ? (
                <div className="banner banner-warn">
                  <div>
                    <strong>No directly readable table in this repository.</strong>
                    <p>Look for a .parquet, .csv, .txt or .json file, or download it manually and use the Upload tab.</p>
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}

          <p className="fineprint">
            Public datasets need no login. Add <code>HF_TOKEN</code> on the server to read gated or private datasets.
          </p>
        </section>
      ) : null}

      {tab === "kaggle" ? (
        <section className="card">
          {!kaggleConnected ? (
            <>
              <h3>Connect Kaggle</h3>
              <p className="muted">
                Kaggle needs an API key. Create one at <code>kaggle.com/settings → API → Create New Token</code>. It is
                saved to <code>~/.config/text-dataset-cleaner/kaggle.json</code> with owner-only permissions and never
                leaves this machine.
              </p>
              <form className="stack" onSubmit={connectKaggle}>
                <label className="field">
                  <span>Kaggle username</span>
                  <input
                    value={creds.username}
                    onChange={(e) => setCreds({ ...creds, username: e.target.value })}
                    autoComplete="username"
                    placeholder="your-kaggle-username"
                  />
                </label>
                <label className="field">
                  <span>API key</span>
                  <input
                    type="password"
                    value={creds.key}
                    onChange={(e) => setCreds({ ...creds, key: e.target.value })}
                    autoComplete="off"
                    placeholder="paste the key from kaggle.json"
                  />
                </label>
                <button className="btn btn-primary" type="submit" disabled={connecting}>
                  {connecting ? <span className="spinner" /> : null} Connect Kaggle
                </button>
              </form>
            </>
          ) : (
            <>
              <div className="connected-row">
                <span className="pill pill-ok">Connected as {health.sources.kaggle.username}</span>
              </div>
              <form className="search-row" onSubmit={searchKaggle}>
                <input
                  value={kaggleQuery}
                  onChange={(e) => setKaggleQuery(e.target.value)}
                  placeholder="Search Kaggle datasets, e.g. tamil news"
                  aria-label="Search Kaggle datasets"
                />
                <button className="btn btn-primary" type="submit" disabled={!!busy}>
                  Search
                </button>
              </form>

              {kaggleResults.length ? (
                <ul className="result-list">
                  {kaggleResults.map((item) => (
                    <li key={item.id} className={kaggleRepo?.id === item.id ? "selected" : ""}>
                      <button className="result" onClick={() => openKaggleRepo(item)}>
                        <span className="result-title">{item.title || item.id}</span>
                        <span className="result-meta">
                          {item.id} · ⬇ {item.downloads} · {bytes(item.size)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              ) : null}

              {kaggleRepo ? (
                <div className="file-panel">
                  <h3>{kaggleRepo.id}</h3>
                  <ul className="file-list">
                    {kaggleFiles.map((f) => (
                      <li key={f.path}>
                        <button
                          className={`file ${f.supported ? "" : "unsupported"}`}
                          onClick={() => f.supported && openKaggleFile(f)}
                          disabled={!f.supported}
                        >
                          <span className="file-name">{f.path}</span>
                          <span className="file-meta">
                            {f.ext} · {bytes(f.size)}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </>
          )}
        </section>
      ) : null}

      {tab === "sample" ? (
        <section className="card">
          <h3>Bundled samples</h3>
          <p className="muted">
            Synthetic Tamil and English rows with the exact corruption this tool fixes: stray spaces inside words,
            glued words, single-letter splits, NBSP and zero-width characters.
          </p>
          <ul className="file-list">
            {samples.map((s) => (
              <li key={s.path}>
                <button className="file" onClick={() => openSample(s.path)}>
                  <span className="file-name">{s.path}</span>
                  <span className="file-meta">{bytes(s.size)}</span>
                </button>
              </li>
            ))}
            {!samples.length ? <li className="muted">No samples found in ./samples</li> : null}
          </ul>
        </section>
      ) : null}
    </main>
  );
}
