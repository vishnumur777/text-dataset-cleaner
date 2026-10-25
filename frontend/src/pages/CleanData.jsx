import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import { EditableDiff, DiffPanel } from "../components/Diff";
import { PlaybackBar } from "../components/PlaybackBar";
import { countChanges, diffWords } from "../diff";
import { useSession, useToast } from "../state";

function humanEta(seconds) {
  if (seconds == null) return "estimating";
  if (seconds < 40) return "seconds";
  if (seconds < 90) return "about a minute";
  const minutes = Math.round(seconds / 60);
  return `about ${minutes} min`;
}

function estimatePass(rows) {
  // Measured on this machine: with three calls in flight the pass sustains one
  // model call per ~5.5 s, and roughly half of a messy column needs a call.
  return humanEta(Math.max(5, rows * 0.5 * 5.5));
}

export default function CleanData() {
  const { summary, adopt, refresh } = useSession();
  const navigate = useNavigate();
  const notify = useToast();
  const [bootstrapping, setBootstrapping] = useState(false);


  const total = summary?.rows ?? 0;
  const [index, setIndex] = useState(0);
  const [row, setRow] = useState(null);
  const [value, setValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [engine, setEngine] = useState("");
  const [saved, setSaved] = useState(summary?.saved ?? 0);
  const [error, setError] = useState(null);
  const [job, setJob] = useState(null);
  const playingRef = useRef(false);
  const autoRan = useRef(-1);
  const jobStatus = useRef("idle");
  const dirtyRef = useRef(false);
  const deepLink = useRef("");

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const demo = params.get("demo");
    const sessionParam = params.get("session");
    const target = sessionParam ? `session:${sessionParam}` : demo ? `demo:${demo}` : "";

    if (!target) {
      if (!summary?.id) navigate("/file-select", { replace: true });
      return;
    }
    // A deep link must win over a session restored from localStorage, otherwise
    // `?session=<id>` silently keeps showing the previous dataset. The ref also
    // stops a failed load from retrying in a loop.
    if (bootstrapping || deepLink.current === target) return;
    if (sessionParam && summary?.id === sessionParam) {
      deepLink.current = target;
      return;
    }

    deepLink.current = target;
    setBootstrapping(true);
    const request = sessionParam
      ? api.session(sessionParam).then((data) => {
          adopt(data);
          notify("Editing session reopened.", "ok");
        })
      : api.openSample(demo === "1" ? "messy_tamil.csv" : demo).then((payload) => {
          adopt(payload);
          notify("Sample dataset loaded — this is demo data.", "ok");
        });
    request.catch((err) => notify(err.message, "error", err.hint)).finally(() => setBootstrapping(false));
  }, [summary?.id, bootstrapping, adopt, navigate, notify]);

  const loadRow = useCallback(
    async (next) => {
      if (!summary?.id) return;
      setLoading(true);
      setError(null);
      try {
        const data = await api.row(summary.id, next);
        setRow(data);
        setValue(data.saved ?? data.suggested ?? data.original);
        setEngine(data.saved ? "saved" : data.engine || "rules");
      } catch (err) {
        setError(err);
        if (err.code === "session_missing") {
          notify("That session expired. Load the dataset again.", "error", err.hint);
        }
      } finally {
        setLoading(false);
      }
    },
    [summary?.id, notify]
  );

  useEffect(() => {
    if (summary?.id) loadRow(index);
  }, [summary?.id, index, loadRow]);

  // ------------------------------------------------------- one-pass pre-clean
  // Restore the job banner after a reload: the server keeps running the pass
  // even if this tab was closed.
  useEffect(() => {
    if (!summary?.id) return;
    let alive = true;
    api
      .precleanStatus(summary.id)
      .then((data) => {
        if (!alive) return;
        jobStatus.current = data.status;
        setJob(data.status === "idle" ? null : data);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [summary?.id]);

  useEffect(() => {
    if (job?.status !== "running" || !summary?.id) return;
    const timer = setInterval(() => {
      api
        .precleanStatus(summary.id)
        .then(setJob)
        .catch(() => {});
    }, 1200);
    return () => clearInterval(timer);
  }, [job?.status, summary?.id]);

  useEffect(() => {
    if (!job) return;
    const previous = jobStatus.current;
    if (job.status === previous) return;
    jobStatus.current = job.status;
    if (previous !== "running" || job.status === "running") return;

    if (job.status === "done") {
      notify(
        `Pre-cleaned all ${job.total} rows — ${job.skipped} were already clean, ${job.failed} kept the rule result.`,
        job.failed ? "warn" : "ok",
        job.warning
      );
      if (!dirtyRef.current) loadRow(index);
      refresh();
    } else if (job.status === "cancelled") {
      notify("Pre-clean stopped. Every finished row is kept.", "info");
      if (!dirtyRef.current) loadRow(index);
      refresh();
    } else {
      notify("Pre-cleaning could not finish.", "error", job.error || "Check the server log, then try again.");
    }
  }, [job, index, loadRow, refresh, notify]);

  const startPreclean = useCallback(async () => {
    if (!summary?.id) return;
    setError(null);
    try {
      const started = await api.preclean(summary.id, "llm", 3);
      jobStatus.current = started.status === "running" ? "running" : started.status;
      setJob(started);
      notify("Cleaning the whole column in one pass…", "info");
    } catch (err) {
      setError(err);
      notify(err.message, "error", err.hint);
    }
  }, [summary?.id, notify]);

  const cancelPreclean = useCallback(async () => {
    if (!summary?.id) return;
    try {
      setJob(await api.precleanCancel(summary.id));
    } catch (err) {
      notify(err.message, "error", err.hint);
    }
  }, [summary?.id, notify]);

  const runModel = useCallback(
    async ({ auto = false } = {}) => {
      if (!summary?.id || running) return;
      setRunning(true);
      setError(null);
      try {
        const result = await api.clean(summary.id, index, "llm");
        setValue(result.cleaned);
        setEngine(result.engine);
        if (result.warning) notify(result.warning, "warn");
        if (!auto) notify(`Cleaned with the ${result.engine === "llm" ? "local model" : "rule engine"}.`, "ok");
        if (auto && playingRef.current && index < total - 1) {
          setTimeout(() => setIndex((i) => Math.min(i + 1, total - 1)), 450);
        }
        return result;
      } catch (err) {
        setError(err);
        setPlaying(false);
        notify(err.message, "error", err.hint);
      } finally {
        setRunning(false);
      }
    },
    [summary?.id, index, total, running, notify]
  );

  // Auto-play: advance through rows. If the one-pass pre-clean already handled
  // a row, just show it and move on; otherwise run the model for this row only.
  useEffect(() => {
    if (!playing || loading || running || !row) return;
    if (autoRan.current === index) return;
    autoRan.current = index;

    if (row.precleaned) {
      if (index < total - 1) {
        const timer = setTimeout(() => {
          if (playingRef.current) setIndex((i) => Math.min(i + 1, total - 1));
        }, 900);
        return () => clearTimeout(timer);
      }
      setPlaying(false);
      return undefined;
    }

    runModel({ auto: true });
    return undefined;
  }, [playing, loading, running, row, index, total, runModel]);

  const save = useCallback(async () => {
    if (!summary?.id || !row) return;
    try {
      const result = await api.save(summary.id, index, value);
      setSaved(result.saved);
      await refresh();
      notify(`Saved row ${index + 1} into the cleaned dataset.`, "ok");
      if (index < total - 1) {
        setIndex(index + 1);
      } else {
        setPlaying(false);
      }
    } catch (err) {
      setError(err);
      notify(err.message, "error", err.hint);
    }
  }, [summary?.id, row, index, value, total, refresh, notify]);

  const go = useCallback(
    (delta) => {
      setPlaying(false);
      setIndex((i) => Math.max(0, Math.min(i + delta, total - 1)));
    },
    [total]
  );

  useEffect(() => {
    const onKey = (e) => {
      const target = e.target;
      const typing =
        target?.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName);
      if (typing && !(e.ctrlKey || e.metaKey)) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        save();
        return;
      }
      if (typing) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        go(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        go(-1);
      } else if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key.toLowerCase() === "r") {
        runModel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, runModel, save]);

  const ops = useMemo(() => (row ? diffWords(row.original, value) : []), [row, value]);
  const changes = useMemo(() => countChanges(ops), [ops]);
  const dirty = row ? value !== (row.saved ?? row.suggested) : false;
  dirtyRef.current = dirty;

  const jobPercent = useMemo(() => {
    if (!job) return 0;
    if (job.status !== "running") return 100;
    if (job.llm_total > 0) return Math.min(99, Math.round((job.llm_done / job.llm_total) * 100));
    const denominator = Math.max(job.total || 0, 1);
    return Math.min(99, Math.round(((job.scanned || 0) / denominator) * 100));
  }, [job]);

  if (!summary) {
    return (
      <main className="page">
        <div className="card empty">
          <h2>No dataset loaded</h2>
          <p className="muted">Start by uploading a file or picking a repository.</p>
          <Link className="btn btn-primary" to="/file-select">
            Open the file picker
          </Link>
        </div>
      </main>
    );
  }

  return (
    <main className="page page-clean">
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

      <div className="clean-toolbar">
        <div>
          <span className="muted">Column</span> <strong>{summary.text_column}</strong>
          <span className={`pill pill-engine ${engine === "llm" ? "pill-ok" : "pill-muted"}`}>
            {engine === "llm" ? "Gemma 3 1B" : engine === "saved" ? "Saved" : "Rule engine"}
          </span>
        </div>
        <div className="toolbar-right">
          <span className="muted">
            {changes.total} character{changes.total === 1 ? "" : "s"} changed
          </span>
          <Link to="/upload" className="btn btn-ghost">
            Export →
          </Link>
        </div>
      </div>

      <div className={`batch-bar ${job?.status === "running" ? "is-running" : ""}`}>
        <div className="batch-copy">
          <strong>
            {job?.status === "running"
              ? "Cleaning the whole column"
              : job?.status === "done"
                ? "Column pre-cleaned"
                : `Clean all ${total} rows in one pass`}
          </strong>
          <span className="muted small">
            {job?.status === "running"
              ? `${job.llm_done} of ${job.llm_total} model calls done · ${
                  job.eta_s == null ? "estimating time left" : `${humanEta(job.eta_s)} left`
                }`
              : job?.status === "done"
                ? `${job.skipped} rows were already clean · ${job.reused} duplicate cells reused · ${job.failed} kept the rule result`
                : `Gemma runs once over only the rows that need it (${estimatePass(total)} for these ${total}), then every row opens instantly.`}
          </span>
        </div>

        {job?.status === "running" ? (
          <div className="batch-right">
            <div className="batch-progress" aria-hidden="true">
              <div className="bar" style={{ width: `${jobPercent}%` }} />
            </div>
            <button className="btn btn-ghost btn-sm" onClick={cancelPreclean}>
              Stop
            </button>
          </div>
        ) : (
          <div className="batch-right">
            {job?.status === "done" ? <span className="pill pill-ok">{job.total} rows ready</span> : null}
            <button className="btn btn-primary btn-sm" onClick={startPreclean}>
              {job?.status === "done" ? "Run again" : `Clean all ${total} rows`}
            </button>
          </div>
        )}
      </div>

      <div className="panels">
        <section className="panel">
          <header className="panel-head">
            <h3>Original</h3>
            <span className="legend">
              <span className="dot dot-del" /> text replaced or removed
            </span>
          </header>
          <div className="panel-body">
            {loading && !row ? <div className="skeleton" /> : null}
            {row ? <DiffPanel original={row.original} edited={value} side="left" /> : null}
          </div>
        </section>

        <section className="panel panel-editable">
          <header className="panel-head">
            <h3>
              Cleaned <span className="muted small">editable</span>
            </h3>
            <div className="panel-actions">
              <button className="btn btn-ghost btn-sm" onClick={() => setValue(row?.suggested ?? "")} disabled={!row}>
                Reset
              </button>
              <button className="btn btn-primary btn-sm" onClick={save} disabled={!row || loading}>
                Save text
              </button>
            </div>
          </header>
          <div className="panel-body">
            {row ? (
              <EditableDiff
                original={row.original}
                value={value}
                onChange={setValue}
                placeholder="The cleaned text will appear here. You can edit it before saving."
              />
            ) : (
              <div className="skeleton" />
            )}
          </div>
          <footer className="panel-foot">
            <span className="legend">
              <span className="dot dot-ins" /> added by the cleaner
            </span>
            {dirty ? <span className="pill pill-warn">Unsaved changes</span> : <span className="pill pill-muted">In sync</span>}
          </footer>
        </section>
      </div>

      {row?.fixes?.length ? (
        <div className="fixes">
          <span className="muted">Fixes on this row</span>
          {row.fixes.map((fix, i) => (
            <span key={i} className={`fix fix-${fix.kind}`} title={fix.note}>
              <s>{fix.before}</s> → <strong>{fix.after}</strong>
            </span>
          ))}
        </div>
      ) : null}

      <p className="shortcuts muted">
        Shortcuts: <kbd>←</kbd>/<kbd>→</kbd> move · <kbd>space</kbd> play/pause · <kbd>R</kbd> clean this row ·{" "}
        <kbd>Ctrl</kbd>+<kbd>S</kbd> save
      </p>

      <PlaybackBar
        index={index}
        total={total}
        onPrev={() => go(-1)}
        onNext={() => go(1)}
        onRun={() => runModel()}
        onTogglePlay={() => setPlaying((p) => !p)}
        running={running}
        playing={playing}
        saved={saved}
        disabled={!row || loading}
      />
    </main>
  );
}
