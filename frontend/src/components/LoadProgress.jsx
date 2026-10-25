import { bytes } from "../format";

const PHASE_LABEL = {
  starting: "Preparing the download...",
  download: "Downloading the file",
  extract: "Unpacking the archive",
  parse: "Reading the table",
  ready: "Ready to clean",
};

function humanEta(seconds) {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s left`;
  return `${Math.round(seconds / 60)}m left`;
}

/** Animated progress for a background download/parse job.
 *
 * The bar is determinate while the server reports Content-Length — the server
 * now publishes the total before the first byte arrives, so the bar appears
 * immediately — and switches to an indeterminate shimmer during parsing, where
 * there is no byte count. Speed and ETA come from the server's own samples so a
 * big file visibly moves.
 */
export function LoadProgress({ job, title = "Loading dataset", fallback = "" }) {
  const percent = job && job.total > 0 ? Math.min(100, Math.round((job.done / job.total) * 100)) : null;
  const message = job?.message || fallback || "Working...";
  const heading = PHASE_LABEL[job?.phase] || title;
  const showBytes = Boolean(job && job.total > 0);
  const speed = job?.speed_bps > 0 && job?.done > 0 && job.done < job.total ? `${bytes(job.speed_bps)}/s` : "";
  const eta = humanEta(job?.eta_s);

  return (
    <div className="load-progress" role="status" aria-live="polite" aria-busy="true">
      <div className="load-progress-head">
        <span className="spinner" />
        <strong>{heading}</strong>
        <span className="load-percent">{percent != null ? `${percent}%` : "…"}</span>
      </div>
      <div className={`load-bar ${percent == null ? "is-indeterminate" : ""}`}>
        <div className="bar" style={percent == null ? undefined : { width: `${percent}%` }} />
      </div>
      <div className="load-progress-foot">
        <span>{message}</span>
        <span>
          {showBytes ? `${bytes(job.done, "0 B")} / ${bytes(job.total, "0 B")}` : ""}
          {speed ? ` · ${speed}` : ""}
          {eta ? ` · ${eta}` : ""}
        </span>
      </div>
    </div>
  );
}