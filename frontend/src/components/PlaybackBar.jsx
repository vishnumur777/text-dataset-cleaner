const IconPrev = () => (
  <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
    <path d="M7 5h2v14H7zM19 5v14l-9-7z" fill="currentColor" />
  </svg>
);
const IconNext = () => (
  <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
    <path d="M15 5h2v14h-2zM5 5l9 7-9 7z" fill="currentColor" />
  </svg>
);
const IconPlay = () => (
  <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
    <path d="M8 5l11 7-11 7z" fill="currentColor" />
  </svg>
);
const IconPause = () => (
  <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
    <path d="M7 5h3v14H7zM14 5h3v14h-3z" fill="currentColor" />
  </svg>
);

export function PlaybackBar({
  index,
  total,
  onPrev,
  onNext,
  onRun,
  onTogglePlay,
  running,
  playing,
  saved,
  disabled,
}) {
  return (
    <div className="playback" role="toolbar" aria-label="Row navigation and cleaning controls">
      <div className="playback-meta">
        <span className="row-counter">
          Row <strong>{total ? index + 1 : 0}</strong> of <strong>{total}</strong>
        </span>
        <span className="saved-counter">{saved} saved</span>
      </div>

      <div className="playback-buttons">
        <button
          className="round"
          onClick={onPrev}
          disabled={disabled || index <= 0}
          title="Previous row (←)"
          aria-label="Previous row"
        >
          <IconPrev />
        </button>

        <button
          className={`round round-main ${playing ? "is-playing" : ""}`}
          onClick={onTogglePlay}
          disabled={disabled || running}
          title={playing ? "Pause after this row (space)" : "Run the model through the rows (space)"}
          aria-label={playing ? "Pause" : "Play"}
        >
          {running ? <span className="spinner" /> : playing ? <IconPause /> : <IconPlay />}
        </button>

        <button
          className="round"
          onClick={onRun}
          disabled={disabled || running}
          title="Clean this row with the local model (R)"
          aria-label="Clean this row"
        >
          <span className="spark">✦</span>
        </button>

        <button
          className="round"
          onClick={onNext}
          disabled={disabled || index >= total - 1}
          title="Next row (→)"
          aria-label="Next row"
        >
          <IconNext />
        </button>
      </div>

      <div className="playback-progress" aria-hidden="true">
        <div className="bar" style={{ width: total ? `${((index + 1) / total) * 100}%` : "0%" }} />
      </div>
    </div>
  );
}
