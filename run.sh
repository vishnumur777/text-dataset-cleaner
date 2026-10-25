#!/usr/bin/env bash
# Start the whole stack: local model server (optional) + FastAPI + built UI.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${TDC_PORT:-8000}"
MODEL="${TDC_MODEL_PATH:-models/gemma-3-1b-it-Q4_K_M.gguf}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/.llama-app/llama}"

say() { printf '\n\033[1;35m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- python env
if [ ! -d .venv ]; then
  say "Creating Python environment"
  python3 -m venv .venv --system-site-packages
fi
say "Installing backend dependencies"
.venv/bin/pip install -q -r backend/requirements.txt

# ------------------------------------------------------------------- samples
if [ ! -f samples/messy_tamil.csv ]; then
  .venv/bin/python scripts/make_sample.py
fi

# --------------------------------------------------------------------- model
if curl -sf -m 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
  say "llama.cpp server already running on :8080"
elif [ -x "$LLAMA_BIN" ] && [ -f "$MODEL" ]; then
  say "Starting llama.cpp with $(basename "$MODEL")"
  mkdir -p .run
  nohup "$LLAMA_BIN" serve -m "$MODEL" --host 127.0.0.1 --port 8080 -c 4096 > .run/llama-server.log 2>&1 &
  echo $! > .run/llama-server.pid
  for _ in $(seq 1 40); do
    curl -sf -m 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf -m 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1 \
    && say "Model server ready" \
    || warn "Model server did not come up; the rule engine will still work (see .run/llama-server.log)"
else
  warn "No llama.cpp binary ($LLAMA_BIN) or model ($MODEL) found."
  warn "The app still runs with the deterministic rule engine; set LLAMA_BIN/TDC_MODEL_PATH to enable Gemma."
fi

# ------------------------------------------------------------------ frontend
if [ ! -f frontend/dist/index.html ] || [ "${TDC_REBUILD_UI:-0}" = "1" ]; then
  if command -v npm >/dev/null 2>&1; then
    say "Building the UI"
    (cd frontend && npm install --no-audit --no-fund --silent && npm run build >/dev/null)
  else
    warn "npm not found and frontend/dist is missing: the API will run, but there is no UI."
  fi
fi

# ------------------------------------------------------------------- server
say "Serving on http://127.0.0.1:$PORT  (Ctrl+C to stop)"
echo "Open http://127.0.0.1:$PORT/file-select"
exec .venv/bin/python -m backend.app
