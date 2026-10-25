const BASE = "/api";

export class ApiError extends Error {
  constructor({ code, message, hint, status }) {
    super(message || "Something went wrong.");
    this.code = code || "error";
    this.hint = hint || "";
    this.status = status || 0;
  }
}

async function parse(res) {
  const type = res.headers.get("content-type") || "";
  if (type.includes("application/json")) {
    return res.json().catch(() => ({}));
  }
  const text = await res.text();
  return { raw: text };
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json", ...(options.headers || {}) }
        : options.headers,
      ...options,
    });
  } catch (cause) {
    throw new ApiError({
      code: "network",
      message: "The cleaner service is not reachable.",
      hint: "Start it with `python -m backend.app` (or ./run.sh) and try again.",
    });
  }

  const data = await parse(res);
  if (!res.ok) {
    const err = data?.error || {};
    throw new ApiError({
      code: err.code || `http_${res.status}`,
      message: err.message || (res.status === 404 ? "That item was not found." : "The request failed."),
      hint: err.hint || "",
      status: res.status,
    });
  }
  return data;
}

const get = (path) => request(path);
const post = (path, body) => request(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  health: () => get("/health"),
  samples: () => get("/samples"),
  openSample: (name) => post(`/sample/open?name=${encodeURIComponent(name)}`),

  hfSearch: (q) => get(`/hf/search?q=${encodeURIComponent(q)}`),
  hfFiles: (repo) => get(`/hf/files?repo=${encodeURIComponent(repo)}`),
  hfOpen: (repo, path) => post("/hf/open", { repo, path }),
  hfOpenJob: (repo, path, revision = "main") => post("/hf/open-job", { repo, path, revision }),

  kaggleSearch: (q) => get(`/kaggle/search?q=${encodeURIComponent(q)}`),
  kaggleFiles: (ref) => get(`/kaggle/files?ref=${encodeURIComponent(ref)}`),
  kaggleOpen: (ref, path) => post("/kaggle/open", { ref, path }),
  kaggleOpenJob: (ref, path) => post("/kaggle/open-job", { ref, path }),
  kaggleConnect: (username, key) => post("/kaggle/connect", { username, key }),

  job: (id) => get(`/jobs/${encodeURIComponent(id)}`),

  upload: (file, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        let body = {};
        try {
          body = JSON.parse(xhr.responseText || "{}");
        } catch {
          body = {};
        }
        if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
        const err = body.error || {};
        reject(new ApiError({
          code: err.code || `http_${xhr.status}`,
          message: err.message || "The upload failed.",
          hint: err.hint || "",
          status: xhr.status,
        }));
      };
      xhr.onerror = () =>
        reject(new ApiError({
          code: "network",
          message: "The upload could not reach the server.",
          hint: "Check that the backend is running, then retry.",
        }));
      const form = new FormData();
      form.append("file", file);
      xhr.send(form);
    }),

  session: (id) => get(`/session/${id}`),
  setColumn: (id, column) => post(`/session/${id}/column`, { column }),
  row: (id, index) => get(`/session/${id}/row?index=${index}`),
  clean: (id, index, mode) => post(`/session/${id}/clean`, { index, mode }),
  save: (id, index, text) => post(`/session/${id}/save`, { index, text }),
  publish: (id, body) => post(`/session/${id}/publish`, body),

  preclean: (id, mode = "llm", workers) => post(`/session/${id}/preclean`, { mode, workers }),
  precleanStatus: (id) => get(`/session/${id}/preclean`),
  precleanCancel: (id) => post(`/session/${id}/preclean/cancel`),

  async download(id, format, filenameHint = "cleaned") {
    let res;
    try {
      res = await fetch(`${BASE}/session/${id}/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ format }),
      });
    } catch {
      throw new ApiError({
        code: "network",
        message: "The export could not reach the server.",
        hint: "Check the backend is still running and retry.",
      });
    }
    if (!res.ok) {
      const data = await parse(res);
      const err = data?.error || {};
      throw new ApiError({
        code: err.code || "export_failed",
        message: err.message || "The export failed.",
        hint: err.hint || "",
        status: res.status,
      });
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filenameHint}.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  },
};
