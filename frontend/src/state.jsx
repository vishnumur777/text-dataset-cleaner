import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "./api";

const SESSION_KEY = "tdc.session";
const SessionCtx = createContext(null);
const ToastCtx = createContext(null);

export function SessionProvider({ children }) {
  const [summary, setSummary] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [restoreError, setRestoreError] = useState(null);

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
    } catch (err) {
      setHealth({ ok: false, error: err.message });
    }
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      const id = localStorage.getItem(SESSION_KEY);
      if (id) {
        try {
          const data = await api.session(id);
          if (alive) setSummary(data);
        } catch (err) {
          localStorage.removeItem(SESSION_KEY);
          if (alive && err instanceof ApiError && err.status !== 404) setRestoreError(err);
        }
      }
      if (alive) setLoading(false);
      loadHealth();
    })();
    return () => {
      alive = false;
    };
  }, [loadHealth]);

  const adopt = useCallback((payload) => {
    const session = payload?.session ?? payload;
    if (!session?.id) return null;
    localStorage.setItem(SESSION_KEY, session.id);
    setSummary(session);
    return session;
  }, []);

  const refresh = useCallback(async () => {
    if (!summary?.id) return null;
    try {
      const data = await api.session(summary.id);
      setSummary(data);
      return data;
    } catch {
      return null;
    }
  }, [summary?.id]);

  const clear = useCallback(() => {
    localStorage.removeItem(SESSION_KEY);
    setSummary(null);
  }, []);

  const value = useMemo(
    () => ({ summary, setSummary, adopt, refresh, clear, health, loadHealth, loading, restoreError }),
    [summary, adopt, refresh, clear, health, loadHealth, loading, restoreError]
  );

  return <SessionCtx.Provider value={value}>{children}</SessionCtx.Provider>;
}

export function useSession() {
  const ctx = useContext(SessionCtx);
  if (!ctx) throw new Error("useSession must be used inside SessionProvider");
  return ctx;
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const seq = useRef(0);

  const notify = useCallback((message, tone = "info", hint = "") => {
    const id = ++seq.current;
    setToasts((list) => [...list, { id, message, hint, tone }]);
    setTimeout(() => setToasts((list) => list.filter((t) => t.id !== id)), tone === "error" ? 9000 : 5000);
  }, []);

  const dismiss = useCallback((id) => setToasts((list) => list.filter((t) => t.id !== id)), []);

  return (
    <ToastCtx.Provider value={notify}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.tone}`}>
            <div>
              <strong>{t.message}</strong>
              {t.hint ? <p>{t.hint}</p> : null}
            </div>
            <button className="toast-close" onClick={() => dismiss(t.id)} aria-label="Dismiss">×</button>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export function useToast() {
  return useContext(ToastCtx);
}
