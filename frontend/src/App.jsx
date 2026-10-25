import { Component } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Header } from "./components/Header";
import CleanData from "./pages/CleanData";
import FileSelect from "./pages/FileSelect";
import Upload from "./pages/Upload";
import { SessionProvider, ToastProvider } from "./state";

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Keep the console useful for debugging without showing a raw stack in the UI.
    console.error("UI error", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="fatal">
          <h1>The interface hit an unexpected error</h1>
          <p>Your dataset is safe. Reload the page to continue; if it happens again, restart the server.</p>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>
            Reload
          </button>
          <details>
            <summary>Technical detail</summary>
            <pre>{String(this.state.error?.message || this.state.error)}</pre>
          </details>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <SessionProvider>
          <ToastProvider>
            <div className="app">
              <Header />
              <Routes>
                <Route path="/" element={<Navigate to="/file-select" replace />} />
                <Route path="/file-select" element={<FileSelect />} />
                <Route path="/clean-data" element={<CleanData />} />
                <Route path="/upload" element={<Upload />} />
                <Route path="*" element={<Navigate to="/file-select" replace />} />
              </Routes>
              <footer className="app-footer">
                <span>Runs entirely on this machine · Gemma 3 1B via llama.cpp · pandas</span>
              </footer>
            </div>
          </ToastProvider>
        </SessionProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
