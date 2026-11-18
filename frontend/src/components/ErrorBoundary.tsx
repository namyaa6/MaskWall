import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * Catches a crash anywhere below it and shows a readable message instead of a blank page. Wrapped around the routes (App.tsx)
 * with `resetKey` = the current path, so moving to another page clears the error. The top menu stays outside it and keeps working.
 * Only the error's name and message are shown (never message text or personal data: nothing here has access to them).
 */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("MaskWall UI crashed:", error, info.componentStack);
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="card error-card crash" role="alert">
        <h1>Something went wrong, please refresh</h1>
        <p>This page hit an unexpected error. Your messages were not affected: nothing is stored, and nothing more was sent.</p>
        <p>
          <button type="button" className="btn btn-primary" onClick={() => window.location.reload()}>
            Refresh the page
          </button>
        </p>
        <details>
          <summary>Technical details</summary>
          <p className="mono">{this.state.error.name}: {this.state.error.message}</p>
        </details>
      </div>
    );
  }
}
