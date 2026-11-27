import { useStore } from "../StoreContext.js";

/** Full-page progress, for the moments between the login screen and the chat. */
export default function LoadingScreen() {
  const { state, actions } = useStore();
  return (
    <section id="loading" className="auth">
      <div className="loading-card">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">✳</div>
          <span>Luma Chat</span>
        </div>
        <div className="loading-status" role="status">
          {!state.loadingFailed && <span className="spinner" aria-hidden="true" />}
          <span>{state.loadingLabel}</span>
        </div>
        {state.loadingFailed && (
          <button type="button" className="button primary retry" onClick={() => void actions.checkSession()}>
            Retry
          </button>
        )}
      </div>
    </section>
  );
}
