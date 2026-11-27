/** The one status and error message. `onRetry` adds a Retry button. */
export default function Notice({ error = false, onRetry, children }) {
  return (
    <div className={error ? "notice error" : "notice"} role={error ? "alert" : "status"}>
      <span>{children}</span>
      {onRetry && (
        <button type="button" className="link-button retry" onClick={onRetry}>Retry</button>
      )}
    </div>
  );
}
