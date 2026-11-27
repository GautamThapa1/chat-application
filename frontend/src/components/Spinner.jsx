/** The one loading indicator. Pass `label` when no text next to it says what is loading. */
export default function Spinner({ label }) {
  return label
    ? <span className="spinner" role="status" aria-label={label} />
    : <span className="spinner" aria-hidden="true" />;
}
