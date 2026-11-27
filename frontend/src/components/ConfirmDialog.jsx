import { useEffect, useRef } from "react";
import { useStore } from "../StoreContext.js";

/** Asks before something that cannot be undone. Escape or the backdrop cancels. */
export default function ConfirmDialog() {
  const { state, actions } = useStore();
  const cancelRef = useRef(null);
  const confirmRef = useRef(null);

  // Start on the safe choice, and give focus back afterwards.
  useEffect(() => {
    const previous = document.activeElement;
    cancelRef.current?.focus();
    return () => previous?.focus?.();
  }, []);

  function onKeyDown(event) {
    if (event.key === "Escape") {
      event.stopPropagation(); // do not also close the drawer underneath
      actions.answerConfirm(false);
    } else if (event.key === "Tab") { // keep focus inside the dialog
      event.preventDefault();
      const next = document.activeElement === cancelRef.current ? confirmRef : cancelRef;
      next.current.focus();
    }
  }

  return (
    <div className="dialog-backdrop" onClick={() => actions.answerConfirm(false)}>
      <div
        className="dialog"
        role="alertdialog"
        aria-modal="true"
        aria-describedby="dialog-message"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <p id="dialog-message">{state.confirm.message}</p>
        <div className="dialog-buttons">
          <button ref={cancelRef} type="button" className="button secondary cancel" onClick={() => actions.answerConfirm(false)}>
            Cancel
          </button>
          <button ref={confirmRef} type="button" className="button primary confirm" onClick={() => actions.answerConfirm(true)}>
            {state.confirm.action}
          </button>
        </div>
      </div>
    </div>
  );
}
