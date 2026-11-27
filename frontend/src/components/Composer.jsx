import { useEffect, useLayoutEffect, useRef } from "react";
import { useStore } from "../StoreContext.js";
import Icon from "./Icon.jsx";

export default function Composer() {
  const store = useStore();
  const { state, actions } = store;
  const inputRef = useRef(null);
  const gen = store.activeChat()?.generation;
  const generating = Boolean(gen);
  // A touch keyboard has no Shift+Enter, so there Enter makes a new line.
  const touch = Boolean(globalThis.matchMedia?.("(pointer: coarse)").matches);

  useEffect(() => {
    inputRef.current?.focus();
  }, [state.inputFocusSeq]);

  // Grow with the text, up to 200px.
  useLayoutEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
  }, [state.input]);

  return (
    <form
      id="form"
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        void actions.submitComposer();
      }}
    >
      <div className="composer-box">
        <textarea
          id="input"
          ref={inputRef}
          rows={1}
          autoComplete="off"
          placeholder="Message Luma…"
          aria-label="Message"
          value={state.input}
          onChange={(e) => actions.setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !touch && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              e.currentTarget.form.requestSubmit();
            }
          }}
        />
        {/* The send button turns into Stop while THIS conversation is generating. */}
        <button
          id="send"
          className={`send-button${generating ? " stop" : ""}`}
          type={generating ? "button" : "submit"}
          title={generating ? "Stop generating" : "Send"}
          aria-label={generating ? "Stop generating" : "Send message"}
          disabled={Boolean(gen?.finishing || gen?.stopRequested)}
          onClick={generating ? actions.stopActiveGeneration : undefined}
        >
          <Icon name={generating ? "stop" : "send"} />
        </button>
      </div>
      {!touch && <p className="composer-note">Enter to send · Shift+Enter for a new line</p>}
    </form>
  );
}
