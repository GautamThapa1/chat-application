import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { PROMPT_HINTS, isVisibleMemoryResult } from "../chatStore.js";
import { useStore } from "../StoreContext.js";
import Icon from "./Icon.jsx";
import Notice from "./Notice.jsx";
import Spinner from "./Spinner.jsx";

function ActionButton({ label, icon, onClick, disabled, title }) {
  return (
    <button type="button" className="action" title={title || label} disabled={disabled} onClick={onClick}>
      <Icon name={icon} />
      <span className="label">{label}</span>
    </button>
  );
}

const COPY_STATES = {
  idle: { label: "Copy", icon: "copy", title: "Copy" },
  copied: { label: "Copied", icon: "check", title: "Copied" },
  failed: { label: "Copy failed", icon: "failed", title: "Copy is not available in this browser context" },
};

function CopyButton({ text }) {
  const [result, setResult] = useState("idle"); // "idle" | "copied" | "failed"
  useEffect(() => {
    if (result === "idle") return undefined;
    const timer = setTimeout(() => setResult("idle"), 1500);
    return () => clearTimeout(timer);
  }, [result]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setResult("copied");
    } catch {
      setResult("failed");
    }
  }
  const { label, icon, title } = COPY_STATES[result];
  return <ActionButton label={label} icon={icon} title={title} onClick={copy} />;
}

function EditBox({ chat, message, busy }) {
  const { actions } = useStore();
  return (
    <div className="edit-box">
      <textarea
        aria-label="Edit message"
        autoFocus
        value={chat.editDraft}
        onChange={(e) => actions.setEditDraft(chat, e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") actions.cancelEdit(chat);
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) actions.saveEdit(chat, message);
        }}
      />
      <div className="edit-buttons">
        <button type="button" className="button secondary" onClick={() => actions.cancelEdit(chat)}>
          Cancel
        </button>
        <button
          type="button"
          className="button primary"
          disabled={busy}
          title={busy ? "Wait for the reply to finish" : undefined}
          onClick={() => actions.saveEdit(chat, message)}
        >
          Save &amp; submit
        </button>
      </div>
    </div>
  );
}

// `live` marks what is not saved yet. Only live entries animate in, so swapping
// them for the saved messages does not replay the animation.
function Message({ chat, message, busy, isLast, live = false }) {
  const { actions } = useStore();
  const isUser = message.role === "user";
  const hasId = message.message_id !== null;
  const editing = isUser && chat.editingId !== null && chat.editingId === message.message_id;
  // The buttons are part of every message from the first paint. They are only
  // disabled (never hidden or delayed) while a reply is being produced, or for
  // the instant before a just-saved message has its database id.
  const waiting = busy ? "Wait for the reply to finish" : "";

  return (
    <div
      className={`message ${isUser ? "user" : "assistant"}${live ? " live" : ""}`}
      data-message-id={hasId ? String(message.message_id) : undefined}
    >
      {!isUser && <div className="assistant-avatar">✳</div>}
      <div className="message-body">
        {editing ? (
          <EditBox chat={chat} message={message} busy={busy} />
        ) : (
          <>
            <div className="bubble">{message.content}</div>
            <div className="actions">
              {isUser ? (
                <>
                  <ActionButton
                    label="Edit"
                    icon="edit"
                    onClick={() => actions.startEdit(chat, message)}
                    disabled={busy || !hasId}
                    title={waiting || "Edit message"}
                  />
                  <CopyButton text={message.content} />
                </>
              ) : (
                <>
                  <CopyButton text={message.content} />
                  <ActionButton
                    label="Regenerate"
                    icon="regenerate"
                    onClick={() => actions.regenerate(chat, message, isLast)}
                    disabled={busy || !hasId}
                    title={waiting || "Regenerate answer"}
                  />
                </>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function StreamingMessage({ gen }) {
  return (
    <div className="message assistant live">
      <div className="assistant-avatar">✳</div>
      <div className="message-body">
        <div className="bubble">
          {gen.text || (
            <div className="thinking" role="status">
              <span className="dots"><span /><span /><span /></span>
              <span>Thinking…</span>
            </div>
          )}
        </div>
        {/* Same buttons as a saved message, disabled until the reply is saved. */}
        <div className="actions">
          <ActionButton label="Copy" icon="copy" disabled title="Available when the reply is finished" />
          <ActionButton label="Regenerate" icon="regenerate" disabled title="Wait for the reply to finish" />
        </div>
      </div>
    </div>
  );
}

function ToolResults({ chat }) {
  const { actions } = useStore();
  return (
    <div className="tool-results" aria-label="Tool results">
      {chat.toolResults.map((item, index) => {
        if (!isVisibleMemoryResult(item)) return null;
        return (
          <div className="tool-result" key={index}>
            <div className="tool-result-title">{item.undone ? "Memory removed" : "Remembered this"}</div>
            <div className="tool-result-content">
              {[item.result.title, item.result.content].filter(Boolean).join("\n")}
            </div>
            {!item.undone && (
              <button
                type="button"
                className="button secondary"
                disabled={Boolean(item.undoing)}
                onClick={() => void actions.undoMemory(chat, index)}
              >
                {item.undoing ? "Undoing…" : "Undo"}
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

function LoadError({ chat }) {
  const { state, actions } = useStore();
  return (
    <Notice error onRetry={() => actions.reloadConversation(state.activeConversationId)}>
      {chat.loadError}
    </Notice>
  );
}

function EmptyState() {
  const { state, actions } = useStore();
  return (
    <div className="empty-state">
      <h2>{state.username ? `Hello, ${state.username}` : "Hello"}</h2>
      <p>How can I help you today?</p>
      <div className="prompt-hints">
        {PROMPT_HINTS.map(([prompt, operation]) => (
          <button key={prompt} type="button" className="hint-button" onClick={() => actions.useHint(prompt)}>
            <span>{prompt}</span>
            <span className="hint-operation">{operation}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function ThreadContent() {
  const store = useStore();
  const chat = store.activeChat();

  if (!chat) return <EmptyState />;
  // A chat that never loaded can still hold a live reply (found by a sync).
  if (!chat.loaded && !chat.messages.length) {
    return chat.loadError
      ? <LoadError chat={chat} />
      : <Notice><Spinner /> Loading chat…</Notice>;
  }

  const gen = chat.generation;
  let visible = chat.messages;
  // While regenerating, the old answer (and everything after it) is hidden:
  // the new answer streams into its place. If it fails, it comes back.
  if (gen?.mode === "regenerate") {
    const index = visible.findIndex((m) => m.message_id === gen.assistantMessageId);
    if (index >= 0) visible = visible.slice(0, index);
  }
  const busy = Boolean(gen);

  return (
    <>
      {!visible.length && !gen && <EmptyState />}
      {visible.map((message, index) => (
        <Message
          key={message.message_id ?? `pending-${index}`}
          chat={chat}
          message={message}
          busy={busy}
          isLast={index === visible.length - 1}
        />
      ))}
      {/* Normal messages are saved only when the reply ends, so the question is
          drawn from the generation itself (it also survives a page refresh). */}
      {gen?.mode === "message" && (
        <Message
          key="pending-question"
          chat={chat}
          message={{ message_id: null, role: "user", content: gen.prompt }}
          busy
          isLast={false}
          live
        />
      )}
      {gen && <StreamingMessage gen={gen} />}
      {/* Always after the reply, so it does not move when the reply is saved. */}
      {chat.toolResults.some(isVisibleMemoryResult) && <ToolResults chat={chat} />}
      {chat.loadError && <LoadError chat={chat} />}
      {chat.notice && <Notice error={chat.noticeIsError}>{chat.notice}</Notice>}
    </>
  );
}

export default function Thread() {
  const store = useStore();
  const logRef = useRef(null);
  const chat = store.activeChat();

  // Auto-scroll after every render, but only while the user is at the bottom.
  useLayoutEffect(() => {
    if (store.scroll.stickToBottom && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  });

  return (
    <div
      id="log"
      aria-label="Conversation"
      ref={logRef}
      onScroll={(e) => {
        const log = e.currentTarget;
        store.scroll.stickToBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
      }}
    >
      <div id="thread" className="thread" aria-live="polite">
        <ThreadContent />
        {!chat && store.state.blankError && <Notice error>{store.state.blankError}</Notice>}
      </div>
    </div>
  );
}
