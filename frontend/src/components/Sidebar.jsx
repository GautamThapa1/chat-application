import { DEFAULT_TITLE, groupLabel } from "../chatStore.js";
import { useStore } from "../StoreContext.js";
import Icon from "./Icon.jsx";

function ConversationList() {
  const { state, actions } = useStore();

  if (state.searchActive) {
    if (!state.searchResults.length) {
      // While searching or after an error the status line above says so.
      if (state.searchPending || state.searchFailed) return null;
      return <p className="conversation-empty">No matching chats</p>;
    }
    return state.searchResults.map((result) => (
      <button
        key={result.conversation_id}
        type="button"
        className="conversation-search-result"
        onClick={() => actions.selectConversation(result.conversation_id)}
      >
        <strong>{result.title || DEFAULT_TITLE}</strong>
        <span>{result.excerpt || ""}</span>
      </button>
    ));
  }

  if (!state.conversations.length) {
    if (state.conversationsLoading) {
      return (
        <p className="conversation-empty" role="status">
          <span className="spinner" aria-hidden="true" /> Loading chats…
        </p>
      );
    }
    if (state.listError) return null;
    return <p className="conversation-empty">No chats yet</p>;
  }

  const rows = [];
  let lastGroup = "";
  for (const conversation of state.conversations) {
    const group = groupLabel(conversation.updated_at);
    if (group !== lastGroup) {
      rows.push(<div key={`group-${group}`} className="conversation-group">{group}</div>);
      lastGroup = group;
    }
    const id = conversation.conversation_id;
    const title = conversation.title || DEFAULT_TITLE;
    const active = id === state.activeConversationId;
    rows.push(
      <div key={id} className={`conversation-item${active ? " active" : ""}`}>
        <button
          type="button"
          className="conversation-select"
          title={title}
          aria-current={active ? "page" : undefined}
          onClick={() => actions.selectConversation(id)}
        >
          <span className="conversation-title">{title}</span>
          {/* A spinner marks chats that are still generating in the background. */}
          {state.chats.get(id)?.generation && (
            <span className="spinner" role="status" aria-label="Generating a reply" />
          )}
        </button>
        <button
          type="button"
          className="conversation-delete"
          title="Delete chat"
          aria-label={`Delete ${title}`}
          onClick={() => void actions.deleteConversation(id)}
        >
          <Icon name="trash" />
        </button>
      </div>,
    );
  }
  return rows;
}

export default function Sidebar() {
  const { state, actions } = useStore();
  const { reminders } = state;

  return (
    <aside id="sidebar" className="sidebar" aria-label="Conversation history">
      <div className="sidebar-top">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">✳</div>
          <span>Luma</span>
        </div>
      </div>
      <button id="new-chat" className="new-chat" type="button" onClick={actions.startNewChat}>
        <Icon name="plus" />
        New chat
      </button>
      <form
        id="conversation-search-form"
        className="conversation-search-form"
        role="search"
        onSubmit={(e) => {
          e.preventDefault();
          void actions.searchPastConversations();
        }}
      >
        <input
          id="conversation-search"
          type="search"
          placeholder="Search past chats"
          aria-label="Search past conversations"
          maxLength={2000}
          value={state.searchQuery}
          onChange={(e) => actions.setSearchQuery(e.target.value)}
        />
        <button type="submit">Search</button>
      </form>
      <p id="conversation-search-status" role="status" aria-live="polite">{state.searchStatus}</p>
      {state.listError && (
        <div className="sidebar-error" role="alert">
          <span>{state.listError}</span>
          <button type="button" className="link-button retry" onClick={() => void actions.loadConversations()}>
            Retry
          </button>
        </div>
      )}
      {state.sidebarError && <div className="sidebar-error" role="alert">{state.sidebarError}</div>}
      <nav id="conversations" aria-label="Conversations">
        <ConversationList />
      </nav>
      <button
        id="load-more-conversations"
        type="button"
        hidden={state.searchActive || state.conversationOffset >= state.conversationTotalCount}
        disabled={state.conversationsLoading}
        onClick={() => void actions.loadMoreConversations()}
      >
        Show more
      </button>
      <button
        id="enable-task-reminders"
        type="button"
        hidden={!reminders.visible}
        disabled={reminders.busy}
        onClick={() => void actions.toggleTaskReminders()}
      >
        {reminders.enabled ? "Disable task reminders" : "Enable task reminders"}
      </button>
      <p id="task-reminder-status" role="status" aria-live="polite">{reminders.status}</p>
      <div className="sidebar-user">
        <div id="user-avatar" className="avatar" aria-hidden="true">
          {(state.username[0] || "?").toUpperCase()}
        </div>
        <div className="user-meta">
          <div id="user-name">{state.username || "Guest"}</div>
          <div id="connection" className={`connection ${state.connection.state}`} role="status">
            {state.connection.label}
          </div>
        </div>
        <button
          id="logout"
          className="icon-button"
          type="button"
          title="Log out"
          aria-label="Log out"
          onClick={() => void actions.logout()}
        >
          <Icon name="logout" />
        </button>
      </div>
    </aside>
  );
}
