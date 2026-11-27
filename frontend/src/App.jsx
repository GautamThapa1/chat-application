import { useEffect } from "react";
import { DEFAULT_TITLE } from "./chatStore.js";
import { useStore } from "./StoreContext.js";
import AuthScreen from "./components/AuthScreen.jsx";
import Composer from "./components/Composer.jsx";
import ConfirmDialog from "./components/ConfirmDialog.jsx";
import Icon from "./components/Icon.jsx";
import LoadingScreen from "./components/LoadingScreen.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Thread from "./components/Thread.jsx";

function ChatApp() {
  const { state, actions } = useStore();
  const listed = state.conversations.find((c) => c.conversation_id === state.activeConversationId);

  // A chat opened from search may be older than the loaded list.
  const title = listed?.title || state.chats.get(state.activeConversationId)?.conversation?.title;

  useEffect(() => {
    document.title = title ? `${title} · Luma` : "Luma Chat";
  }, [title]);

  useEffect(() => {
    if (!state.sidebarOpen) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") actions.closeSidebar();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [state.sidebarOpen, actions]);

  return (
    <div id="chat" className={`app${state.sidebarOpen ? " sidebar-open" : ""}`}>
      <div id="scrim" className="scrim" onClick={actions.closeSidebar} />
      <Sidebar />
      <section className="chat">
        <header className="topbar">
          <button id="menu" className="icon-button" type="button" aria-label="Open conversations" onClick={actions.openSidebar}>
            <Icon name="menu" />
          </button>
          <h1 id="chat-title">{title || DEFAULT_TITLE}</h1>
        </header>
        <Thread />
        <Composer />
      </section>
      {state.confirm && <ConfirmDialog />}
    </div>
  );
}

export default function App() {
  const { state } = useStore();
  if (state.view === "auth") return <AuthScreen />;
  if (state.view === "chat") return <ChatApp />;
  return <LoadingScreen />;
}
