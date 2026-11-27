import { useEffect } from "react";
import { DEFAULT_TITLE } from "./chatStore.js";
import { useStore } from "./StoreContext.js";
import AuthScreen from "./components/AuthScreen.jsx";
import Composer from "./components/Composer.jsx";
import Icon from "./components/Icon.jsx";
import LoadingScreen from "./components/LoadingScreen.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Thread from "./components/Thread.jsx";

function ChatApp() {
  const { state, actions } = useStore();
  const listed = state.conversations.find((c) => c.conversation_id === state.activeConversationId);

  useEffect(() => {
    document.title = listed?.title ? `${listed.title} · Luma` : "Luma Chat";
  }, [listed?.title]);

  return (
    <div id="chat" className={`app${state.sidebarOpen ? " sidebar-open" : ""}`}>
      <div id="scrim" className="scrim" onClick={actions.closeSidebar} />
      <Sidebar />
      <section className="chat">
        <header className="topbar">
          <button id="menu" className="icon-button" type="button" aria-label="Open conversations" onClick={actions.openSidebar}>
            <Icon name="menu" />
          </button>
          <h1 id="chat-title">{listed?.title || DEFAULT_TITLE}</h1>
        </header>
        <Thread />
        <Composer />
      </section>
    </div>
  );
}

export default function App() {
  const { state } = useStore();
  if (state.view === "auth") return <AuthScreen />;
  if (state.view === "chat") return <ChatApp />;
  return <LoadingScreen />;
}
