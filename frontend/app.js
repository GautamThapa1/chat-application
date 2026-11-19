// Same host as the page, so the browser sends the session cookie. Becomes
// wss:// automatically once the site is served over HTTPS.
const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

const logEl = document.getElementById("log");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const connectionEl = document.getElementById("connection");
const clearButtonEl = document.getElementById("clear");
const logoutButtonEl = document.getElementById("logout");
const newChatButtonEl = document.getElementById("new-chat");
const conversationsEl = document.getElementById("conversations");

const chatEl = document.getElementById("chat");
const authEl = document.getElementById("auth");
const authFormEl = document.getElementById("auth-form");
const authTitleEl = document.getElementById("auth-title");
const authUsernameEl = document.getElementById("auth-username");
const authPasswordEl = document.getElementById("auth-password");
const authErrorEl = document.getElementById("auth-error");
const authSubmitEl = document.getElementById("auth-submit");
const authSwitchEl = document.getElementById("auth-switch");
const authSwitchTextEl = document.getElementById("auth-switch-text");

const pendingBubbles = new Map();
let nextRequestId = 0;
let ws = null;
let authMode = "login";
let conversations = [];
let activeConversationId = null;
let conversationLoadGeneration = 0;
let userStateGeneration = 0;

const promptHints = [
  ["Remember that my backend uses FastAPI", "Save note"],
  ["Show my notes", "Retrieve notes"],
  ["Create a task to finish gRPC", "Create task"],
  ["Show my tasks", "List tasks"],
  ["Mark task 1 completed", "Complete task"],
  ["Hello", "General chat"],
];

async function api(path, method = "GET", body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  let data = null;
  try {
    data = await response.json();
  } catch {
    // Non-JSON response bodies have no structured error details.
  }
  return { ok: response.ok, status: response.status, data };
}

function errorMessage(data) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => `${item.loc?.[item.loc.length - 1] ?? "input"}: ${item.msg}`)
      .join(" · ");
  }
  return "Something went wrong. Please try again.";
}

function clearUserState() {
  userStateGeneration += 1;
  const socket = ws;
  ws = null;
  socket?.close();
  conversationLoadGeneration += 1;
  conversations = [];
  activeConversationId = null;
  renderConversations();
  resetChat();
}

function showAuth(message = "") {
  clearUserState();
  chatEl.hidden = true;
  authEl.hidden = false;
  authErrorEl.textContent = message;
  authUsernameEl.focus();
}

function showChat() {
  authEl.hidden = true;
  chatEl.hidden = false;
  inputEl.focus();
}

function resetChat() {
  logEl.replaceChildren(createEmptyState());
  pendingBubbles.clear();
}

function setAuthMode(mode) {
  authMode = mode;
  const registering = mode === "register";
  authTitleEl.textContent = registering ? "Create account" : "Log in";
  authSubmitEl.textContent = registering ? "Create account" : "Log in";
  authSwitchTextEl.textContent = registering ? "Already have an account?" : "No account yet?";
  authSwitchEl.textContent = registering ? "Log in" : "Create one";
  authPasswordEl.autocomplete = registering ? "new-password" : "current-password";
  authErrorEl.textContent = "";
}

authSwitchEl.addEventListener("click", () => {
  setAuthMode(authMode === "login" ? "register" : "login");
});

authFormEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = authUsernameEl.value.trim();
  const password = authPasswordEl.value;
  if (!username || !password) {
    authErrorEl.textContent = "Enter a username and password.";
    return;
  }

  authSubmitEl.disabled = true;
  authErrorEl.textContent = "";
  try {
    if (authMode === "register") {
      const registered = await api("/api/register", "POST", { username, password });
      if (!registered.ok) {
        authErrorEl.textContent = errorMessage(registered.data);
        return;
      }
    }

    const login = await api("/api/login", "POST", { username, password });
    if (!login.ok) {
      authErrorEl.textContent = errorMessage(login.data);
      return;
    }

    authPasswordEl.value = "";
    clearUserState();
    setAuthMode("login");
    showChat();
    await loadConversations();
  } catch {
    authErrorEl.textContent = "Could not reach the server.";
  } finally {
    authSubmitEl.disabled = false;
  }
});

logoutButtonEl.addEventListener("click", async () => {
  clearUserState();
  setConnection("Disconnected", "error");
  let message = "";
  try {
    const result = await api("/api/logout", "POST");
    if (!result.ok) {
      message = `Could not end the server session: ${errorMessage(result.data)}`;
    }
  } catch {
    message = "Could not reach the server to end your session.";
  }
  showAuth(message);
});

async function checkSession() {
  try {
    const me = await api("/api/me");
    if (me.ok) {
      clearUserState();
      showChat();
      await loadConversations();
    } else if (me.status === 401) {
      showAuth();
    } else {
      showAuth(errorMessage(me.data));
    }
  } catch {
    showAuth("Could not reach the server.");
  }
}

async function loadConversations() {
  clearUserState();
  const generation = userStateGeneration;
  try {
    const result = await api("/api/conversations");
    if (generation !== userStateGeneration) return;
    if (!result.ok) {
      if (result.status === 401) {
        showAuth("Your session has ended. Please log in again.");
      } else {
        addLine("system", errorMessage(result.data));
        setConnection("Could not load chats", "error");
      }
      return;
    }

    conversations = result.data.conversations;
    renderConversations();
    setConnection(
      conversations.length ? "Select a chat" : "Start a new chat",
      "",
    );
    if (!conversations.length) resetChat();
  } catch {
    if (generation === userStateGeneration) {
      addLine("system", "Could not reach the server to load chats.");
      setConnection("Could not load chats", "error");
    }
  }
}

function renderConversations() {
  conversationsEl.replaceChildren();
  if (!conversations.length) {
    const empty = document.createElement("p");
    empty.className = "conversation-empty";
    empty.textContent = "No chats yet";
    conversationsEl.appendChild(empty);
    return;
  }

  for (const conversation of conversations) {
    const item = document.createElement("div");
    item.className = "conversation-item";
    if (conversation.conversation_id === activeConversationId) item.classList.add("active");

    const select = document.createElement("button");
    select.type = "button";
    select.className = "conversation-select";
    select.textContent = conversation.title || "New chat";
    select.title = conversation.title || "New chat";
    select.addEventListener("click", () => {
      void selectConversation(conversation.conversation_id);
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "conversation-delete";
    remove.textContent = "Delete";
    remove.setAttribute("aria-label", `Delete ${conversation.title || "conversation"}`);
    remove.addEventListener("click", () => {
      void deleteConversation(conversation.conversation_id);
    });

    item.append(select, remove);
    conversationsEl.appendChild(item);
  }
}

newChatButtonEl.addEventListener("click", () => {
  void createConversation();
});

async function createConversation() {
  const generation = userStateGeneration;
  newChatButtonEl.disabled = true;
  try {
    const result = await api("/api/conversations", "POST", { title: "New chat" });
    if (generation !== userStateGeneration) return;
    if (!result.ok) {
      if (result.status === 401) {
        showAuth("Your session has ended. Please log in again.");
      } else {
        addLine("system", errorMessage(result.data));
      }
      return;
    }
    conversations = [result.data.conversation, ...conversations];
    renderConversations();
    await selectConversation(result.data.conversation.conversation_id);
  } catch {
    if (generation === userStateGeneration) {
      addLine("system", "Could not reach the server to create a chat.");
    }
  } finally {
    newChatButtonEl.disabled = false;
  }
}

async function selectConversation(conversationId) {
  const userGeneration = userStateGeneration;
  const socket = ws;
  ws = null;
  socket?.close();
  activeConversationId = conversationId;
  const generation = ++conversationLoadGeneration;
  resetChat();
  renderConversations();
  setConnection("Loading chat", "");

  try {
    const result = await api(`/api/conversations/${conversationId}`);
    if (
      userGeneration !== userStateGeneration
      || generation !== conversationLoadGeneration
    ) return;
    if (!result.ok) {
      if (result.status === 401) {
        showAuth("Your session has ended. Please log in again.");
      } else {
        addLine("system", errorMessage(result.data));
        setConnection("Could not load chat", "error");
      }
      return;
    }

    if (result.data.messages.length) {
      for (const message of result.data.messages) {
        addLine(message.role, message.content);
      }
    } else {
      resetChat();
    }
    connectWebSocket(conversationId);
  } catch {
    if (
      userGeneration === userStateGeneration
      && generation === conversationLoadGeneration
    ) {
      addLine("system", "Could not reach the server to load this chat.");
      setConnection("Could not load chat", "error");
    }
  }
}

async function deleteConversation(conversationId) {
  const generation = userStateGeneration;
  try {
    const result = await api(`/api/conversations/${conversationId}`, "DELETE");
    if (generation !== userStateGeneration) return;
    if (!result.ok) {
      if (result.status === 401) {
        showAuth("Your session has ended. Please log in again.");
      } else {
        setConnection(errorMessage(result.data), "error");
      }
      return;
    }

    conversations = conversations.filter(
      (item) => item.conversation_id !== conversationId,
    );
    if (activeConversationId === conversationId) {
      const socket = ws;
      ws = null;
      socket?.close();
      activeConversationId = null;
      resetChat();
    }
    renderConversations();
    if (activeConversationId === null && conversations.length) {
      await selectConversation(conversations[0].id);
    } else if (activeConversationId === null) {
      setConnection("Start a new chat", "");
    }
  } catch {
    if (generation === userStateGeneration) {
      setConnection("Could not reach the server to delete this chat.", "error");
    }
  }
}

function connectWebSocket(conversationId) {
  if (activeConversationId !== conversationId) return;
  const socket = new WebSocket(WS_URL);
  ws = socket;
  setConnection("Connecting", "");

  socket.onopen = () => {
    if (socket !== ws || activeConversationId !== conversationId) return;
    setConnection("Connected", "connected");
    addLine("system", "connected");
  };

  socket.onclose = async () => {
    if (socket !== ws || activeConversationId !== conversationId) return;
    setConnection("Disconnected", "error");
    removePendingBubbles();
    addLine("system", "disconnected");

    try {
      const me = await api("/api/me");
      if (me.status === 401) {
        showAuth("Your session has ended. Please log in again.");
      }
    } catch {
      setConnection("Disconnected: server unavailable", "error");
    }
  };

  socket.onerror = (err) => {
    if (socket !== ws || activeConversationId !== conversationId) return;
    setConnection("Connection error", "error");
    removePendingBubbles();
    addLine("system", `error: ${err.message || "websocket error"}`);
  };

  socket.onmessage = (event) => {
    if (socket !== ws || activeConversationId !== conversationId) return;
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      addLine("system", `raw: ${event.data}`);
      return;
    }

    if (msg.type === "start") {
      const bubble = pendingBubbles.get(msg.request_id);
      if (bubble) {
        bubble.classList.remove("thinking");
        bubble.replaceChildren();
      }
    } else if (msg.type === "chunk") {
      const bubble = pendingBubbles.get(msg.request_id);
      if (bubble) {
        bubble.classList.remove("thinking");
        bubble.textContent += msg.content;
        logEl.scrollTop = logEl.scrollHeight;
      }
    } else if (msg.type === "complete") {
      pendingBubbles.delete(msg.request_id);
      refreshConversationTitle(conversationId);
    } else if (msg.type === "error" || msg.error) {
      const bubble = pendingBubbles.get(msg.request_id);
      bubble?.remove();
      pendingBubbles.delete(msg.request_id);
      addLine("system", `error: ${msg.error}`);
    } else {
      addLine(msg.role || "assistant", msg.content);
    }
  };
}

async function refreshConversationTitle(conversationId) {
  const generation = userStateGeneration;
  try {
    const result = await api(`/api/conversations/${conversationId}`);
    if (generation !== userStateGeneration || activeConversationId !== conversationId) return;
    if (!result.ok) {
      if (result.status === 401) {
        showAuth("Your session has ended. Please log in again.");
      } else {
        setConnection("Could not refresh chat title", "error");
      }
      return;
    }
    conversations = conversations.map((item) =>
      item.conversation_id === conversationId ? result.data.conversation : item,
    );
    renderConversations();
  } catch {
    if (generation === userStateGeneration) {
      setConnection("Chat saved; title refresh failed", "error");
    }
  }
}

clearButtonEl.addEventListener("click", () => {
  resetChat();
});

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  if (!activeConversationId) {
    addLine("system", "Create or select a chat before sending a message.");
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addLine("system", "Not connected — try again in a moment.");
    return;
  }

  const requestId = String(++nextRequestId);
  ws.send(JSON.stringify({
    type: "message",
    request_id: requestId,
    conversation_id: activeConversationId,
    content: text,
  }));
  addLine("user", text);
  pendingBubbles.set(requestId, createThinkingBubble());
  inputEl.value = "";
});

function addLine(role, content) {
  logEl.querySelector(".empty-state")?.remove();

  const div = createMessageBubble(role);
  div.textContent = content;
  logEl.scrollTop = logEl.scrollHeight;
}

function createMessageBubble(role) {
  logEl.querySelector(".empty-state")?.remove();

  const div = document.createElement("div");
  const safeRole = ["user", "assistant"].includes(role) ? role : "system";
  div.className = `msg ${safeRole}`;
  logEl.appendChild(div);
  return div;
}

function createThinkingBubble() {
  const bubble = createMessageBubble("assistant");
  bubble.classList.add("thinking");
  bubble.setAttribute("role", "status");

  const spinner = document.createElement("span");
  spinner.className = "thinking-spinner";
  spinner.setAttribute("aria-hidden", "true");

  const label = document.createElement("span");
  label.textContent = "Thinking...";

  bubble.append(spinner, label);
  logEl.scrollTop = logEl.scrollHeight;
  return bubble;
}

function removePendingBubbles() {
  for (const bubble of pendingBubbles.values()) {
    bubble.remove();
  }
  pendingBubbles.clear();
}

function createEmptyState() {
  const emptyState = document.createElement("div");
  emptyState.className = "empty-state";
  const title = document.createElement("p");
  title.className = "empty-state-title";
  title.textContent = "Try one of these messages:";

  const hints = document.createElement("div");
  hints.className = "prompt-hints";
  for (const [prompt, operation] of promptHints) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "hint-button";

    const promptText = document.createElement("span");
    promptText.textContent = prompt;
    const operationText = document.createElement("span");
    operationText.className = "hint-operation";
    operationText.textContent = operation;

    button.append(promptText, operationText);
    button.addEventListener("click", () => {
      inputEl.value = prompt;
      inputEl.focus();
    });
    hints.appendChild(button);
  }
  emptyState.append(title, hints);
  return emptyState;
}

function setConnection(label, state) {
  connectionEl.textContent = label;
  connectionEl.className = `connection ${state}`;
}

checkSession();
