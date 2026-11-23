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
const stopButtonEl = document.getElementById("stop");
const conversationsEl = document.getElementById("conversations");
const loadMoreConversationsEl = document.getElementById(
  "load-more-conversations",
);

const chatEl = document.getElementById("chat");
const authEl = document.getElementById("auth");
const userNameEl = document.getElementById("user-name");
const authFormEl = document.getElementById("auth-form");
const authTitleEl = document.getElementById("auth-title");
const authUsernameEl = document.getElementById("auth-username");
const authPasswordEl = document.getElementById("auth-password");
const authErrorEl = document.getElementById("auth-error");
const authSubmitEl = document.getElementById("auth-submit");
const authSwitchEl = document.getElementById("auth-switch");
const authSwitchTextEl = document.getElementById("auth-switch-text");
const HEARTBEAT_INTERVAL_MS = 25000; // how often we ask "are you alive?"
const PONG_TIMEOUT_MS = 10000; // how long we wait for the answer
const CONVERSATION_PAGE_SIZE = 20;
let reconnectAttempt = 0; // consecutive failed reconnects; 0 when connected

// Exponential backoff with jitter: ~1s, 2s, 4s ... capped at 30s. The random
// half stops every client retrying at the same instant after a server restart.
function reconnectDelayMs(attempt) {
  const ceiling = Math.min(30000, 1000 * 2 ** attempt);
  return ceiling / 2 + Math.random() * (ceiling / 2);
}

function scheduleReconnect(deadSocket, conversationId, initialPrompt = "") {
  const delay = reconnectDelayMs(reconnectAttempt++);
  setConnection(`Reconnecting in ${Math.ceil(delay / 1000)}s...`, "error");
  setTimeout(() => {
    // Skip if the user moved on (switched chat, logged out) in the meantime.
    if (ws !== deadSocket || activeConversationId !== conversationId) return;
    connectWebSocket(conversationId, initialPrompt);
  }, delay);
}

const pendingBubbles = new Map();

function syncBusyState() {
  stopButtonEl.hidden = pendingBubbles.size === 0;
  stopButtonEl.disabled = false;
}

let nextRequestId = 0;
let ws = null;
let authMode = "login";
let conversations = [];
let conversationOffset = 0;
let conversationTotalCount = 0;
let conversationsLoading = false;
let activeConversationId = null;
let conversationLoadGeneration = 0;
let userStateGeneration = 0;
let conversationCreationInProgress = false;

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
  reconnectAttempt = 0;
  socket?.close();
  conversationLoadGeneration += 1;
  conversations = [];
  conversationOffset = 0;
  conversationTotalCount = 0;
  conversationsLoading = false;
  activeConversationId = null;
  setCurrentUserName("");
  renderConversations();
  resetChat();
}

function setCurrentUserName(username) {
  if (userNameEl) {
    userNameEl.textContent = username || "Guest";
  }
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

function resetChat(showHints = true) {
  logEl.replaceChildren();
  if (showHints) logEl.appendChild(createEmptyState());
  pendingBubbles.clear();
  syncBusyState();
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
    setCurrentUserName(username);
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
      setCurrentUserName(me.data.username);
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
  const generation = userStateGeneration;
  try {
    const result = await api(
      `/api/conversations?limit=${CONVERSATION_PAGE_SIZE}&offset=0`,
    );
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

    appendConversations(result.data.conversations);
    conversationOffset =
      result.data.pagination.offset + result.data.conversations.length;
    conversationTotalCount = result.data.pagination.total_count;
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

async function loadMoreConversations() {
  if (
    conversationsLoading
    || conversationOffset >= conversationTotalCount
  ) return;

  const generation = userStateGeneration;
  const offset = conversationOffset;
  conversationsLoading = true;
  updateLoadMoreConversationsButton();

  try {
    const result = await api(
      `/api/conversations?limit=${CONVERSATION_PAGE_SIZE}&offset=${offset}`,
    );
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

    appendConversations(result.data.conversations);
    conversationOffset =
      result.data.pagination.offset + result.data.conversations.length;
    conversationTotalCount = result.data.pagination.total_count;
    renderConversations();
  } catch {
    if (generation === userStateGeneration) {
      addLine("system", "Could not reach the server to load chats.");
      setConnection("Could not load chats", "error");
    }
  } finally {
    if (generation === userStateGeneration) {
      conversationsLoading = false;
      updateLoadMoreConversationsButton();
    }
  }
}

function appendConversations(newConversations) {
  const conversationIds = new Set(
    conversations.map((conversation) => conversation.conversation_id),
  );
  for (const conversation of newConversations) {
    if (conversationIds.has(conversation.conversation_id)) continue;
    conversations.push(conversation);
    conversationIds.add(conversation.conversation_id);
  }
}

function updateLoadMoreConversationsButton() {
  loadMoreConversationsEl.hidden = conversationOffset >= conversationTotalCount;
  loadMoreConversationsEl.disabled = conversationsLoading;
}

function renderConversations() {
  updateLoadMoreConversationsButton();
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

loadMoreConversationsEl.addEventListener("click", () => {
  void loadMoreConversations();
});

async function createConversation(initialPrompt = "") {
  if (conversationCreationInProgress) return;

  const generation = userStateGeneration;
  conversationCreationInProgress = true;
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
    const newConversation = result.data.conversation;
    conversations = [
      newConversation,
      ...conversations.filter(
        (item) => item.conversation_id !== newConversation.conversation_id,
      ),
    ];
    conversationOffset += 1;
    conversationTotalCount += 1;
    renderConversations();
    await selectConversation(newConversation.conversation_id, initialPrompt);
  } catch {
    if (generation === userStateGeneration) {
      addLine("system", "Could not reach the server to create a chat.");
    }
  } finally {
    conversationCreationInProgress = false;
    newChatButtonEl.disabled = false;
  }
}

async function selectConversation(conversationId, initialPrompt = "") {
  const userGeneration = userStateGeneration;
  const socket = ws;
  ws = null;
  reconnectAttempt = 0;
  socket?.close();
  activeConversationId = conversationId;
  const generation = ++conversationLoadGeneration;
  resetChat(false);
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
        addLine(
          message.role,
          message.content,
          message.message_id,
        );
      }
    } else if (!initialPrompt) {
      resetChat();
    }
    connectWebSocket(conversationId, initialPrompt);
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

    const wasLoaded = conversations.some(
      (item) => item.conversation_id === conversationId,
    );
    conversations = conversations.filter(
      (item) => item.conversation_id !== conversationId,
    );
    if (wasLoaded) {
      if (conversationOffset > 0) conversationOffset -= 1;
      conversationTotalCount = Math.max(0, conversationTotalCount - 1);
    }
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

function connectWebSocket(conversationId, initialPrompt = "") {
  if (activeConversationId !== conversationId) return;
  const socket = new WebSocket(WS_URL);
  let promptToSend = initialPrompt;
  ws = socket;
  setConnection("Connecting", "");

  let pingTimer = null;
  let pongTimer = null;

  function stopHeartbeat() {
    clearInterval(pingTimer);
    clearTimeout(pongTimer);
  }

  function startHeartbeat() {
    pingTimer = setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({ type: "ping" }));
      pongTimer = setTimeout(() => {
        socket.close();
        handleClose(); // don't wait: a dead socket may never fire onclose
      }, PONG_TIMEOUT_MS);
    }, HEARTBEAT_INTERVAL_MS);
  }

  socket.onopen = () => {
    if (socket !== ws || activeConversationId !== conversationId) return;
    reconnectAttempt = 0;
    setConnection("Connected", "connected");
    startHeartbeat();
    addLine("system", "connected");
    if (
      promptToSend
      && sendMessage(promptToSend, inputEl.value.trim() === promptToSend)
    ) {
      promptToSend = "";
    }
  };

  let handled = false; // handleClose may be reached twice; run it once

  async function handleClose() {
    if (handled) return;
    handled = true;
    stopHeartbeat();
    // Intentional closes set ws = null first, so they stop here.
    if (socket !== ws || activeConversationId !== conversationId) return;

    if (pendingBubbles.size > 0) {
      removePendingBubbles();
      addLine(
        "system",
        "Connection lost before the reply finished. Please resend your message.",
      );
    }

    try {
      const me = await api("/api/me");
      if (me.status === 401) {
        showAuth("Your session has ended. Please log in again.");
        return;
      }
    } catch {
      // Server unreachable: fall through and keep retrying.
    }
    if (socket !== ws || activeConversationId !== conversationId) return;
    scheduleReconnect(socket, conversationId, promptToSend);
  }

  socket.onclose = handleClose;

  // The browser always fires "close" right after "error", so handleClose
  // covers the UI; here we only make sure the timers stop.
  socket.onerror = stopHeartbeat;

  socket.onmessage = (event) => {
    if (socket !== ws || activeConversationId !== conversationId) return;
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      addLine("system", `raw: ${event.data}`);
      return;
    }

    if (msg.type === "pong") {
      clearTimeout(pongTimer); // the server answered in time
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
      syncBusyState();
      refreshConversationTitle(conversationId);
    } else if (msg.type === "cancelled") {
      const bubble = pendingBubbles.get(msg.request_id);
      if (bubble && (bubble.classList.contains("thinking") || !bubble.textContent)) {
        bubble.remove();
      }
      pendingBubbles.delete(msg.request_id);
      syncBusyState();
      addLine("system", "Stopped.");
    } else if (msg.type === "error" || msg.error) {
      const bubble = pendingBubbles.get(msg.request_id);
      bubble?.remove();
      pendingBubbles.delete(msg.request_id);
      syncBusyState();
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
  sendMessage(inputEl.value, true);
});

function sendMessage(text, clearInput = false) {
  text = text.trim();
  if (!text) return false;

  if (!activeConversationId) {
    addLine("system", "Create or select a chat before sending a message.");
    return false;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addLine("system", "Not connected — try again in a moment.");
    return false;
  }
  if (pendingBubbles.size > 0) {
    addLine("system", "Wait for the reply to finish, or press Stop.");
    return false;
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
  syncBusyState();
  if (clearInput) inputEl.value = "";
  return true;
}

stopButtonEl.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  for (const requestId of pendingBubbles.keys()) {
    ws.send(JSON.stringify({ type: "cancel", request_id: requestId }));
  }
  stopButtonEl.disabled = true;
});

function addLine(role, content, messageId = null) {
  logEl.querySelector(".empty-state")?.remove();

  const div = createMessageBubble(role);
  div.textContent = content;
  if (messageId !== null) {
    div.dataset.messageId = String(messageId);
  }
  if (role === "user" && messageId !== null) {
    addMessageEditButton(div, messageId);
  } else if (role === "assistant" && messageId !== null) {
    addMessageRegenerateButton(div, messageId);
  }
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


function addMessageEditButton(bubble, messageId) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "message-action";
  button.textContent = "Edit";
  button.addEventListener("click", () => {
    if (pendingBubbles.size > 0) return;
    const original = bubble.childNodes[0]?.textContent || "";
    const input = window.prompt("Edit message", original);
    if (input === null || !input.trim() || input.trim() === original) return;
    sendEditMessage(messageId, input.trim());
  });
  bubble.append(document.createElement("br"), button);
}


function addMessageRegenerateButton(bubble, messageId) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "message-action";
  button.textContent = "Regenerate";
  button.addEventListener("click", () => {
    if (pendingBubbles.size > 0) return;
    sendRegenerateMessage(messageId, bubble);
  });
  bubble.append(document.createElement("br"), button);
}


function sendEditMessage(messageId, text) {
  if (!activeConversationId || !ws || ws.readyState !== WebSocket.OPEN) return;
  const requestId = String(++nextRequestId);
  ws.send(JSON.stringify({
    type: "message",
    mode: "edit",
    message_id: messageId,
    request_id: requestId,
    conversation_id: activeConversationId,
    content: text,
  }));
  // Remove the old branch from the UI immediately. The server has already
  // validated ownership and atomically truncates later database rows.
  const bubbles = Array.from(logEl.querySelectorAll(".msg"));
  const bubbleIndex = bubbles.indexOf(
    document.querySelector(`[data-message-id="${messageId}"]`)
  );
  if (bubbleIndex >= 0) {
    const editedBubble = bubbles[bubbleIndex];
    editedBubble.textContent = text;
    delete editedBubble.dataset.messageId;
    bubbles.slice(bubbleIndex + 1).forEach((oldBubble) => oldBubble.remove());
  }
  pendingBubbles.set(requestId, createThinkingBubble());
  syncBusyState();
}


function sendRegenerateMessage(assistantMessageId, existingBubble) {
  if (!activeConversationId || !ws || ws.readyState !== WebSocket.OPEN) return;
  const requestId = String(++nextRequestId);
  ws.send(JSON.stringify({
    type: "message",
    mode: "regenerate",
    assistant_message_id: assistantMessageId,
    request_id: requestId,
    conversation_id: activeConversationId,
    content: "",
  }));
  existingBubble.classList.add("thinking");
  existingBubble.replaceChildren();
  pendingBubbles.set(requestId, existingBubble);
  syncBusyState();
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
  syncBusyState();
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
      if (!activeConversationId) {
        if (conversationCreationInProgress) return;
        inputEl.value = prompt;
        void createConversation(prompt);
        return;
      }
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
