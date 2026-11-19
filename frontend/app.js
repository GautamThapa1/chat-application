// Same host as the page, so the browser sends the session cookie. Becomes
// wss:// automatically once the site is served over HTTPS.
const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

const logEl = document.getElementById("log");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const connectionEl = document.getElementById("connection");
const clearButtonEl = document.getElementById("clear");
const logoutButtonEl = document.getElementById("logout");

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

function showAuth(message = "") {
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
    setAuthMode("login");
    showChat();
    connectWebSocket();
  } catch {
    authErrorEl.textContent = "Could not reach the server.";
  } finally {
    authSubmitEl.disabled = false;
  }
});

logoutButtonEl.addEventListener("click", async () => {
  const socket = ws;
  ws = null;
  socket?.close();

  let message = "";
  try {
    const result = await api("/api/logout", "POST");
    if (!result.ok) {
      message = `Could not end the server session: ${errorMessage(result.data)}`;
    }
  } catch {
    message = "Could not reach the server to end your session.";
  }
  resetChat();
  setConnection("Disconnected", "error");
  showAuth(message);
});

async function checkSession() {
  try {
    const me = await api("/api/me");
    if (me.ok) {
      showChat();
      connectWebSocket();
    } else if (me.status === 401) {
      showAuth();
    } else {
      showAuth(errorMessage(me.data));
    }
  } catch {
    showAuth("Could not reach the server.");
  }
}

function connectWebSocket() {
  if (ws) {
    const old = ws;
    ws = null;
    old.close();
  }

  const socket = new WebSocket(WS_URL);
  ws = socket;
  setConnection("Connecting", "");

  socket.onopen = () => {
    setConnection("Connected", "connected");
    addLine("system", "connected");
  };

  socket.onclose = async () => {
    if (socket !== ws) return;
    setConnection("Disconnected", "error");
    removePendingBubbles();
    addLine("system", "disconnected");

    try {
      const me = await api("/api/me");
      if (me.status === 401) {
        ws = null;
        resetChat();
        showAuth("Your session has ended. Please log in again.");
      }
    } catch {
      // The chat remains visible with its disconnected status if the server is unreachable.
    }
  };

  socket.onerror = (err) => {
    if (socket !== ws) return;
    setConnection("Connection error", "error");
    removePendingBubbles();
    addLine("system", `error: ${err.message || "websocket error"}`);
  };

  socket.onmessage = (event) => {
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

clearButtonEl.addEventListener("click", () => {
  logEl.replaceChildren(createEmptyState());
  pendingBubbles.clear();
});

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addLine("system", "not connected — try again in a moment");
    return;
  }

  const requestId = String(++nextRequestId);
  ws.send(JSON.stringify({ type: "message", request_id: requestId, content: text }));
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
  const emptyState = document.createElement("p");
  emptyState.className = "empty-state";
  emptyState.textContent = "Send a message to start the conversation.";
  return emptyState;
}

function setConnection(label, state) {
  connectionEl.textContent = label;
  connectionEl.className = `connection ${state}`;
}

checkSession();
