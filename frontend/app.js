// frontend/app.js

const WS_URL = "ws://localhost:8000/ws";

const logEl = document.getElementById("log");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const connectionEl = document.getElementById("connection");
const clearButtonEl = document.getElementById("clear");

const pendingBubbles = new Map();
let nextRequestId = 0;

const ws = new WebSocket(WS_URL);

ws.onopen = () => {
  setConnection("Connected", "connected");
  addLine("system", "connected");
};

ws.onclose = () => {
  setConnection("Disconnected", "error");
  removePendingBubbles();
  addLine("system", "disconnected");
};

ws.onerror = (err) => {
  setConnection("Connection error", "error");
  removePendingBubbles();
  addLine("system", `error: ${err.message || "websocket error"}`);
};

clearButtonEl.addEventListener("click", () => {
  logEl.replaceChildren(createEmptyState());
  pendingBubbles.clear();
});

ws.onmessage = (event) => {
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

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  if (ws.readyState !== WebSocket.OPEN) {
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