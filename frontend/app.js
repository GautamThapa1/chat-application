// frontend/app.js

const WS_URL = "ws://localhost:8000/ws";

const logEl = document.getElementById("log");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const connectionEl = document.getElementById("connection");
const clearButtonEl = document.getElementById("clear");

let streamingBubble = null;

const ws = new WebSocket(WS_URL);

ws.onopen = () => {
  setConnection("Connected", "connected");
  addLine("system", "connected");
};

ws.onclose = () => {
  setConnection("Disconnected", "error");
  addLine("system", "disconnected");
};

ws.onerror = (err) => {
  setConnection("Connection error", "error");
  addLine("system", `error: ${err.message || "websocket error"}`);
};

clearButtonEl.addEventListener("click", () => {
  logEl.replaceChildren(createEmptyState());
  streamingBubble = null;
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
    streamingBubble = createMessageBubble("assistant");
  } else if (msg.type === "chunk") {
    if (!streamingBubble) {
      streamingBubble = createMessageBubble("assistant");
    }
    streamingBubble.textContent += msg.content;
    logEl.scrollTop = logEl.scrollHeight;
  } else if (msg.type === "complete") {
    streamingBubble = null;
  } else if (msg.error) {
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

  ws.send(text);
  addLine("user", text);
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