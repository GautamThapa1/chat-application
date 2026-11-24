// ===========================================================================
// How this file is organised
//
//  * ONE WebSocket for the whole signed-in session. Switching conversations
//    never touches it, so it can never cancel a running reply.
//  * `chats` holds one state object per conversation id (its messages and its
//    own generation status). The screen is just a view of the active one.
//  * Every server event carries conversation_id + request_id, so events are
//    routed to the right conversation and stale ones are ignored.
//  * The database is the source of truth: when a reply finishes (or when you
//    come back to a chat) messages are re-loaded from the server.
// ===========================================================================

// Same host as the page, so the browser sends the session cookie. Becomes
// wss:// automatically once the site is served over HTTPS.
const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

const HEARTBEAT_INTERVAL_MS = 25000; // how often we ask "are you alive?"
const PONG_TIMEOUT_MS = 10000; // how long we wait for the answer
const CONVERSATION_PAGE_SIZE = 20;
const DEFAULT_TITLE = "New chat";

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const appEl = $("chat");
const authEl = $("auth");
const logEl = $("log");
const threadEl = $("thread");
const formEl = $("form");
const inputEl = $("input");
const sendButtonEl = $("send");
const connectionEl = $("connection");
const logoutButtonEl = $("logout");
const newChatButtonEl = $("new-chat");
const conversationsEl = $("conversations");
const loadMoreConversationsEl = $("load-more-conversations");
const chatTitleEl = $("chat-title");
const menuButtonEl = $("menu");
const scrimEl = $("scrim");
const userNameEl = $("user-name");
const userAvatarEl = $("user-avatar");
const authFormEl = $("auth-form");
const authTitleEl = $("auth-title");
const authUsernameEl = $("auth-username");
const authPasswordEl = $("auth-password");
const authErrorEl = $("auth-error");
const authSubmitEl = $("auth-submit");
const authSwitchEl = $("auth-switch");
const authSwitchTextEl = $("auth-switch-text");

// ---------- State ----------
let ws = null; // the one socket (null while disconnected)
let reconnectAttempt = 0; // consecutive failed reconnects; 0 when connected
let reconnectTimer = null;
let pingTimer = null;
let pongTimer = null;

let signedIn = false;
let username = "";
let authMode = "login";
let userStateGeneration = 0; // bumped on login/logout so late responses are dropped

let conversations = []; // sidebar list (newest first)
let conversationOffset = 0;
let conversationTotalCount = 0;
let conversationsLoading = false;
let activeConversationId = null; // null = the blank "new chat" screen
let stickToBottom = true; // auto-scroll only while the user is at the bottom
let streamBubble = null; // the element the live reply is written into
let creatingConversation = false;

/**
 * Per-conversation state. Nothing here is shared between conversations.
 *   messages   : saved messages [{message_id, role, content}]
 *   generation : null, or the reply being produced right now:
 *                {requestId, mode, prompt, assistantMessageId, text, finishing}
 *   loaded     : true once messages were fetched at least once
 *   notice     : a small status/error line shown at the end of the thread
 *   draft      : unsent text in the composer for this conversation
 */
const chats = new Map();

function chatState(conversationId) {
  let chat = chats.get(conversationId);
  if (!chat) {
    chat = {
      messages: [],
      generation: null,
      toolResults: [],
      loaded: false,
      notice: "",
      noticeIsError: false,
      draft: "",
      editingId: null,
      editDraft: "",
      loadSeq: 0,
    };
    chats.set(conversationId, chat);
  }
  return chat;
}

function activeChat() {
  return activeConversationId === null ? null : chatState(activeConversationId);
}

const promptHints = [
  ["Remember that my backend uses FastAPI", "Save note"],
  ["Show my notes", "Retrieve notes"],
  ["Create a task to finish gRPC", "Create task"],
  ["Show my tasks", "List tasks"],
  ["Mark task 1 completed", "Complete task"],
  ["Hello", "General chat"],
];

// ---------- Small helpers ----------
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Static, trusted SVG paths (never built from user text).
const ICONS = {
  copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
  regenerate: '<path d="M21 12a9 9 0 0 0-15.5-6.3L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 15.5 6.3L21 16"/><path d="M16 16h5v5"/>',
  trash: '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
};

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name]; // constant strings above only
  return svg;
}

// Request ids must be unique forever: the server ignores a second save that
// reuses an id, so a counter that restarts at 1 after a page reload would make
// new answers silently disappear.
function newRequestId() {
  if (window.crypto?.randomUUID) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function shortTitle(text) {
  const firstLine = text.trim().split(/\r?\n/)[0].replace(/\s+/g, " ");
  return firstLine.length > 48 ? `${firstLine.slice(0, 47).trimEnd()}…` : firstLine;
}

// SQLite stores "YYYY-MM-DD HH:MM:SS" in UTC without a zone marker.
function parseTime(value) {
  if (!value) return 0;
  const iso = value.includes("T") ? value : value.replace(" ", "T");
  const time = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  return Number.isNaN(time) ? 0 : time;
}

function groupLabel(value) {
  const day = 24 * 60 * 60 * 1000;
  const startOfToday = new Date().setHours(0, 0, 0, 0);
  const time = parseTime(value);
  if (time >= startOfToday) return "Today";
  if (time >= startOfToday - day) return "Yesterday";
  if (time >= startOfToday - 7 * day) return "Previous 7 days";
  return "Older";
}

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

function setConnection(label, state) {
  connectionEl.textContent = label;
  connectionEl.className = `connection ${state}`;
}

function setNotice(chat, text, isError = false) {
  chat.notice = text;
  chat.noticeIsError = isError;
}

// ---------- Sign in / out ----------
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

function setCurrentUser(name) {
  username = name || "";
  userNameEl.textContent = username || "Guest";
  userAvatarEl.textContent = (username[0] || "?").toUpperCase();
}

function clearUserState() {
  userStateGeneration += 1;
  closeSocket();
  chats.clear();
  conversations = [];
  conversationOffset = 0;
  conversationTotalCount = 0;
  conversationsLoading = false;
  activeConversationId = null;
  streamBubble = null;
  signedIn = false;
  setCurrentUser("");
  inputEl.value = "";
  renderAll();
}

function showAuth(message = "") {
  clearUserState();
  appEl.hidden = true;
  authEl.hidden = false;
  authErrorEl.textContent = message;
  authUsernameEl.focus();
}

function showChat() {
  authEl.hidden = true;
  appEl.hidden = false;
  signedIn = true;
  inputEl.focus();
}

authSwitchEl.addEventListener("click", () => {
  setAuthMode(authMode === "login" ? "register" : "login");
});

authFormEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = authUsernameEl.value.trim();
  const password = authPasswordEl.value;
  if (!name || !password) {
    authErrorEl.textContent = "Enter a username and password.";
    return;
  }

  authSubmitEl.disabled = true;
  authErrorEl.textContent = "";
  try {
    if (authMode === "register") {
      const registered = await api("/api/register", "POST", { username: name, password });
      if (!registered.ok) {
        authErrorEl.textContent = errorMessage(registered.data);
        return;
      }
    }

    const login = await api("/api/login", "POST", { username: name, password });
    if (!login.ok) {
      authErrorEl.textContent = errorMessage(login.data);
      return;
    }

    authPasswordEl.value = "";
    clearUserState();
    setAuthMode("login");
    showChat();
    setCurrentUser(name);
    connectWebSocket();
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
      setCurrentUser(me.data.username);
      showChat();
      connectWebSocket();
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

// ---------- WebSocket (one per session) ----------
// Exponential backoff with jitter: ~1s, 2s, 4s ... capped at 30s. The random
// half stops every client retrying at the same instant after a server restart.
function reconnectDelayMs(attempt) {
  const ceiling = Math.min(30000, 1000 * 2 ** attempt);
  return ceiling / 2 + Math.random() * (ceiling / 2);
}

function stopHeartbeat() {
  clearInterval(pingTimer);
  clearTimeout(pongTimer);
}

function closeSocket() {
  clearTimeout(reconnectTimer);
  stopHeartbeat();
  const socket = ws;
  ws = null; // set first: the onclose handler ignores sockets that are not current
  reconnectAttempt = 0;
  socket?.close();
}

function socketIsOpen() {
  return ws !== null && ws.readyState === WebSocket.OPEN;
}

function wsSend(payload) {
  if (!socketIsOpen()) return false;
  ws.send(JSON.stringify(payload));
  return true;
}

function connectWebSocket() {
  if (!signedIn || ws) return;
  const generation = userStateGeneration;
  const socket = new WebSocket(WS_URL);
  ws = socket;
  setConnection("Connecting…", "");

  socket.onopen = () => {
    if (socket !== ws) return;
    reconnectAttempt = 0;
    setConnection("Connected", "connected");
    startHeartbeat(socket);
    // Ask which replies are still running on the server. After a reconnect
    // this is how we learn about replies that kept going while we were away.
    wsSend({ type: "sync" });
  };

  let handledClose = false; // may be reached twice (heartbeat + onclose)
  async function handleClose() {
    if (handledClose) return;
    handledClose = true;
    stopHeartbeat();
    if (socket !== ws) return; // an intentional close
    ws = null;
    setConnection("Reconnecting…", "error");
    // Running replies are NOT lost: the server keeps generating and saving.
    // We only lose the live view until we reconnect and sync.
    try {
      const me = await api("/api/me");
      if (me.status === 401) {
        showAuth("Your session has ended. Please log in again.");
        return;
      }
    } catch {
      // Server unreachable: keep retrying.
    }
    if (generation !== userStateGeneration) return;
    const delay = reconnectDelayMs(reconnectAttempt++);
    setConnection(`Reconnecting in ${Math.ceil(delay / 1000)}s…`, "error");
    reconnectTimer = setTimeout(connectWebSocket, delay);
  }

  socket.onclose = handleClose;
  socket.onerror = stopHeartbeat; // "close" always follows "error"
  socket.onmessage = (event) => {
    if (socket !== ws) return;
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    handleServerEvent(msg);
  };

  function startHeartbeat(current) {
    stopHeartbeat();
    pingTimer = setInterval(() => {
      if (current.readyState !== WebSocket.OPEN) return;
      current.send(JSON.stringify({ type: "ping" }));
      pongTimer = setTimeout(() => {
        current.close();
        handleClose(); // a dead socket may never fire onclose
      }, PONG_TIMEOUT_MS);
    }, HEARTBEAT_INTERVAL_MS);
  }
}

// ---------- Server events -> per-conversation state ----------
function handleServerEvent(msg) {
  if (msg.type === "pong") {
    clearTimeout(pongTimer); // the server answered in time
    return;
  }
  if (msg.type === "active") {
    reconcileActive(msg.generations || []);
    return;
  }

  const conversationId = msg.conversation_id;
  if (conversationId === undefined || conversationId === null) {
    // An error that belongs to no conversation (e.g. invalid payload).
    if (msg.type === "error" && activeChat()) {
      setNotice(activeChat(), msg.error, true);
      renderThread();
    }
    return;
  }

  const chat = chatState(conversationId);
  const gen = chat.generation;
  // Stale-event guard: only the reply we are tracking may change this chat.
  const isCurrent = gen !== null && gen.requestId === msg.request_id;

  switch (msg.type) {
    case "start":
      if (!isCurrent) wsSend({ type: "sync" }); // started elsewhere (another tab)
      break;

    case "chunk":
      if (!isCurrent) {
        wsSend({ type: "sync" });
        break;
      }
      // `offset` makes chunks idempotent: a chunk that is already part of a
      // synced snapshot just overwrites the same text instead of doubling it.
      gen.text = gen.text.slice(0, msg.offset ?? gen.text.length) + msg.content;
      if (conversationId === activeConversationId) paintStream(gen);
      break;

    case "tool_result":
      if (!isCurrent) {
        wsSend({ type: "sync" });
        break;
      }
      if (!Number.isInteger(msg.index) || msg.index < 0 || typeof msg.tool !== "string") {
        break;
      }
      chat.toolResults[msg.index] = { tool: msg.tool, result: msg.result };
      if (conversationId === activeConversationId) renderThread();
      break;

    case "complete":
    case "cancelled":
      if (gen !== null && !isCurrent) break; // an old reply's event: ignore
      if (msg.type === "cancelled") setNotice(chat, "Stopped.");
      finishGeneration(conversationId);
      break;

    case "error":
      if (gen !== null && !isCurrent) break;
      failGeneration(conversationId, msg.error);
      break;

    default:
      break;
  }
}

// The server told us what is running. Bring every conversation in line.
function reconcileActive(list) {
  const runningIds = new Set();
  for (const item of list) {
    runningIds.add(item.conversation_id);
    const chat = chatState(item.conversation_id);
    chat.toolResults = Array.isArray(item.tool_results) ? item.tool_results : [];
    chat.generation = {
      requestId: item.request_id,
      mode: item.mode,
      prompt: item.prompt,
      assistantMessageId: item.assistant_message_id,
      text: item.content || "",
      finishing: false,
    };
    if (!chat.loaded) void loadMessages(item.conversation_id);
  }
  // A reply we thought was running but the server no longer has: it finished
  // (and was saved) while we were disconnected, so just reload the result.
  for (const [conversationId, chat] of chats) {
    if (chat.generation && !runningIds.has(conversationId) && !chat.generation.finishing) {
      finishGeneration(conversationId);
    }
  }
  renderConversations();
  renderThread();
}

// A reply is over. Keep showing its text until the saved messages arrive, so
// the screen does not flicker; then replace it with the database version.
function finishGeneration(conversationId) {
  const chat = chatState(conversationId);
  const finished = chat.generation;
  if (finished) finished.finishing = true;
  void loadMessages(conversationId).then(() => {
    // Only clear the reply we were finishing, never a newer one.
    if (chat.generation === finished) chat.generation = null;
    renderConversations();
    if (conversationId === activeConversationId) renderThread();
  });
  renderConversations();
}

function failGeneration(conversationId, error) {
  const chat = chatState(conversationId);
  const gen = chat.generation;
  chat.generation = null;
  setNotice(chat, error || "Something went wrong.", true);
  // A normal message is only saved when the reply succeeds. Give the text
  // back so the user can resend it with one click.
  if (gen?.mode === "message" && conversationId === activeConversationId && !inputEl.value) {
    inputEl.value = gen.prompt;
    autosizeInput();
  } else if (gen?.mode === "message" && !chat.draft) {
    chat.draft = gen.prompt;
  }
  // Edit already changed the database before it failed: show the real state.
  void loadMessages(conversationId);
  renderConversations();
  if (conversationId === activeConversationId) renderThread();
}

// ---------- Loading data ----------
async function loadMessages(conversationId) {
  const chat = chatState(conversationId);
  const seq = ++chat.loadSeq; // a newer request makes older answers stale
  const generation = userStateGeneration;
  try {
    const result = await api(`/api/conversations/${conversationId}`);
    if (generation !== userStateGeneration) return;
    if (!result.ok) {
      if (result.status === 401) {
        showAuth("Your session has ended. Please log in again.");
      } else if (result.status === 404) {
        dropConversation(conversationId);
      } else if (seq === chat.loadSeq) {
        setNotice(chat, errorMessage(result.data), true);
      }
      return;
    }
    if (seq !== chat.loadSeq) return; // stale response
    chat.messages = result.data.messages;
    chat.loaded = true;
    updateConversationInList(result.data.conversation);
  } catch {
    if (generation === userStateGeneration && seq === chat.loadSeq) {
      setNotice(chat, "Could not reach the server to load this chat.", true);
    }
  }
  if (conversationId === activeConversationId && !chat.generation?.finishing) {
    renderThread();
  }
}

function updateConversationInList(conversation) {
  const known = conversations.some((c) => c.conversation_id === conversation.conversation_id);
  if (!known) return;
  conversations = conversations.map((item) =>
    item.conversation_id === conversation.conversation_id ? conversation : item,
  );
  sortConversations();
  renderConversations();
  if (conversation.conversation_id === activeConversationId) {
    chatTitleEl.textContent = conversation.title || DEFAULT_TITLE;
  }
}

function sortConversations() {
  conversations.sort((a, b) => parseTime(b.updated_at) - parseTime(a.updated_at));
}

async function loadConversations() {
  const generation = userStateGeneration;
  try {
    const result = await api(`/api/conversations?limit=${CONVERSATION_PAGE_SIZE}&offset=0`);
    if (generation !== userStateGeneration) return;
    if (!result.ok) {
      if (result.status === 401) showAuth("Your session has ended. Please log in again.");
      else setConnection("Could not load chats", "error");
      return;
    }
    appendConversations(result.data.conversations);
    conversationOffset = result.data.pagination.offset + result.data.conversations.length;
    conversationTotalCount = result.data.pagination.total_count;
    renderAll();
  } catch {
    if (generation === userStateGeneration) setConnection("Could not load chats", "error");
  }
}

async function loadMoreConversations() {
  if (conversationsLoading || conversationOffset >= conversationTotalCount) return;
  const generation = userStateGeneration;
  const offset = conversationOffset;
  conversationsLoading = true;
  renderConversations();
  try {
    const result = await api(`/api/conversations?limit=${CONVERSATION_PAGE_SIZE}&offset=${offset}`);
    if (generation !== userStateGeneration) return;
    if (!result.ok) {
      if (result.status === 401) showAuth("Your session has ended. Please log in again.");
      return;
    }
    appendConversations(result.data.conversations);
    conversationOffset = result.data.pagination.offset + result.data.conversations.length;
    conversationTotalCount = result.data.pagination.total_count;
  } catch {
    // Keep the list we have; the button stays available to retry.
  } finally {
    if (generation === userStateGeneration) {
      conversationsLoading = false;
      renderConversations();
    }
  }
}

function appendConversations(items) {
  const known = new Set(conversations.map((c) => c.conversation_id));
  for (const item of items) {
    if (!known.has(item.conversation_id)) conversations.push(item);
  }
}

// ---------- Conversations: select / create / delete ----------
function closeSidebar() {
  appEl.classList.remove("sidebar-open");
}

// Switching only changes which conversation is DISPLAYED. It does not touch the
// socket and does not cancel anything: the other chat keeps generating.
function selectConversation(conversationId) {
  closeSidebar();
  if (conversationId === activeConversationId) return;
  saveDraft();
  activeConversationId = conversationId;
  restoreDraft();
  stickToBottom = true;
  renderAll();
  // Refresh from the database (skip nothing: it is cheap, and a reply may have
  // finished while we were looking at another chat).
  void loadMessages(conversationId);
}

function saveDraft() {
  const chat = activeChat();
  if (chat) chat.draft = inputEl.value;
}

function restoreDraft() {
  const chat = activeChat();
  inputEl.value = chat ? chat.draft : "";
  autosizeInput();
}

// "New chat" is lazy: it shows a blank screen and the conversation is only
// created on the server when the first message is sent. Nothing is cleared.
function startNewChat() {
  closeSidebar();
  if (activeConversationId === null) {
    inputEl.focus();
    return;
  }
  saveDraft();
  activeConversationId = null;
  inputEl.value = "";
  autosizeInput();
  renderAll();
  inputEl.focus();
}

async function createConversation() {
  if (creatingConversation) return null;
  creatingConversation = true;
  const generation = userStateGeneration;
  try {
    const result = await api("/api/conversations", "POST", { title: DEFAULT_TITLE });
    if (generation !== userStateGeneration) return null;
    if (!result.ok) {
      if (result.status === 401) showAuth("Your session has ended. Please log in again.");
      else showBlankError(errorMessage(result.data));
      return null;
    }
    const created = result.data.conversation;
    conversations = [
      created,
      ...conversations.filter((c) => c.conversation_id !== created.conversation_id),
    ];
    conversationOffset += 1;
    conversationTotalCount += 1;
    const chat = chatState(created.conversation_id);
    chat.loaded = true; // brand new: nothing to fetch
    activeConversationId = created.conversation_id;
    return created.conversation_id;
  } catch {
    if (generation === userStateGeneration) showBlankError("Could not reach the server to create a chat.");
    return null;
  } finally {
    creatingConversation = false;
  }
}

function showBlankError(text) {
  const blank = el("div", "notice error", text);
  threadEl.appendChild(blank);
}

async function deleteConversation(conversationId) {
  if (!window.confirm("Delete this chat? This cannot be undone.")) return;
  const generation = userStateGeneration;
  // Stop a running reply first so the server does not try to save into a
  // conversation that no longer exists.
  const running = chats.get(conversationId)?.generation;
  if (running) wsSend({ type: "cancel", request_id: running.requestId, conversation_id: conversationId });
  try {
    const result = await api(`/api/conversations/${conversationId}`, "DELETE");
    if (generation !== userStateGeneration) return;
    if (!result.ok) {
      if (result.status === 401) showAuth("Your session has ended. Please log in again.");
      else setConnection(errorMessage(result.data), "error");
      return;
    }
    dropConversation(conversationId);
  } catch {
    if (generation === userStateGeneration) {
      setConnection("Could not reach the server to delete this chat.", "error");
    }
  }
}

// Remove a conversation from the screen and from memory.
function dropConversation(conversationId) {
  const wasListed = conversations.some((c) => c.conversation_id === conversationId);
  conversations = conversations.filter((c) => c.conversation_id !== conversationId);
  chats.delete(conversationId);
  if (wasListed) {
    if (conversationOffset > 0) conversationOffset -= 1;
    conversationTotalCount = Math.max(0, conversationTotalCount - 1);
  }
  if (activeConversationId === conversationId) {
    activeConversationId = null;
    inputEl.value = "";
    autosizeInput();
    // Fixed: the old code read `conversations[0].id`, which does not exist.
    if (conversations.length) {
      selectConversation(conversations[0].conversation_id);
      return;
    }
  }
  renderAll();
}

// ---------- Sending ----------
async function submitMessage(text) {
  text = text.trim();
  if (!text) return false;

  if (!socketIsOpen()) {
    notifyActive("Not connected — try again in a moment.", true);
    return false;
  }
  if (activeConversationId === null) {
    const id = await createConversation();
    if (id === null) return false;
    renderAll();
  }
  const chat = activeChat();
  if (chat.generation) {
    setNotice(chat, "Wait for the reply to finish, or press Stop.");
    renderThread();
    return false;
  }

  const conversationId = activeConversationId;
  const requestId = newRequestId();
  const sent = wsSend({
    type: "message",
    request_id: requestId,
    conversation_id: conversationId,
    content: text,
  });
  if (!sent) {
    notifyActive("Not connected — try again in a moment.", true);
    return false;
  }

  // Show a title right away; the server confirms it when the reply is saved.
  const listed = conversations.find((c) => c.conversation_id === conversationId);
  if (listed && (!listed.title || listed.title === DEFAULT_TITLE) && chat.messages.length === 0) {
    listed.title = shortTitle(text);
  }
  if (listed) listed.updated_at = new Date().toISOString();
  sortConversations();

  chat.generation = {
    requestId, mode: "message", prompt: text, assistantMessageId: null, text: "", finishing: false,
  };
  chat.toolResults = [];
  setNotice(chat, "");
  stickToBottom = true;
  renderAll();
  return true;
}

function notifyActive(text, isError) {
  const chat = activeChat();
  if (chat) {
    setNotice(chat, text, isError);
    renderThread();
  } else {
    showBlankError(text);
  }
}

// Edit: change a user message and re-answer from there (later messages go).
function sendEdit(chat, message, newText) {
  const conversationId = activeConversationId;
  const requestId = newRequestId();
  const sent = wsSend({
    type: "message",
    mode: "edit",
    message_id: message.message_id,
    request_id: requestId,
    conversation_id: conversationId,
    content: newText,
  });
  if (!sent) {
    setNotice(chat, "Not connected — try again in a moment.", true);
    renderThread();
    return;
  }
  // Mirror what the server does: keep messages up to this one, new text.
  const index = chat.messages.findIndex((m) => m.message_id === message.message_id);
  chat.messages = chat.messages.slice(0, index + 1);
  chat.messages[index] = { ...message, content: newText };
  chat.editingId = null;
  chat.generation = {
    requestId, mode: "edit", prompt: newText, assistantMessageId: null, text: "", finishing: false,
  };
  chat.toolResults = [];
  setNotice(chat, "");
  stickToBottom = true;
  renderAll();
}

// Regenerate: write a new answer for an existing assistant message.
function sendRegenerate(chat, message) {
  const conversationId = activeConversationId;
  const requestId = newRequestId();
  const sent = wsSend({
    type: "message",
    mode: "regenerate",
    assistant_message_id: message.message_id,
    request_id: requestId,
    conversation_id: conversationId,
    content: "",
  });
  if (!sent) {
    setNotice(chat, "Not connected — try again in a moment.", true);
    renderThread();
    return;
  }
  chat.generation = {
    requestId, mode: "regenerate", prompt: "", assistantMessageId: message.message_id, text: "", finishing: false,
  };
  chat.toolResults = [];
  setNotice(chat, "");
  stickToBottom = true;
  renderAll();
}

function stopActiveGeneration() {
  const chat = activeChat();
  if (!chat?.generation || chat.generation.finishing) return;
  wsSend({
    type: "cancel",
    request_id: chat.generation.requestId,
    conversation_id: activeConversationId,
  });
  sendButtonEl.disabled = true; // until the server confirms with "cancelled"
}

// ---------- Rendering ----------
function renderAll() {
  renderConversations();
  renderThread();
}

function renderConversations() {
  loadMoreConversationsEl.hidden = conversationOffset >= conversationTotalCount;
  loadMoreConversationsEl.disabled = conversationsLoading;
  conversationsEl.replaceChildren();

  if (!conversations.length) {
    conversationsEl.appendChild(el("p", "conversation-empty", "No chats yet"));
    return;
  }

  let lastGroup = "";
  for (const conversation of conversations) {
    const group = groupLabel(conversation.updated_at);
    if (group !== lastGroup) {
      conversationsEl.appendChild(el("div", "conversation-group", group));
      lastGroup = group;
    }

    const id = conversation.conversation_id;
    const title = conversation.title || DEFAULT_TITLE;
    const item = el("div", "conversation-item");
    if (id === activeConversationId) item.classList.add("active");

    const select = el("button", "conversation-select");
    select.type = "button";
    select.title = title;
    if (id === activeConversationId) select.setAttribute("aria-current", "page");
    select.appendChild(el("span", "conversation-title", title));
    // A spinner marks chats that are still generating in the background.
    if (chats.get(id)?.generation) {
      const spinner = el("span", "spinner");
      spinner.setAttribute("role", "status");
      spinner.setAttribute("aria-label", "Generating a reply");
      select.appendChild(spinner);
    }
    select.addEventListener("click", () => selectConversation(id));

    const remove = el("button", "conversation-delete");
    remove.type = "button";
    remove.title = "Delete chat";
    remove.setAttribute("aria-label", `Delete ${title}`);
    remove.appendChild(icon("trash"));
    remove.addEventListener("click", () => void deleteConversation(id));

    item.append(select, remove);
    conversationsEl.appendChild(item);
  }
}

function renderThread() {
  streamBubble = null;
  threadEl.replaceChildren();
  const chat = activeChat();

  const listed = conversations.find((c) => c.conversation_id === activeConversationId);
  chatTitleEl.textContent = (listed && listed.title) || DEFAULT_TITLE;
  document.title = listed?.title ? `${listed.title} · Luma` : "Luma Chat";

  updateComposer();

  if (!chat) {
    threadEl.appendChild(createEmptyState());
    return;
  }
  if (!chat.loaded) {
    threadEl.appendChild(el("div", "notice", "Loading chat…"));
    return;
  }

  const gen = chat.generation;
  let visible = chat.messages;
  // While regenerating, the old answer (and everything after it) is hidden:
  // the new answer streams into its place. If it fails, it comes back.
  if (gen?.mode === "regenerate") {
    const index = visible.findIndex((m) => m.message_id === gen.assistantMessageId);
    if (index >= 0) visible = visible.slice(0, index);
  }

  if (!visible.length && !gen) {
    threadEl.appendChild(createEmptyState());
  }

  const busy = Boolean(gen);
  visible.forEach((message, index) => {
    threadEl.appendChild(renderMessage(chat, message, {
      busy,
      isLast: index === visible.length - 1,
    }));
  });

  if (gen) {
    // Normal messages are saved only when the reply ends, so the question is
    // drawn from the generation itself (it also survives a page refresh).
    if (gen.mode === "message") {
      threadEl.appendChild(renderMessage(
        chat, { message_id: null, role: "user", content: gen.prompt }, { busy: true, isLast: false },
      ));
    }
  }
  if (chat.toolResults.some(isVisibleMemoryResult)) {
    threadEl.appendChild(renderToolResults(chat));
  }
  if (gen) threadEl.appendChild(renderStreamingMessage(gen));

  if (chat.notice) {
    threadEl.appendChild(el("div", chat.noticeIsError ? "notice error" : "notice", chat.notice));
  }
  scrollToBottomIfNeeded();
}

function actionButton(label, iconName, onClick, disabled, title) {
  const button = el("button", "action");
  button.type = "button";
  button.title = title || label;
  button.disabled = disabled;
  button.append(icon(iconName), el("span", "label", label));
  button.addEventListener("click", onClick);
  return button;
}

function copyButton(message) {
  const button = actionButton("Copy", "copy", async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      button.replaceChildren(icon("check"), el("span", "label", "Copied"));
      setTimeout(() => button.replaceChildren(icon("copy"), el("span", "label", "Copy")), 1500);
    } catch {
      button.title = "Copy is not available in this browser context";
    }
  }, false);
  return button;
}

function renderMessage(chat, message, { busy, isLast }) {
  const isUser = message.role === "user";
  const row = el("div", `message ${isUser ? "user" : "assistant"}`);
  if (message.message_id !== null) row.dataset.messageId = String(message.message_id);
  if (!isUser) row.appendChild(el("div", "assistant-avatar", "✳"));

  const body = el("div", "message-body");

  if (isUser && chat.editingId !== null && chat.editingId === message.message_id) {
    body.appendChild(renderEditBox(chat, message));
    row.appendChild(body);
    return row;
  }

  body.appendChild(el("div", "bubble", message.content));

  // The buttons are part of every message from the first paint. They are only
  // disabled (never hidden or delayed) while a reply is being produced, or for
  // the instant before a just-saved message has its database id.
  const actions = el("div", "actions");
  const hasId = message.message_id !== null;
  const waiting = busy ? "Wait for the reply to finish" : "";
  if (isUser) {
    actions.append(
      actionButton("Edit", "edit", () => {
        chat.editingId = message.message_id;
        chat.editDraft = message.content;
        renderThread();
        threadEl.querySelector(".edit-box textarea")?.focus();
      }, busy || !hasId, waiting || "Edit message"),
      copyButton(message),
    );
  } else {
    actions.append(
      copyButton(message),
      actionButton("Regenerate", "regenerate", () => {
        if (!isLast && !window.confirm("Regenerating replaces this answer and removes the messages after it. Continue?")) return;
        sendRegenerate(chat, message);
      }, busy || !hasId, waiting || "Regenerate answer"),
    );
  }
  body.appendChild(actions);
  row.appendChild(body);
  return row;
}

function renderEditBox(chat, message) {
  const box = el("div", "edit-box");
  const textarea = el("textarea");
  textarea.value = chat.editDraft;
  textarea.setAttribute("aria-label", "Edit message");
  textarea.addEventListener("input", () => { chat.editDraft = textarea.value; });
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Escape") cancel.click();
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) save.click();
  });

  const save = el("button", "button primary", "Save & submit");
  save.type = "button";
  save.addEventListener("click", () => {
    const text = textarea.value.trim();
    if (!text || text === message.content) {
      chat.editingId = null;
      renderThread();
      return;
    }
    sendEdit(chat, message, text);
  });
  const cancel = el("button", "button secondary", "Cancel");
  cancel.type = "button";
  cancel.addEventListener("click", () => {
    chat.editingId = null;
    renderThread();
  });

  const buttons = el("div", "edit-buttons");
  buttons.append(cancel, save);
  box.append(textarea, buttons);
  return box;
}

function renderStreamingMessage(gen) {
  const row = el("div", "message assistant");
  row.appendChild(el("div", "assistant-avatar", "✳"));
  const body = el("div", "message-body");
  const bubble = el("div", "bubble");
  body.appendChild(bubble);
  // Same buttons as a saved message, from the very first paint. They are
  // disabled only until the reply is saved and has a database id.
  const actions = el("div", "actions");
  actions.append(
    actionButton("Copy", "copy", () => {}, true, "Available when the reply is finished"),
    actionButton("Regenerate", "regenerate", () => {}, true, "Wait for the reply to finish"),
  );
  body.appendChild(actions);
  row.appendChild(body);
  streamBubble = bubble;
  paintStream(gen);
  return row;
}

function isVisibleMemoryResult(item) {
  return item?.tool === "save_memory"
    && item.result?.status === "🧠 Memory saved"
    && Number.isSafeInteger(item.result.memory_id)
    && item.result.memory_id > 0;
}

function renderToolResults(chat) {
  const section = el("div", "tool-results");
  section.setAttribute("aria-label", "Tool results");
  for (const [index, item] of chat.toolResults.entries()) {
    if (!isVisibleMemoryResult(item)) continue;
    const result = el("div", "tool-result");
    result.appendChild(el(
      "div",
      "tool-result-title",
      item.undone ? "Memory removed" : "Remembered this",
    ));
    result.appendChild(el(
      "div",
      "tool-result-content",
      [item.result.title, item.result.content].filter(Boolean).join("\n"),
    ));
    if (!item.undone) {
      const undo = el(
        "button",
        "button secondary",
        item.undoing ? "Undoing…" : "Undo",
      );
      undo.type = "button";
      undo.disabled = Boolean(item.undoing);
      undo.addEventListener("click", () => undoMemory(chat, index));
      result.appendChild(undo);
    }
    section.appendChild(result);
  }
  return section;
}

async function undoMemory(chat, index) {
  const item = chat.toolResults[index];
  if (!item || item.undone || item.undoing) return;

  item.undoing = true;
  if (chats.get(activeConversationId) === chat) renderThread();
  try {
    const response = await api(`/api/memories/${item.result.memory_id}`, "DELETE");
    if (!response.ok) {
      throw new Error(errorMessage(response.data));
    }
    item.undone = true;
    if (chat.notice.startsWith("Could not undo memory:")) setNotice(chat, "");
  } catch (error) {
    const detail = error instanceof Error ? error.message : "The request failed.";
    setNotice(chat, `Could not undo memory: ${detail}`, true);
  } finally {
    item.undoing = false;
    if (chats.get(activeConversationId) === chat) renderThread();
  }
}

// Cheap update used for every chunk: only the live bubble is touched.
function paintStream(gen) {
  if (!streamBubble) return;
  if (gen.text) {
    if (streamBubble.textContent !== gen.text) streamBubble.textContent = gen.text;
  } else if (!streamBubble.firstChild) {
    const thinking = el("div", "thinking");
    thinking.setAttribute("role", "status");
    const dots = el("span", "dots");
    dots.append(el("span"), el("span"), el("span"));
    thinking.append(dots, el("span", "", "Thinking…"));
    streamBubble.appendChild(thinking);
  }
  scrollToBottomIfNeeded();
}

function createEmptyState() {
  const state = el("div", "empty-state");
  state.appendChild(el("h2", "", username ? `Hello, ${username}` : "Hello"));
  state.appendChild(el("p", "", "How can I help you today?"));

  const hints = el("div", "prompt-hints");
  for (const [prompt, operation] of promptHints) {
    const button = el("button", "hint-button");
    button.type = "button";
    button.append(el("span", "", prompt), el("span", "hint-operation", operation));
    button.addEventListener("click", () => {
      inputEl.value = prompt;
      autosizeInput();
      // With no open chat the hint starts one right away.
      if (activeConversationId === null) void submitMessage(prompt).then((ok) => { if (ok) clearComposer(); });
      else inputEl.focus();
    });
    hints.appendChild(button);
  }
  state.appendChild(hints);
  return state;
}

// ---------- Composer ----------
function autosizeInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 200)}px`;
}

function clearComposer() {
  inputEl.value = "";
  autosizeInput();
  const chat = activeChat();
  if (chat) chat.draft = "";
}

// The send button turns into Stop while THIS conversation is generating.
function updateComposer() {
  const gen = activeChat()?.generation;
  const generating = Boolean(gen);
  sendButtonEl.classList.toggle("stop", generating);
  sendButtonEl.type = generating ? "button" : "submit";
  sendButtonEl.title = generating ? "Stop generating" : "Send";
  sendButtonEl.setAttribute("aria-label", generating ? "Stop generating" : "Send message");
  sendButtonEl.replaceChildren(icon(generating ? "stop" : "send"));
  sendButtonEl.disabled = Boolean(gen?.finishing);
}

sendButtonEl.addEventListener("click", () => {
  if (activeChat()?.generation) stopActiveGeneration();
});

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = inputEl.value;
  if (!text.trim()) return;
  if (await submitMessage(text)) clearComposer();
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

inputEl.addEventListener("input", autosizeInput);

// ---------- Scrolling ----------
logEl.addEventListener("scroll", () => {
  stickToBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 80;
});

function scrollToBottomIfNeeded() {
  if (stickToBottom) logEl.scrollTop = logEl.scrollHeight;
}

// ---------- Sidebar controls ----------
newChatButtonEl.addEventListener("click", startNewChat);
loadMoreConversationsEl.addEventListener("click", () => void loadMoreConversations());
menuButtonEl.addEventListener("click", () => appEl.classList.add("sidebar-open"));
scrimEl.addEventListener("click", closeSidebar);

checkSession();
