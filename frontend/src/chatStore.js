// ===========================================================================
// All app logic lives here, independent of React. Components subscribe with
// useSyncExternalStore and re-render on every emit().
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

const HEARTBEAT_INTERVAL_MS = 25000; // how often we ask "are you alive?"
const PONG_TIMEOUT_MS = 10000; // how long we wait for the answer
const CONVERSATION_PAGE_SIZE = 20;
export const DEFAULT_TITLE = "New chat";

const NOT_CONNECTED = "Not connected — try again in a moment.";
const WAIT_FOR_REPLY = "Wait for the reply to finish, or press Stop.";
const STOPPED = "Stopped.";
// Waits between session checks while the server wakes up (about a minute in all).
const SESSION_RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 15000, 30000];

export const PROMPT_HINTS = [
  ["Remember that my backend uses FastAPI", "Save note"],
  ["Show my notes", "Retrieve notes"],
  ["Create a task to finish gRPC", "Create task"],
  ["Show my tasks", "List tasks"],
  ["Mark task 1 completed", "Complete task"],
  ["Hello", "General chat"],
];

// ---------- Pure helpers ----------

// Request ids must be unique forever: the server ignores a second save that
// reuses an id, so a counter that restarts at 1 after a page reload would make
// new answers silently disappear.
function newRequestId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function shortTitle(text) {
  const firstLine = text.trim().split(/\r?\n/)[0].replace(/\s+/g, " ");
  return firstLine.length > 48 ? `${firstLine.slice(0, 47).trimEnd()}…` : firstLine;
}

// SQLite stores "YYYY-MM-DD HH:MM:SS" in UTC without a zone marker.
export function parseTime(value) {
  if (!value) return 0;
  const iso = value.includes("T") ? value : value.replace(" ", "T");
  const time = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  return Number.isNaN(time) ? 0 : time;
}

export function groupLabel(value) {
  const day = 24 * 60 * 60 * 1000;
  const startOfToday = new Date().setHours(0, 0, 0, 0);
  const time = parseTime(value);
  if (time >= startOfToday) return "Today";
  if (time >= startOfToday - day) return "Yesterday";
  if (time >= startOfToday - 7 * day) return "Previous 7 days";
  return "Older";
}

// One validation error from the server, in plain words.
function fieldError(item) {
  const field = String(item.loc?.[item.loc.length - 1] ?? "input");
  const name = field.charAt(0).toUpperCase() + field.slice(1);
  if (field === "username" && item.type === "string_pattern_mismatch") {
    return "Username can only use letters, numbers and underscores.";
  }
  if (item.msg?.startsWith("String should ")) {
    return `${item.msg.replace("String should", `${name} should`)}.`;
  }
  return `${field}: ${item.msg}`;
}

export function errorMessage(data) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map(fieldError).join(" · ");
  return "Something went wrong. Please try again.";
}

export function isVisibleMemoryResult(item) {
  return item?.tool === "save_memory"
    && item.result?.status === "🧠 Memory saved"
    && Number.isSafeInteger(item.result.memory_id)
    && item.result.memory_id > 0;
}

function decodeBase64Url(value) {
  const padding = "=".repeat((4 - value.length % 4) % 4);
  const base64 = (value + padding).replace(/-/g, "+").replace(/_/g, "/");
  return Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
}

// Where the backend lives. Empty = same origin as the page (local dev, Docker).
// In production set VITE_API_URL (e.g. https://api.example.com) at build time.
const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");
// Set when the WebSocket host is on another site than the page (e.g. page on
// *.vercel.app with /api proxied, socket straight to wss://*.onrender.com/ws).
// The browser will not send the cookie there, so we first fetch a single-use
// ticket through the proxied API and pass it in the URL.
const WS_URL = (import.meta.env.VITE_WS_URL || "").replace(/\/+$/, "");

async function api(path, method = "GET", body) {
  // "include" sends the session cookie to the API subdomain as well.
  const options = { method, headers: {}, credentials: "include" };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(`${API_BASE}${path}`, options);
  let data = null;
  try {
    data = await response.json();
  } catch {
    // Non-JSON response bodies have no structured error details.
  }
  // The sign-in limiter answers {"error": ...}, not the usual {"detail": ...}.
  if (response.status === 429 && !data?.detail) {
    data = { detail: "Too many attempts. Please wait a few minutes and try again." };
  }
  return { ok: response.ok, status: response.status, data };
}

// ---------- The store ----------
export function createChatStore() {
  // The API host (or the page's host when API_BASE is empty), so the browser
  // sends the session cookie. https -> wss automatically.
  const wsUrl = () => {
    if (WS_URL) return WS_URL;
    const base = new URL(API_BASE || location.origin);
    return `${base.protocol === "https:" ? "wss" : "ws"}://${base.host}/ws`;
  };

  const listeners = new Set();
  let version = 0;
  function emit() {
    version += 1;
    for (const listener of listeners) listener();
  }

  // Everything the UI renders. Mutated in place; `version` signals changes.
  const state = {
    view: "loading", // "loading" | "auth" | "chat"
    loadingLabel: "Loading…", // shown while view is "loading"
    loadingFailed: false, // the server never answered: offer Retry
    authMode: "login",
    authError: "",
    authSubmitting: false,
    authFocusSeq: 0,
    username: "",
    connection: { label: "Disconnected", state: "" },
    conversations: [], // sidebar list (newest first)
    conversationOffset: 0,
    conversationTotalCount: 0,
    conversationsLoading: false,
    searchQuery: "",
    searchActive: false,
    searchResults: [],
    searchStatus: "",
    searchPending: false,
    searchFailed: false,
    listError: "", // the chat list could not be loaded
    sidebarError: "", // a sidebar action failed (not a connection problem)
    blankDraft: "", // unsent text of the blank "new chat" screen
    activeConversationId: null, // null = the blank "new chat" screen
    chats: new Map(),
    input: "", // composer text
    inputFocusSeq: 0,
    blankError: "", // an error shown on the blank "new chat" screen
    sidebarOpen: false,
    confirm: null, // {message, action} while a dialog is open
    reminders: { visible: false, enabled: false, busy: false, status: "" },
  };

  // Not rendered directly.
  let ws = null; // the one socket (null while disconnected)
  let reconnectAttempt = 0; // consecutive failed reconnects; 0 when connected
  let reconnectTimer = null;
  let sessionTimer = null;
  let sessionCheck = 0; // only the newest session check may change the screen
  let pingTimer = null;
  let pongTimer = null;
  let signedIn = false;
  let pushRegistration = null;
  let pushSubscription = null;
  let userStateGeneration = 0; // bumped on login/logout so late responses are dropped
  let conversationSearchSequence = 0;
  let creatingConversation = false;
  let navigation = 0; // bumped whenever the user moves to another chat
  const scroll = { stickToBottom: true }; // auto-scroll only while at the bottom

  /**
   * Per-conversation state. Nothing here is shared between conversations.
   *   messages   : saved messages [{message_id, role, content}]
   *   generation : null, or the reply being produced right now:
   *                {requestId, mode, prompt, assistantMessageId, text, finishing}
   *   loaded     : true once messages were fetched at least once
   *   notice     : a small status/error line shown at the end of the thread
   *   draft      : unsent text in the composer for this conversation
   */
  function chatState(conversationId) {
    let chat = state.chats.get(conversationId);
    if (!chat) {
      chat = {
        messages: [],
        generation: null,
        toolResults: [],
        loaded: false,
        notice: "",
        noticeIsError: false,
        loadError: "", // the last reload failed; offers Retry
        draft: "",
        editingId: null,
        editDraft: "",
        loadSeq: 0,
      };
      state.chats.set(conversationId, chat);
    }
    return chat;
  }

  function activeChat() {
    return state.activeConversationId === null ? null : chatState(state.activeConversationId);
  }

  function setConnection(label, connState) {
    state.connection = { label, state: connState };
    emit();
  }

  function setNotice(chat, text, isError = false) {
    chat.notice = text;
    chat.noticeIsError = isError;
  }

  function focusInput() {
    state.inputFocusSeq += 1;
  }

  // ---------- Confirmation dialog ----------
  let resolveConfirm = null;

  /** Resolves to true only if the user presses the `action` button. */
  function askConfirm(message, action) {
    answerConfirm(false); // never leave an earlier question waiting
    return new Promise((resolve) => {
      resolveConfirm = resolve;
      state.confirm = { message, action };
      emit();
    });
  }

  function answerConfirm(answer) {
    if (!resolveConfirm) return;
    const resolve = resolveConfirm;
    resolveConfirm = null;
    state.confirm = null;
    emit();
    resolve(answer);
  }

  // ---------- Task reminders (Web Push) ----------
  function setReminders(patch) {
    state.reminders = { ...state.reminders, ...patch };
    emit();
  }

  async function savePushSubscription(subscription) {
    const serialized = subscription.toJSON();
    if (!serialized.endpoint || !serialized.keys?.p256dh || !serialized.keys?.auth) {
      throw new Error("The browser returned an incomplete push subscription.");
    }
    const result = await api("/api/push/subscription", "POST", serialized);
    if (!result.ok) throw new Error(errorMessage(result.data));
  }

  async function syncTaskReminderSettings() {
    const generation = userStateGeneration;
    const stale = () => generation !== userStateGeneration;
    setReminders({ visible: false, status: "" });
    try {
      const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      if (timezone) {
        const result = await api("/api/timezone", "POST", { timezone });
        if (stale()) return;
        if (!result.ok) {
          setReminders({ status: `Timezone was not saved: ${errorMessage(result.data)}` });
        }
      }

      if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
        setReminders({ status: "Web Push is not supported in this browser." });
        return;
      }

      const config = await api("/api/push/config");
      if (stale()) return;
      if (!config.ok) throw new Error(errorMessage(config.data));
      if (!config.data?.enabled || !config.data.publicKey) {
        setReminders({ status: "Task reminders are not configured on this server." });
        return;
      }

      pushRegistration = await navigator.serviceWorker.register("/service-worker.js");
      const subscription = await pushRegistration.pushManager.getSubscription();
      if (stale()) return;
      pushSubscription = subscription;
      if (pushSubscription) {
        await savePushSubscription(pushSubscription);
        if (stale()) return;
        setReminders({
          visible: true, enabled: true, status: "Task reminders are enabled on this device.",
        });
      } else {
        setReminders({ visible: true, enabled: false });
      }
    } catch (error) {
      if (stale()) return;
      setReminders({ status: `Task reminders unavailable: ${error.message}` });
    }
  }

  async function toggleTaskReminders() {
    setReminders({ busy: true, status: "" });
    try {
      if (pushSubscription) {
        const result = await api(
          "/api/push/subscription",
          "DELETE",
          { endpoint: pushSubscription.endpoint },
        );
        if (!result.ok) throw new Error(errorMessage(result.data));
        await pushSubscription.unsubscribe();
        pushSubscription = null;
        setReminders({ enabled: false, status: "Task reminders are disabled on this device." });
        return;
      }

      const permission = await Notification.requestPermission();
      if (permission !== "granted") {
        setReminders({ status: "Allow notifications in your browser to enable reminders." });
        return;
      }

      const config = await api("/api/push/config");
      if (!config.ok || !config.data?.publicKey) {
        throw new Error(errorMessage(config.data));
      }
      pushRegistration ??= await navigator.serviceWorker.register("/service-worker.js");
      pushSubscription = await pushRegistration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: decodeBase64Url(config.data.publicKey),
      });
      await savePushSubscription(pushSubscription);
      setReminders({ enabled: true, status: "Task reminders are enabled on this device." });
    } catch (error) {
      // Chromium raises AbortError when its push service (FCM) is unreachable or
      // disabled, e.g. Brave's default "Use Google services for push messaging" = off.
      const status = error.name === "AbortError" && /push service/i.test(error.message)
        ? "Your browser's push service is unavailable. In Brave, enable "
          + "\"Use Google services for push messaging\" in brave://settings/privacy "
          + "and restart the browser, or try another browser."
        : `Could not update task reminders: ${error.message}`;
      setReminders({ status });
    } finally {
      setReminders({ busy: false });
    }
  }

  // ---------- Sign in / out ----------
  function setAuthMode(mode) {
    state.authMode = mode;
    state.authError = "";
    emit();
  }

  function toggleAuthMode() {
    setAuthMode(state.authMode === "login" ? "register" : "login");
  }

  function setCurrentUser(name) {
    state.username = name || "";
  }

  function clearUserState() {
    userStateGeneration += 1;
    answerConfirm(false);
    closeSocket();
    state.chats.clear();
    state.conversations = [];
    state.conversationOffset = 0;
    state.conversationTotalCount = 0;
    state.conversationsLoading = false;
    conversationSearchSequence += 1;
    state.searchActive = false;
    state.searchResults = [];
    state.searchQuery = "";
    state.searchStatus = "";
    state.searchPending = false;
    state.searchFailed = false;
    state.listError = "";
    state.sidebarError = "";
    state.blankDraft = "";
    state.activeConversationId = null;
    state.blankError = "";
    signedIn = false;
    ticketGeneration = null;
    pushSubscription = null;
    state.reminders = { visible: false, enabled: false, busy: false, status: "" };
    state.sidebarOpen = false;
    setCurrentUser("");
    state.input = "";
    emit();
  }

  function showAuth(message = "") {
    clearUserState();
    state.loadingFailed = false;
    state.view = "auth";
    state.authError = message;
    state.authFocusSeq += 1;
    emit();
  }

  function showChat() {
    state.view = "chat";
    signedIn = true;
    focusInput();
    emit();
  }

  // Independent of each other, so none waits for another: reminder setup can be
  // slow and must not hold back the chats or the live connection.
  function startSession() {
    void connectWebSocket();
    void loadConversations();
    void syncTaskReminderSettings();
  }

  /** Returns true when the user is now signed in. */
  async function submitAuth(rawName, password) {
    const name = rawName.trim();
    if (!name || !password) {
      state.authError = "Enter a username and password.";
      emit();
      return false;
    }

    state.authSubmitting = true;
    state.authError = "";
    emit();
    try {
      if (state.authMode === "register") {
        const registered = await api("/api/register", "POST", { username: name, password });
        if (!registered.ok) {
          state.authError = errorMessage(registered.data);
          return false;
        }
      }

      const login = await api("/api/login", "POST", { username: name, password });
      if (!login.ok) {
        const created = state.authMode === "register";
        if (created) state.authMode = "login"; // registering again would fail
        state.authError = created
          ? `Your account was created. Log in to continue. (${errorMessage(login.data)})`
          : errorMessage(login.data);
        return false;
      }

      clearUserState();
      state.authMode = "login";
      state.authError = "";
      setCurrentUser(name);
      showChat();
      startSession();
      return true;
    } catch {
      state.authError = "Could not reach the server.";
      return false;
    } finally {
      state.authSubmitting = false;
      emit();
    }
  }

  async function logout() {
    if (state.view !== "chat") return;
    // Leave the chat first, so an emptied chat screen is never shown. The login
    // screen waits for the server: a late logout reply would delete the cookie
    // of a session started in the meantime.
    const endpoint = pushSubscription?.endpoint;
    state.view = "loading";
    state.loadingLabel = "Signing out…";
    clearUserState();
    setConnection("Disconnected", "error");
    let message = "";
    if (endpoint) {
      // Stop reminders for this account on this device while still signed in.
      // The browser keeps its subscription, so the next login restores them.
      try {
        await api("/api/push/subscription", "DELETE", { endpoint });
      } catch {
        // Logging out matters more; the logout call reports an unreachable server.
      }
    }
    try {
      const result = await api("/api/logout", "POST");
      if (!result.ok) {
        message = `Could not end the server session: ${errorMessage(result.data)}`;
      }
    } catch {
      message = "Could not reach the server to end your session.";
    }
    showAuth(message);
  }

  // A sleeping server (free hosting) takes a while to answer. That is not the
  // same as being logged out, so keep asking before giving up.
  async function checkSession() {
    const check = ++sessionCheck;
    state.view = "loading";
    state.loadingLabel = "Loading…";
    state.loadingFailed = false;
    emit();
    for (let attempt = 0; ; attempt += 1) {
      let me = null;
      try {
        me = await api("/api/me");
      } catch {
        // Unreachable: retry below.
      }
      if (check !== sessionCheck) return;
      if (me?.ok) {
        clearUserState();
        setCurrentUser(me.data.username);
        showChat();
        startSession();
        return;
      }
      if (me && me.status < 500) {
        showAuth(me.status === 401 ? "" : errorMessage(me.data));
        return;
      }
      if (attempt >= SESSION_RETRY_DELAYS_MS.length) break;
      state.loadingLabel = "Waking up the server…";
      emit();
      await new Promise((resolve) => {
        sessionTimer = setTimeout(resolve, SESSION_RETRY_DELAYS_MS[attempt]);
      });
    }
    state.loadingLabel = "Could not reach the server.";
    state.loadingFailed = true;
    emit();
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

  let ticketGeneration = null; // the session that is waiting for a ticket

  function scheduleReconnect() {
    const delay = reconnectDelayMs(reconnectAttempt++);
    setConnection(`Reconnecting in ${Math.ceil(delay / 1000)}s…`, "error");
    reconnectTimer = setTimeout(connectWebSocket, delay);
  }

  async function connectWebSocket() {
    if (!signedIn || ws || ticketGeneration === userStateGeneration) return;
    const generation = userStateGeneration;
    let url = wsUrl();
    if (WS_URL) {
      ticketGeneration = generation;
      setConnection("Connecting…", "");
      let result = null;
      try {
        result = await api("/api/ws-ticket", "POST");
      } catch {
        // Server unreachable (e.g. waking up): retry below.
      } finally {
        if (ticketGeneration === generation) ticketGeneration = null;
      }
      if (generation !== userStateGeneration || !signedIn || ws) return;
      if (result?.status === 401) {
        showAuth("Your session has ended. Please log in again.");
        return;
      }
      if (!result?.ok || !result.data?.ticket) {
        scheduleReconnect();
        return;
      }
      url = `${WS_URL}?ticket=${encodeURIComponent(result.data.ticket)}`;
    }
    const socket = new WebSocket(url);
    ws = socket;
    setConnection("Connecting…", "");

    socket.onopen = () => {
      if (socket !== ws) return;
      reconnectAttempt = 0;
      forgetNotConnected();
      // Only replies that were already tracked may be ended by the answer to
      // this sync: one sent after it is not in that answer yet.
      for (const chat of state.chats.values()) {
        if (chat.generation) chat.generation.awaitingSync = true;
      }
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
      scheduleReconnect();
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

  function forgetNotConnected() {
    for (const chat of state.chats.values()) {
      if (chat.notice === NOT_CONNECTED) setNotice(chat, "");
    }
    if (state.blankError === NOT_CONNECTED) state.blankError = "";
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
        emit();
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
        emit();
        break;

      case "tool_result":
        if (!isCurrent) {
          wsSend({ type: "sync" });
          break;
        }
        if (!Number.isInteger(msg.index) || msg.index < 0 || typeof msg.tool !== "string") {
          break;
        }
        chat.toolResults = [...chat.toolResults];
        chat.toolResults[msg.index] = { tool: msg.tool, result: msg.result };
        emit();
        break;

      case "complete":
      case "cancelled":
        if (gen !== null && !isCurrent) break; // an old reply's event: ignore
        if (msg.type === "cancelled") setNotice(chat, STOPPED);
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
    for (const [conversationId, chat] of state.chats) {
      const gen = chat.generation;
      if (gen?.awaitingSync && !runningIds.has(conversationId) && !gen.finishing) {
        finishGeneration(conversationId);
      }
    }
    emit();
  }

  // A reply is over. Keep showing its text until the saved messages arrive, so
  // the screen does not flicker; then swap in the database version in one step.
  function finishGeneration(conversationId) {
    const chat = chatState(conversationId);
    if (chat.generation) chat.generation.finishing = true;
    if (chat.notice === WAIT_FOR_REPLY) setNotice(chat, "");
    void loadMessages(conversationId);
    emit();
  }

  // Put text that never reached the database back where the user can resend it,
  // without throwing away anything typed since.
  function returnPrompt(conversationId, prompt) {
    const chat = chatState(conversationId);
    const join = (typed) => (typed && typed !== prompt ? `${prompt}\n${typed}` : prompt);
    if (conversationId === state.activeConversationId) state.input = join(state.input);
    else chat.draft = join(chat.draft);
  }

  function failGeneration(conversationId, error) {
    const chat = chatState(conversationId);
    const gen = chat.generation;
    chat.generation = null;
    setNotice(chat, error || "Something went wrong.", true);
    // A normal message is only saved when the reply succeeds. Give the text
    // back so the user can resend it with one click.
    if (gen?.mode === "message") returnPrompt(conversationId, gen.prompt);
    // Reload to show what the database really holds.
    void loadMessages(conversationId).then(() => {
      if (gen?.mode === "edit") reopenRejectedEdit(chat, gen);
    });
    emit();
  }

  // The server refused an edit: offer the new text again instead of losing it.
  function reopenRejectedEdit(chat, gen) {
    const message = chat.messages.find((m) => m.message_id === gen.editMessageId);
    if (!message || message.content === gen.prompt || chat.generation) return;
    chat.editingId = message.message_id;
    chat.editDraft = gen.prompt;
    emit();
  }

  // ---------- Loading data ----------
  async function loadMessages(conversationId) {
    const chat = chatState(conversationId);
    const seq = ++chat.loadSeq; // a newer request makes older answers stale
    const generation = userStateGeneration;
    let result = null;
    try {
      result = await api(`/api/conversations/${conversationId}`);
    } catch {
      // Handled below, like an error answer.
    }
    if (generation !== userStateGeneration) return;
    if (result?.status === 401) {
      showAuth("Your session has ended. Please log in again.");
      return;
    }
    if (result?.status === 404) {
      dropConversation(conversationId);
      return;
    }
    if (seq !== chat.loadSeq) return;

    // A finished reply is cleared in the same update that shows its messages,
    // so it is never drawn twice (once live, once saved).
    const finished = chat.generation?.finishing ? chat.generation : null;
    if (finished) chat.generation = null;

    if (result?.ok) {
      const before = new Set(chat.messages.map((m) => m.message_id));
      chat.messages = result.data.messages;
      chat.loaded = true;
      chat.loadError = "";
      chat.conversation = result.data.conversation;
      updateConversationInList(result.data.conversation);
      const saved = (m) => m.role === "user" && m.content === finished.prompt
        && !before.has(m.message_id);
      if (finished?.mode === "message" && !chat.messages.some(saved)) {
        returnPrompt(conversationId, finished.prompt);
        setNotice(chat, chat.notice === STOPPED
          ? "Stopped. Your message is back in the box."
          : "Your message was not saved. It is back in the box.");
      }
    } else {
      chat.loadError = result
        ? errorMessage(result.data)
        : "Could not reach the server to load this chat.";
      if (finished) keepUnsaved(conversationId, chat, finished);
    }
    emit();
  }

  // The reload after a reply failed: keep what is on screen. These messages
  // have no database id, so their Edit and Regenerate stay disabled.
  function keepUnsaved(conversationId, chat, gen) {
    if (!gen.text) {
      if (gen.mode === "message") returnPrompt(conversationId, gen.prompt);
      return;
    }
    const reply = { message_id: null, role: "assistant", content: gen.text };
    if (gen.mode === "regenerate") {
      const index = chat.messages.findIndex((m) => m.message_id === gen.assistantMessageId);
      if (index >= 0) chat.messages = [...chat.messages.slice(0, index), reply];
      return;
    }
    const question = gen.mode === "message"
      ? [{ message_id: null, role: "user", content: gen.prompt }]
      : [];
    chat.messages = [...chat.messages, ...question, reply];
  }

  function reloadConversation(conversationId) {
    chatState(conversationId).loadError = "";
    emit();
    void loadMessages(conversationId);
  }

  function updateConversationInList(conversation) {
    const known = state.conversations.some(
      (c) => c.conversation_id === conversation.conversation_id,
    );
    if (!known) return;
    state.conversations = state.conversations.map((item) =>
      item.conversation_id === conversation.conversation_id ? conversation : item,
    );
    sortConversations();
  }

  function sortConversations() {
    state.conversations = [...state.conversations].sort(
      (a, b) => parseTime(b.updated_at) - parseTime(a.updated_at),
    );
  }

  function appendConversations(items) {
    const known = new Set(state.conversations.map((c) => c.conversation_id));
    state.conversations = [
      ...state.conversations,
      ...items.filter((item) => !known.has(item.conversation_id)),
    ];
    sortConversations(); // out-of-order rows would repeat date headings
  }

  async function loadConversations() {
    const generation = userStateGeneration;
    state.conversationsLoading = true;
    state.listError = "";
    emit();
    try {
      const result = await api(`/api/conversations?limit=${CONVERSATION_PAGE_SIZE}&offset=0`);
      if (generation !== userStateGeneration) return;
      if (!result.ok) {
        if (result.status === 401) showAuth("Your session has ended. Please log in again.");
        else state.listError = `Could not load chats: ${errorMessage(result.data)}`;
        return;
      }
      appendConversations(result.data.conversations);
      state.conversationOffset = result.data.pagination.offset + result.data.conversations.length;
      state.conversationTotalCount = result.data.pagination.total_count;
    } catch {
      if (generation === userStateGeneration) {
        state.listError = "Could not load chats: the server cannot be reached.";
      }
    } finally {
      if (generation === userStateGeneration) {
        state.conversationsLoading = false;
        emit();
      }
    }
  }

  async function loadMoreConversations() {
    if (state.conversationsLoading || state.conversationOffset >= state.conversationTotalCount) {
      return;
    }
    const generation = userStateGeneration;
    const offset = state.conversationOffset;
    state.conversationsLoading = true;
    emit();
    try {
      const result = await api(
        `/api/conversations?limit=${CONVERSATION_PAGE_SIZE}&offset=${offset}`,
      );
      if (generation !== userStateGeneration) return;
      if (!result.ok) {
        if (result.status === 401) showAuth("Your session has ended. Please log in again.");
        return;
      }
      appendConversations(result.data.conversations);
      state.conversationOffset = result.data.pagination.offset + result.data.conversations.length;
      state.conversationTotalCount = result.data.pagination.total_count;
    } catch {
      // Keep the list we have; the button stays available to retry.
    } finally {
      if (generation === userStateGeneration) {
        state.conversationsLoading = false;
        emit();
      }
    }
  }

  // ---------- Conversations: select / create / delete ----------
  function openSidebar() {
    state.sidebarOpen = true;
    emit();
  }

  function closeSidebar() {
    state.sidebarOpen = false;
  }

  function resetSearch() {
    state.searchActive = false;
    state.searchResults = [];
    state.searchQuery = "";
    state.searchStatus = "";
    state.searchPending = false;
    state.searchFailed = false;
  }

  function saveDraft() {
    const chat = activeChat();
    if (chat) chat.draft = state.input;
    else state.blankDraft = state.input;
  }

  function restoreDraft() {
    const chat = activeChat();
    state.input = chat ? chat.draft : state.blankDraft;
  }

  // Switching only changes which conversation is DISPLAYED. It does not touch
  // the socket and does not cancel anything: the other chat keeps generating.
  function selectConversation(conversationId) {
    closeSidebar();
    resetSearch();
    if (conversationId === state.activeConversationId) {
      emit();
      return;
    }
    saveDraft();
    navigation += 1;
    showConversation(conversationId);
  }

  function showConversation(conversationId) {
    state.activeConversationId = conversationId;
    state.blankError = "";
    restoreDraft();
    scroll.stickToBottom = true;
    emit();
    // Refresh from the database (skip nothing: it is cheap, and a reply may
    // have finished while we were looking at another chat).
    void loadMessages(conversationId);
  }

  // "New chat" is lazy: it shows a blank screen and the conversation is only
  // created on the server when the first message is sent. Nothing is cleared.
  function startNewChat() {
    closeSidebar();
    resetSearch();
    focusInput();
    if (state.activeConversationId === null) {
      emit();
      return;
    }
    saveDraft();
    navigation += 1;
    state.activeConversationId = null;
    state.blankError = "";
    restoreDraft();
    emit();
  }

  function showBlankError(text) {
    state.blankError = text;
    emit();
  }

  async function createConversation() {
    if (creatingConversation) return null;
    creatingConversation = true;
    const generation = userStateGeneration;
    const startedAt = navigation;
    try {
      const result = await api("/api/conversations", "POST", { title: DEFAULT_TITLE });
      if (generation !== userStateGeneration) return null;
      if (!result.ok) {
        if (result.status === 401) showAuth("Your session has ended. Please log in again.");
        else showBlankError(errorMessage(result.data));
        return null;
      }
      const created = result.data.conversation;
      state.conversations = [
        created,
        ...state.conversations.filter((c) => c.conversation_id !== created.conversation_id),
      ];
      state.conversationOffset += 1;
      state.conversationTotalCount += 1;
      const chat = chatState(created.conversation_id);
      chat.loaded = true; // brand new: nothing to fetch
      // Show it only if the user is still waiting for it.
      if (navigation === startedAt) {
        state.activeConversationId = created.conversation_id;
        state.blankError = "";
      }
      emit();
      return created.conversation_id;
    } catch {
      if (generation === userStateGeneration) {
        showBlankError("Could not reach the server to create a chat.");
      }
      return null;
    } finally {
      creatingConversation = false;
    }
  }

  async function deleteConversation(conversationId) {
    if (!await askConfirm("Delete this chat? This cannot be undone.", "Delete")) return;
    const generation = userStateGeneration;
    // Stop a running reply first so the server does not try to save into a
    // conversation that no longer exists.
    const running = state.chats.get(conversationId)?.generation;
    if (running) {
      wsSend({ type: "cancel", request_id: running.requestId, conversation_id: conversationId });
    }
    state.sidebarError = "";
    emit();
    try {
      const result = await api(`/api/conversations/${conversationId}`, "DELETE");
      if (generation !== userStateGeneration) return;
      if (!result.ok) {
        if (result.status === 401) showAuth("Your session has ended. Please log in again.");
        else showSidebarError(errorMessage(result.data));
        return;
      }
      dropConversation(conversationId);
    } catch {
      if (generation === userStateGeneration) {
        showSidebarError("Could not reach the server to delete this chat.");
      }
    }
  }

  function showSidebarError(text) {
    state.sidebarError = text;
    emit();
  }

  // Remove a conversation from the screen and from memory.
  function dropConversation(conversationId) {
    const wasListed = state.conversations.some((c) => c.conversation_id === conversationId);
    state.conversations = state.conversations.filter((c) => c.conversation_id !== conversationId);
    state.chats.delete(conversationId);
    state.searchResults = state.searchResults.filter((r) => r.conversation_id !== conversationId);
    if (state.searchActive) state.searchStatus = matchCount(state.searchResults.length);
    if (wasListed) {
      if (state.conversationOffset > 0) state.conversationOffset -= 1;
      state.conversationTotalCount = Math.max(0, state.conversationTotalCount - 1);
    }
    if (state.activeConversationId === conversationId) {
      navigation += 1;
      state.activeConversationId = null;
      // Not selectConversation: the drawer and search stay as they are.
      if (state.conversations.length) {
        showConversation(state.conversations[0].conversation_id);
        return;
      }
      restoreDraft();
    }
    emit();
  }

  // ---------- Sending ----------
  function notifyActive(text, isError) {
    const chat = activeChat();
    if (chat) {
      setNotice(chat, text, isError);
      emit();
    } else {
      showBlankError(text);
    }
  }

  function startGeneration(chat, fields) {
    chat.generation = {
      assistantMessageId: null, prompt: "", text: "", finishing: false, ...fields,
    };
    chat.toolResults = [];
    setNotice(chat, "");
    scroll.stickToBottom = true;
    emit();
  }

  /** Returns the id of the conversation the message was sent to, or null. */
  async function submitMessage(rawText) {
    const text = rawText.trim();
    if (!text) return null;

    if (!socketIsOpen()) {
      notifyActive(NOT_CONNECTED, true);
      return null;
    }
    // Fixed now: the user may look at another chat while one is being created.
    const conversationId = state.activeConversationId ?? await createConversation();
    if (conversationId === null) return null;
    const chat = chatState(conversationId);
    if (chat.generation) {
      setNotice(chat, WAIT_FOR_REPLY);
      emit();
      return null;
    }

    const requestId = newRequestId();
    const sent = wsSend({
      type: "message",
      request_id: requestId,
      conversation_id: conversationId,
      content: text,
    });
    if (!sent) {
      notifyActive(NOT_CONNECTED, true);
      return null;
    }

    // Show a title right away; the server confirms it when the reply is saved.
    let listed = state.conversations.find((c) => c.conversation_id === conversationId);
    if (!listed && chat.conversation) { // an older chat, opened from search
      listed = { ...chat.conversation };
      state.conversations = [...state.conversations, listed];
      state.conversationOffset += 1; // it is now the newest on the server
    }
    if (listed && (!listed.title || listed.title === DEFAULT_TITLE) && chat.messages.length === 0) {
      listed.title = shortTitle(text);
    }
    if (listed) listed.updated_at = new Date().toISOString();
    sortConversations();

    startGeneration(chat, { requestId, mode: "message", prompt: text });
    return conversationId;
  }

  async function submitComposer() {
    const text = state.input;
    const source = state.activeConversationId;
    if (!text.trim()) return;
    const target = await submitMessage(text);
    if (target !== null) clearSent(text, [source, target]);
  }

  // Remove the text that was just sent, and nothing typed since.
  function clearSent(text, ids) {
    if (ids.includes(state.activeConversationId) && state.input === text) state.input = "";
    for (const id of ids) {
      if (id === null) {
        if (state.blankDraft === text) state.blankDraft = "";
      } else if (state.chats.get(id)?.draft === text) {
        state.chats.get(id).draft = "";
      }
    }
    emit();
  }

  function setInput(value) {
    state.input = value;
    emit();
  }

  // Edit: change a user message and re-answer from there (later messages go).
  function startEdit(chat, message) {
    chat.editingId = message.message_id;
    chat.editDraft = message.content;
    emit();
  }

  function setEditDraft(chat, value) {
    chat.editDraft = value;
    emit();
  }

  function cancelEdit(chat) {
    chat.editingId = null;
    emit();
  }

  function saveEdit(chat, message) {
    if (chat.generation) { // never replace a reply that is still running
      setNotice(chat, WAIT_FOR_REPLY);
      emit();
      return;
    }
    const newText = chat.editDraft.trim();
    if (!newText || newText === message.content) {
      cancelEdit(chat);
      return;
    }
    const requestId = newRequestId();
    const sent = wsSend({
      type: "message",
      mode: "edit",
      message_id: message.message_id,
      request_id: requestId,
      conversation_id: state.activeConversationId,
      content: newText,
    });
    if (!sent) {
      setNotice(chat, NOT_CONNECTED, true);
      emit();
      return;
    }
    // Mirror what the server does: keep messages up to this one, new text.
    const index = chat.messages.findIndex((m) => m.message_id === message.message_id);
    chat.messages = chat.messages.slice(0, index + 1);
    chat.messages[index] = { ...message, content: newText };
    chat.editingId = null;
    startGeneration(chat, {
      requestId, mode: "edit", prompt: newText, editMessageId: message.message_id,
    });
  }

  // Regenerate: write a new answer for an existing assistant message.
  async function regenerate(chat, message, isLast) {
    if (!isLast && !await askConfirm(
      "Regenerating replaces this answer and removes the messages after it. Continue?",
      "Regenerate",
    )) return;
    // Things may have moved on while the dialog was open.
    if (chat.generation || chat !== activeChat()) return;
    const requestId = newRequestId();
    const sent = wsSend({
      type: "message",
      mode: "regenerate",
      assistant_message_id: message.message_id,
      request_id: requestId,
      conversation_id: state.activeConversationId,
      content: "",
    });
    if (!sent) {
      setNotice(chat, NOT_CONNECTED, true);
      emit();
      return;
    }
    startGeneration(chat, {
      requestId, mode: "regenerate", assistantMessageId: message.message_id,
    });
  }

  function stopActiveGeneration() {
    const chat = activeChat();
    if (!chat?.generation || chat.generation.finishing) return;
    const sent = wsSend({
      type: "cancel",
      request_id: chat.generation.requestId,
      conversation_id: state.activeConversationId,
    });
    if (sent) {
      chat.generation.stopRequested = true; // until the server confirms with "cancelled"
    } else {
      setNotice(chat, NOT_CONNECTED, true);
    }
    emit();
  }

  // A suggestion card sends at once and leaves the composer alone.
  async function useHint(prompt) {
    const sent = await submitMessage(prompt);
    if (sent === null && !state.input) {
      state.input = prompt;
      emit();
    }
  }

  // ---------- Memory undo ----------
  async function undoMemory(chat, index) {
    const item = chat.toolResults[index];
    if (!item || item.undone || item.undoing) return;

    const generation = userStateGeneration;
    item.undoing = true;
    emit();
    try {
      const response = await api(`/api/memories/${item.result.memory_id}`, "DELETE");
      if (generation !== userStateGeneration) return;
      if (response.status === 401) {
        showAuth("Your session has ended. Please log in again.");
        return;
      }
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
      emit();
    }
  }

  // ---------- Search ----------
  function setSearchQuery(value) {
    state.searchQuery = value;
    if (!value.trim()) {
      conversationSearchSequence += 1;
      state.searchActive = false;
      state.searchResults = [];
      state.searchStatus = "";
      state.searchPending = false;
      state.searchFailed = false;
    }
    emit();
  }

  function matchCount(count) {
    return `${count} matching ${count === 1 ? "chat" : "chats"}`;
  }

  async function searchPastConversations() {
    const normalized = state.searchQuery.trim();
    if (!normalized) {
      setSearchQuery("");
      return;
    }

    const generation = userStateGeneration;
    const sequence = ++conversationSearchSequence;
    state.searchActive = true;
    state.searchResults = [];
    state.searchStatus = "Searching…";
    state.searchPending = true;
    state.searchFailed = false;
    emit();
    try {
      const result = await api("/api/conversations/search", "POST", {
        query: normalized,
        limit: 20,
      });
      if (generation !== userStateGeneration || sequence !== conversationSearchSequence) return;
      if (!result.ok) {
        if (result.status === 401) showAuth("Your session has ended. Please log in again.");
        else state.searchStatus = errorMessage(result.data);
        state.searchPending = false;
        state.searchFailed = true;
        emit();
        return;
      }
      state.searchResults = result.data.results || [];
      state.searchStatus = matchCount(state.searchResults.length);
      state.searchPending = false;
      emit();
    } catch {
      if (generation === userStateGeneration && sequence === conversationSearchSequence) {
        state.searchStatus = "Could not search past chats.";
        state.searchPending = false;
        state.searchFailed = true;
        emit();
      }
    }
  }

  function closeSidebarAndEmit() {
    closeSidebar();
    emit();
  }

  return {
    state,
    scroll,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    getVersion: () => version,
    activeChat,
    actions: {
      checkSession,
      toggleAuthMode,
      submitAuth,
      logout,
      toggleTaskReminders,
      selectConversation,
      startNewChat,
      deleteConversation,
      loadMoreConversations,
      loadConversations,
      setSearchQuery,
      searchPastConversations,
      openSidebar,
      closeSidebar: closeSidebarAndEmit,
      setInput,
      submitComposer,
      stopActiveGeneration,
      useHint,
      startEdit,
      setEditDraft,
      cancelEdit,
      saveEdit,
      regenerate,
      undoMemory,
      answerConfirm,
      reloadConversation,
    },
    // Exposed for tests and shutdown.
    dispose() {
      sessionCheck += 1;
      clearTimeout(sessionTimer);
      closeSocket();
    },
  };
}
