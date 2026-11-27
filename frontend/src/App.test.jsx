// Frontend tests: render the REAL React app in jsdom with a fake server.
//
//   cd frontend && npm install && npm test
//
// The fake WebSocket lets a test play the server: it records what the page
// sends and lets the test push events (chunk, complete, ...) back.

import { act, cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, test, vi } from "vitest";
import assert from "node:assert/strict";
import App from "./App.jsx";
import { createChatStore } from "./chatStore.js";
import { StoreContext } from "./StoreContext.js";

const stores = [];
afterEach(() => {
  cleanup();
  for (const store of stores) store.dispose();
  stores.length = 0;
  delete navigator.serviceWorker;
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const tick = (ms = 5) => act(() => new Promise((resolve) => setTimeout(resolve, ms)));

/** A promise the test resolves by hand, to keep a fake request in flight. */
function deferred() {
  let release;
  const promise = new Promise((resolve) => { release = resolve; });
  return { promise, release };
}

/**
 * Boot the page with a fake backend. `db` plays the database.
 * `holds` maps an API path (no query string), optionally prefixed by a method
 * ("POST /api/x"), to a promise: that request is not answered until it resolves.
 */
async function boot({ username = "Gautam", db, modules, signedOut = false, holds = {} } = {}) {
  const { App: AppComponent = App, createChatStore: makeStore = createChatStore, StoreContext: Context = StoreContext } = modules || {};
  const sockets = [];

  class FakeWebSocket {
    static OPEN = 1;
    constructor(url) {
      this.url = url;
      this.readyState = 0;
      this.sent = [];
      this.closed = false;
      sockets.push(this);
      setTimeout(() => {
        this.readyState = 1;
        this.onopen?.();
      }, 0);
    }
    send(data) { this.sent.push(JSON.parse(data)); }
    close() { this.closed = true; this.readyState = 3; act(() => { this.onclose?.(); }); }
    serverSend(event) { act(() => { this.onmessage?.({ data: JSON.stringify(event) }); }); }
    sentOfType(type) { return this.sent.filter((m) => m.type === type); }
  }
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("confirm", () => true);

  const now = new Date().toISOString();
  db ??= {
    conversations: [
      { conversation_id: 1, title: "Chat A", created_at: now, updated_at: now },
      { conversation_id: 2, title: "Chat B", created_at: now, updated_at: now },
    ],
    messages: {
      1: [
        { message_id: 11, conversation_id: 1, role: "user", content: "question A" },
        { message_id: 12, conversation_id: 1, role: "assistant", content: "answer A" },
      ],
      2: [
        { message_id: 21, conversation_id: 2, role: "user", content: "question B" },
        { message_id: 22, conversation_id: 2, role: "assistant", content: "answer B" },
      ],
    },
  };
  db.deletedMemoryIds ??= [];
  db.calls = [];

  const json = (data, status = 200) => Promise.resolve({
    ok: status < 400, status, json: () => Promise.resolve(data),
  });
  const respond = (path, options) => {
    const method = options.method || "GET";
    db.calls.push(`${method} ${path}`);
    if (path === "/api/me") {
      if (db.meFailures > 0) {
        db.meFailures -= 1;
        return json(null, 502);
      }
      return signedOut
        ? json({ detail: "Not logged in." }, 401)
        : json({ user_id: 1, username });
    }
    if (path === "/api/register" && method === "POST") {
      return db.registerReply ? json(...db.registerReply) : json({ user_id: 1 }, 201);
    }
    if (path === "/api/login" && method === "POST") {
      return db.loginReply ? json(...db.loginReply) : json({ user_id: 1 });
    }
    if (path === "/api/push/config") return json({ enabled: true, publicKey: "AAAA" });
    if (path === "/api/push/subscription") return json({ ok: true });
    if (path === "/api/logout" && method === "POST") {
      if (db.logoutError) return json({ detail: db.logoutError }, 503);
      return json({ ok: true });
    }
    if (path === "/api/ws-ticket" && method === "POST") {
      db.ticketsIssued = (db.ticketsIssued || 0) + 1;
      return json({ ticket: `t-${db.ticketsIssued}` });
    }
    if (path === "/api/conversations/search" && method === "POST") {
      if (db.failSearch) return json({ detail: "Search is down." }, 503);
      return json({ results: db.searchResults || [] });
    }
    if (path.startsWith("/api/conversations?")) {
      if (db.failList) return json({ detail: "Service temporarily unavailable." }, 503);
      return json({
        conversations: structuredClone(db.conversations.slice(0, db.pageSize)),
        pagination: { limit: 20, offset: 0, total_count: db.conversations.length },
      });
    }
    const match = path.match(/^\/api\/conversations\/(\d+)$/);
    if (match && method === "DELETE") {
      if (db.failDelete) return json({ detail: "Could not delete." }, 503);
      db.conversations = db.conversations.filter((c) => c.conversation_id !== Number(match[1]));
      return json({ ok: true });
    }
    if (match && method === "GET") {
      const id = Number(match[1]);
      if (db.missing?.includes(id)) return json({ detail: "Conversation not found." }, 404);
      if (db.failLoad) return json({ detail: "Service temporarily unavailable." }, 503);
      return json({
        conversation: db.conversations.find((c) => c.conversation_id === id),
        // A real response is a fresh copy, never the "database" array itself.
        messages: structuredClone(db.messages[id] || []),
      });
    }
    if (path === "/api/conversations" && method === "POST") {
      const id = db.conversations.length + 1;
      const created = { conversation_id: id, title: "New chat", created_at: now, updated_at: now };
      db.conversations.unshift(created);
      db.messages[id] = [];
      return json({ conversation: created }, 201);
    }
    const memoryMatch = path.match(/^\/api\/memories\/(\d+)$/);
    if (memoryMatch && method === "DELETE") {
      if (db.sessionExpired) return json({ detail: "Not logged in." }, 401);
      if (db.deleteMemoryError) return json({ detail: db.deleteMemoryError }, 503);
      db.deletedMemoryIds.push(Number(memoryMatch[1]));
      return json({ ok: true });
    }
    return json({ detail: "not found" }, 404);
  };
  vi.stubGlobal("fetch", (path, options = {}) => {
    const route = path.split("?")[0];
    const hold = holds[`${options.method || "GET"} ${route}`] ?? holds[route];
    return hold ? hold.then(() => respond(path, options)) : respond(path, options);
  });

  const store = makeStore();
  stores.push(store);
  render(
    <Context.Provider value={store}>
      <AppComponent />
    </Context.Provider>,
  );
  // With a held request the session check may not finish, so do not wait on it.
  if (Object.keys(holds).length) act(() => { void store.actions.checkSession(); });
  else await act(() => store.actions.checkSession());
  await tick(20);

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const click = (element) => act(() => { element.click(); });
  const page = {
    window, db, sockets, store, $, $$, click,
    get socket() { return sockets[sockets.length - 1]; },
    threadText: () => $("#thread").textContent,
    sidebarItem: (title) => $$(".conversation-item").find((i) => i.textContent.includes(title)),
    async open(title) {
      click(page.sidebarItem(title).querySelector(".conversation-select"));
      await tick(20);
    },
    async send(text) {
      fireEvent.change($("#input"), { target: { value: text } });
      fireEvent.submit($("#form"));
      await tick(20);
    },
    buttons: (label) => $$("#thread .action").filter((b) => b.textContent.includes(label)),
  };
  return page;
}

test("shows the signed-in user's real name", async () => {
  const page = await boot({ username: "Gautam" });
  assert.equal(page.$("#user-name").textContent, "Gautam");
  assert.equal(page.$("#user-avatar").textContent, "G");
  assert.match(page.threadText(), /Hello, Gautam/);
});

test("switching conversations does not close the socket or cancel anything", async () => {
  const page = await boot();
  await page.open("Chat A");
  await page.send("write a long story");

  const socket = page.socket;
  const sent = socket.sentOfType("message");
  assert.equal(sent.length, 1);
  assert.equal(sent[0].conversation_id, 1);

  await page.open("Chat B");

  assert.equal(page.sockets.length, 1, "no new socket was created");
  assert.equal(socket.closed, false, "the socket was not closed");
  assert.equal(socket.sentOfType("cancel").length, 0, "no cancel was sent");
});

test("a reply keeps streaming into its own conversation while another is open", async () => {
  const page = await boot();
  await page.open("Chat A");
  await page.send("hello");
  const { request_id } = page.socket.sentOfType("message")[0];

  await page.open("Chat B");
  page.socket.serverSend({ type: "start", conversation_id: 1, request_id });
  page.socket.serverSend({ type: "chunk", conversation_id: 1, request_id, content: "Hel", offset: 0 });
  page.socket.serverSend({ type: "chunk", conversation_id: 1, request_id, content: "lo", offset: 3 });

  // Chat B shows only its own messages, and Chat A is marked as generating.
  assert.match(page.threadText(), /question B/);
  assert.doesNotMatch(page.threadText(), /Hello/);
  assert.ok(page.sidebarItem("Chat A").querySelector(".spinner"));
  assert.ok(!page.sidebarItem("Chat B").querySelector(".spinner"));

  // Coming back shows the live text and the pending question.
  await page.open("Chat A");
  assert.match(page.threadText(), /hello/);
  assert.match(page.threadText(), /Hello/);
  assert.ok(page.$("#send").classList.contains("stop"), "Stop button for the generating chat");
});

test("a reply finished in the background is shown, saved once, with working buttons", async () => {
  const page = await boot();
  await page.open("Chat A");
  await page.send("hello");
  const { request_id } = page.socket.sentOfType("message")[0];
  await page.open("Chat B");

  page.socket.serverSend({ type: "chunk", conversation_id: 1, request_id, content: "Hi there", offset: 0 });
  // The server saves to the database first, then announces "complete".
  page.db.messages[1].push(
    { message_id: 13, conversation_id: 1, role: "user", content: "hello" },
    { message_id: 14, conversation_id: 1, role: "assistant", content: "Hi there" },
  );
  page.socket.serverSend({ type: "complete", conversation_id: 1, request_id });
  await tick(20);

  assert.ok(!page.sidebarItem("Chat A").querySelector(".spinner"), "spinner is gone");

  await page.open("Chat A");
  const assistantBubbles = page.$$(".message.assistant .bubble").map((b) => b.textContent);
  assert.deepEqual(assistantBubbles, ["answer A", "Hi there"], "no duplicated reply");
  assert.ok(page.buttons("Edit").every((b) => !b.disabled), "Edit works");
  assert.ok(page.buttons("Regenerate").every((b) => !b.disabled), "Regenerate works");
});

test("starting a new chat keeps every other conversation's messages", async () => {
  const page = await boot();
  await page.open("Chat A");
  page.click(page.$("#new-chat"));
  await tick();
  assert.match(page.threadText(), /Hello, Gautam/, "blank new chat screen");

  await page.open("Chat A");
  assert.match(page.threadText(), /question A/);
  assert.match(page.threadText(), /answer A/);
});

test("the first message in a blank chat creates a conversation and sends to it", async () => {
  const page = await boot();
  page.click(page.$("#new-chat"));
  await page.send("brand new topic");

  const sent = page.socket.sentOfType("message");
  assert.equal(sent.length, 1);
  assert.equal(sent[0].conversation_id, 3, "sent to the freshly created conversation");
  assert.ok(page.$$(".conversation-item").some((i) => i.textContent.includes("brand new topic")),
    "sidebar title comes from the first message");
});

test("stale and duplicate events are ignored", async () => {
  const page = await boot();
  await page.open("Chat A");
  await page.send("hello");
  const { request_id } = page.socket.sentOfType("message")[0];

  page.socket.serverSend({ type: "chunk", conversation_id: 1, request_id, content: "Hi", offset: 0 });
  page.socket.serverSend({ type: "chunk", conversation_id: 1, request_id, content: "Hi", offset: 0 }); // duplicate
  page.socket.serverSend({ type: "chunk", conversation_id: 1, request_id: "old-request", content: "STALE", offset: 0 });
  page.socket.serverSend({
    type: "tool_result",
    conversation_id: 1,
    request_id: "old-request",
    index: 0,
    tool: "create_task",
    result: { status: "STALE" },
  });

  const streaming = page.$$(".message.assistant .bubble").pop().textContent;
  assert.equal(streaming, "Hi");
  assert.ok(!page.$(".tool-results"));
});

test("internal search results stay hidden while the assistant reply remains visible", async () => {
  const page = await boot();
  await page.open("Chat A");
  await page.send("how do I prefer answers?");
  const { request_id } = page.socket.sentOfType("message")[0];

  page.socket.serverSend({
    type: "tool_result",
    conversation_id: 1,
    request_id,
    index: 0,
    tool: "search_notes",
    result: {
      notes: [{
        title: "Memory: Answer style",
        content: "The user prefers concise answers.",
      }],
    },
  });
  assert.ok(!page.$(".tool-results"));
  assert.doesNotMatch(page.threadText(), /Memory: Answer style|prefers concise answers/);

  page.socket.serverSend({
    type: "chunk",
    conversation_id: 1,
    request_id,
    content: "I'll keep my replies concise.",
    offset: 0,
  });
  assert.match(page.threadText(), /I'll keep my replies concise/);

  page.db.messages[1].push(
    { message_id: 13, conversation_id: 1, role: "user", content: "how do I prefer answers?" },
    { message_id: 14, conversation_id: 1, role: "assistant", content: "I'll keep my replies concise." },
  );
  page.socket.serverSend({ type: "complete", conversation_id: 1, request_id });
  await tick(20);

  assert.ok(!page.$(".tool-results"));
  assert.equal(page.$$(".message.assistant .bubble").at(-1).textContent, "I'll keep my replies concise.");
});

test("saved memories show an undo action and remove the memory", async () => {
  const page = await boot();
  await page.open("Chat A");
  await page.send("I prefer concise answers");
  const { request_id } = page.socket.sentOfType("message")[0];

  page.socket.serverSend({
    type: "tool_result",
    conversation_id: 1,
    request_id,
    index: 0,
    tool: "save_memory",
    result: {
      memory_id: 42,
      title: "Answer style",
      content: "The user prefers concise answers.",
      status: "🧠 Memory saved",
    },
  });

  assert.match(page.threadText(), /Remembered this/);
  assert.match(page.threadText(), /prefers concise answers/);
  page.click(page.$(".tool-result button"));
  await tick(20);

  assert.deepEqual(page.db.deletedMemoryIds, [42]);
  assert.match(page.threadText(), /Memory removed/);
  assert.ok(!page.$(".tool-result button"));
});

test("memory undo errors are shown and can be retried", async () => {
  const page = await boot();
  page.db.deleteMemoryError = "Service temporarily unavailable.";
  await page.open("Chat A");
  await page.send("I prefer concise answers");
  const { request_id } = page.socket.sentOfType("message")[0];
  page.socket.serverSend({
    type: "tool_result",
    conversation_id: 1,
    request_id,
    index: 0,
    tool: "save_memory",
    result: {
      memory_id: 42,
      title: "Answer style",
      content: "Concise answers.",
      status: "🧠 Memory saved",
    },
  });

  page.click(page.$(".tool-result button"));
  await tick(20);

  assert.match(page.threadText(), /Could not undo memory/);
  assert.ok(page.$(".tool-result button"), "the undo action remains retryable");

  page.db.deleteMemoryError = null;
  page.click(page.$(".tool-result button"));
  await tick(20);

  assert.deepEqual(page.db.deletedMemoryIds, [42]);
  assert.doesNotMatch(page.threadText(), /Could not undo memory/);
  assert.match(page.threadText(), /Memory removed/);
});

test("internal tool results stay hidden if the assistant response is cancelled or errors", async () => {
  const page = await boot();
  await page.open("Chat A");

  await page.send("create a task");
  let { request_id } = page.socket.sentOfType("message").at(-1);
  page.socket.serverSend({
    type: "tool_result",
    conversation_id: 1,
    request_id,
    index: 0,
    tool: "create_task",
    result: { task_id: 7, status: "📌 Task added" },
  });
  page.socket.serverSend({ type: "cancelled", conversation_id: 1, request_id });
  await tick(20);
  assert.ok(!page.$(".tool-results"));
  assert.doesNotMatch(page.threadText(), /Task added/);
  assert.match(page.$(".notice").textContent, /Stopped/);

  await page.send("create another task");
  request_id = page.socket.sentOfType("message").at(-1).request_id;
  page.socket.serverSend({
    type: "tool_result",
    conversation_id: 1,
    request_id,
    index: 0,
    tool: "create_task",
    result: { task_id: 8, status: "📌 Task added" },
  });
  page.socket.serverSend({
    type: "error",
    conversation_id: 1,
    request_id,
    error: "Tool follow-up failed",
  });
  await tick(20);
  assert.ok(!page.$(".tool-results"));
  assert.doesNotMatch(page.threadText(), /task_id.*8|Task added/s);
  assert.match(page.$(".notice.error").textContent, /Tool follow-up failed/);
});

test("Edit and Regenerate are on every saved message immediately, no hover needed", async () => {
  const page = await boot();
  await page.open("Chat A");
  assert.equal(page.buttons("Edit").length, 1);
  assert.equal(page.buttons("Regenerate").length, 1);
  assert.equal(page.buttons("Edit")[0].disabled, false);
  assert.equal(page.buttons("Regenerate")[0].disabled, false);

  // During a reply they are present but disabled, never missing.
  await page.send("another");
  assert.ok(page.buttons("Edit").length >= 2);
  assert.ok(page.buttons("Edit").every((b) => b.disabled));
  assert.ok(page.buttons("Regenerate").every((b) => b.disabled));
});

test("Edit and Regenerate send the right request", async () => {
  const page = await boot();
  await page.open("Chat A");

  page.click(page.buttons("Regenerate")[0]);
  await tick();
  const regenerate = page.socket.sentOfType("message").pop();
  assert.equal(regenerate.mode, "regenerate");
  assert.equal(regenerate.assistant_message_id, 12);
  assert.equal(regenerate.conversation_id, 1);
});

test("inline edit sends the edited text for the right message", async () => {
  const page = await boot();
  await page.open("Chat A");

  page.click(page.buttons("Edit")[0]);
  await tick();
  const textarea = page.$(".edit-box textarea");
  fireEvent.change(textarea, { target: { value: "question A, reworded" } });
  page.click(page.$(".edit-box .button.primary"));
  await tick();

  const edit = page.socket.sentOfType("message").pop();
  assert.equal(edit.mode, "edit");
  assert.equal(edit.message_id, 11);
  assert.equal(edit.content, "question A, reworded");
});

test("after a reconnect the page re-syncs a reply that kept running", async () => {
  const page = await boot();
  await page.open("Chat A");
  await page.send("hello");
  const { request_id } = page.socket.sentOfType("message")[0];

  // The connection drops while the server keeps generating.
  page.socket.close();
  await tick(20);
  // Wait for the reconnect timer (>= 0.5s with backoff).
  for (let i = 0; i < 40 && page.sockets.length < 2; i++) await tick(50);
  assert.equal(page.sockets.length, 2, "reconnected");
  await tick(20);

  const fresh = page.socket;
  assert.equal(fresh.sentOfType("sync").length, 1, "asked the server what is running");
  fresh.serverSend({
    type: "active",
    generations: [{
      conversation_id: 1, request_id, mode: "message", prompt: "hello",
      assistant_message_id: null, content: "Partial so far",
      tool_results: [{
        tool: "create_task",
        result: { task_id: 7, status: "📌 Task added" },
      }],
    }],
  });
  assert.match(page.threadText(), /Partial so far/);
  assert.ok(!page.$(".tool-results"));
  assert.doesNotMatch(page.threadText(), /Task added/);
  assert.ok(page.sidebarItem("Chat A").querySelector(".spinner"));
});


test("semantic search shows conversation chunk previews and opens the selected chat", async () => {
  const page = await boot();
  page.db.searchResults = [
    {
      conversation_id: 2,
      title: "Chat B",
      excerpt: "[USER] question B\n[ASSISTANT] answer B",
      updated_at: new Date().toISOString(),
      score: 0.93,
    },
  ];

  fireEvent.change(page.$("#conversation-search"), { target: { value: "how did we solve this?" } });
  fireEvent.submit(page.$("#conversation-search-form"));
  await tick(20);

  assert.equal(page.$$(".conversation-search-result").length, 1);
  assert.match(page.$("#conversations").textContent, /answer B/);
  assert.equal(page.$("#conversation-search-status").textContent, "1 matching chat");

  page.click(page.$(".conversation-search-result"));
  await tick(20);
  assert.match(page.threadText(), /question B/);
  assert.equal(page.$$(".conversation-search-result").length, 0);
});

test("with VITE_WS_URL the socket connects to that host with a fresh single-use ticket", async () => {
  vi.stubEnv("VITE_WS_URL", "wss://api.example.onrender.com/ws");
  vi.resetModules();
  const modules = {
    App: (await import("./App.jsx")).default,
    createChatStore: (await import("./chatStore.js")).createChatStore,
    StoreContext: (await import("./StoreContext.js")).StoreContext,
  };
  try {
    const page = await boot({ modules });
    assert.equal(page.socket.url, "wss://api.example.onrender.com/ws?ticket=t-1");

    page.socket.close(); // server dropped us: reconnect must use a NEW ticket
    await tick(1500);
    assert.equal(page.socket.url, "wss://api.example.onrender.com/ws?ticket=t-2");
  } finally {
    vi.unstubAllEnvs();
    vi.resetModules();
  }
});

async function logIn(page) {
  fireEvent.change(page.$("#auth-username"), { target: { value: "Gautam" } });
  fireEvent.change(page.$("#auth-password"), { target: { value: "password123" } });
  fireEvent.submit(page.$("#auth-form"));
  await tick(20);
}

test("a loading screen is shown while the session is being checked", async () => {
  const me = deferred();
  const page = await boot({ holds: { "/api/me": me.promise } });

  assert.match(document.body.textContent, /Loading…/);

  me.release();
  await tick(20);
  assert.ok(page.$("#chat"));
  assert.doesNotMatch(document.body.textContent, /Loading…/);
});

test("logging in shows progress and locks the form until the server answers", async () => {
  const login = deferred();
  const page = await boot({ signedOut: true, holds: { "/api/login": login.promise } });
  await logIn(page);

  assert.match(page.$("#auth-submit").textContent, /Logging in…/);
  assert.ok(page.$("#auth-submit .spinner"));
  assert.equal(page.$("#auth-submit").disabled, true);
  assert.equal(page.$("#auth-switch").disabled, true);
  assert.equal(page.$("#auth-username").readOnly, true);
  assert.equal(page.$("#auth-password").readOnly, true);

  login.release();
  await tick(20);
  assert.ok(page.$("#chat"));
  assert.equal(page.$("#user-name").textContent, "Gautam");
});

test("registering shows its own progress label", async () => {
  const register = deferred();
  const page = await boot({ signedOut: true, holds: { "/api/register": register.promise } });
  page.click(page.$("#auth-switch"));
  await logIn(page);

  assert.match(page.$("#auth-submit").textContent, /Creating account…/);
});

test("chats and the socket do not wait for task reminder setup", async () => {
  const timezone = deferred();
  const page = await boot({ holds: { "/api/timezone": timezone.promise } });

  assert.ok(page.sidebarItem("Chat A"), "conversations are listed");
  assert.equal(page.sockets.length, 1, "socket connected");
  assert.equal(page.$("#connection").textContent, "Connected");
});

test("the sidebar says chats are loading instead of claiming there are none", async () => {
  const list = deferred();
  const page = await boot({ holds: { "/api/conversations": list.promise } });

  assert.match(page.$("#conversations").textContent, /Loading chats…/);
  assert.doesNotMatch(page.$("#conversations").textContent, /No chats yet/);

  list.release();
  await tick(20);
  assert.ok(page.sidebarItem("Chat A"));
  assert.doesNotMatch(page.$("#conversations").textContent, /Loading chats…/);
});

test("logging out shows progress, then the login screen", async () => {
  const logout = deferred();
  const page = await boot({ holds: { "/api/logout": logout.promise } });
  page.click(page.$("#logout"));
  await tick(20);

  assert.match(document.body.textContent, /Signing out…/);
  assert.ok(!page.$("#chat"), "no emptied chat screen while waiting");
  assert.ok(!page.$("#auth"), "cannot log in again before logout ends");

  logout.release();
  await tick(20);
  assert.ok(page.$("#auth-form"));
  assert.equal(page.$("#auth-error").textContent, "");
});

test("a failed logout is reported on the login screen", async () => {
  const page = await boot();
  page.db.logoutError = "Service temporarily unavailable.";
  page.click(page.$("#logout"));
  await tick(20);

  assert.match(page.$("#auth-error").textContent, /Could not end the server session/);
});

/** Send a message in Chat A and return its request id. */
async function sendInChatA(page, text = "hello") {
  await page.open("Chat A");
  await page.send(text);
  return page.socket.sentOfType("message").at(-1).request_id;
}

/** Drop the connection and wait until the page has opened a new socket. */
async function reconnect(page) {
  const before = page.sockets.length;
  page.socket.close();
  for (let i = 0; i < 80 && page.sockets.length === before; i++) await tick(50);
  await tick(20);
}

const memoryEvent = (request_id) => ({
  type: "tool_result", conversation_id: 1, request_id, index: 0, tool: "save_memory",
  result: { memory_id: 42, title: "Answer style", content: "Concise.", status: "🧠 Memory saved" },
});

const comesAfter = (later, earlier) => Boolean(
  earlier.compareDocumentPosition(later) & Node.DOCUMENT_POSITION_FOLLOWING,
);

test("the memory card stays below the reply while it streams and after it finishes", async () => {
  const page = await boot();
  const request_id = await sendInChatA(page);
  page.socket.serverSend(memoryEvent(request_id));
  page.socket.serverSend({ type: "chunk", conversation_id: 1, request_id, content: "Noted", offset: 0 });
  const reply = () => page.$$(".message.assistant").at(-1);

  assert.ok(comesAfter(page.$(".tool-results"), reply()), "below while streaming");

  page.db.messages[1].push(
    { message_id: 13, conversation_id: 1, role: "user", content: "hello" },
    { message_id: 14, conversation_id: 1, role: "assistant", content: "Noted" },
  );
  page.socket.serverSend({ type: "complete", conversation_id: 1, request_id });
  await tick(20);
  assert.ok(comesAfter(page.$(".tool-results"), reply()), "below after finishing");
});

test("only the live question and reply animate in, so finishing does not flicker", async () => {
  const page = await boot();
  const request_id = await sendInChatA(page);
  assert.equal(page.$$(".message.live").length, 2);

  page.db.messages[1].push(
    { message_id: 13, conversation_id: 1, role: "user", content: "hello" },
    { message_id: 14, conversation_id: 1, role: "assistant", content: "Hi" },
  );
  page.socket.serverSend({ type: "complete", conversation_id: 1, request_id });
  await tick(20);
  assert.equal(page.$$(".message.live").length, 0);
  assert.equal(page.$("#input").value, "", "a saved prompt is not put back");
});

test("stopping before anything is saved puts the message back in the box", async () => {
  const page = await boot();
  const request_id = await sendInChatA(page, "keep me");
  page.socket.serverSend({ type: "cancelled", conversation_id: 1, request_id });
  await tick(20);

  assert.equal(page.$("#input").value, "keep me");
  assert.match(page.$(".notice").textContent, /Stopped\. Your message is back in the box/);
});

test("a message lost with the connection is put back in the box", async () => {
  const page = await boot();
  await sendInChatA(page, "keep me");
  await reconnect(page);
  page.socket.serverSend({ type: "active", generations: [] });
  await tick(20);

  assert.equal(page.$("#input").value, "keep me");
  assert.match(page.$(".notice").textContent, /not saved.*back in the box/);
});

test("an older sync answer does not end a reply that was just sent", async () => {
  const page = await boot();
  await sendInChatA(page, "just sent");
  page.socket.serverSend({ type: "active", generations: [] });
  await tick(20);

  assert.ok(page.$("#send").classList.contains("stop"));
  assert.equal(page.$("#input").value, "");
});

test("a finished reply stays on screen if reloading fails, and Retry reloads it", async () => {
  const page = await boot();
  const request_id = await sendInChatA(page);
  page.socket.serverSend({ type: "chunk", conversation_id: 1, request_id, content: "Hi there", offset: 0 });
  page.db.failLoad = true;
  page.socket.serverSend({ type: "complete", conversation_id: 1, request_id });
  await tick(20);

  assert.match(page.threadText(), /Hi there/);
  assert.match(page.threadText(), /hello/);
  assert.ok(!page.$("#send").classList.contains("stop"), "the composer is usable again");

  page.db.failLoad = false;
  page.db.messages[1].push(
    { message_id: 13, conversation_id: 1, role: "user", content: "hello" },
    { message_id: 14, conversation_id: 1, role: "assistant", content: "Hi there" },
  );
  page.click(page.$(".retry"));
  await tick(20);
  assert.ok(!page.$(".retry"));
  assert.equal(page.$$(".message.assistant .bubble").filter((b) => b.textContent === "Hi there").length, 1);
  assert.ok(page.buttons("Edit").every((b) => !b.disabled));
});

test("a chat that fails to load shows the error and a Retry button", async () => {
  const page = await boot();
  page.db.failLoad = true;
  await page.open("Chat A");

  assert.doesNotMatch(page.threadText(), /Loading chat/);
  assert.match(page.threadText(), /Service temporarily unavailable/);

  page.db.failLoad = false;
  page.click(page.$(".retry"));
  await tick(20);
  assert.match(page.threadText(), /answer A/);
});

test("an edit cannot be saved while a reply is running", async () => {
  const page = await boot();
  await page.open("Chat A");
  page.click(page.buttons("Edit")[0]);
  await page.send("something else");

  assert.equal(page.$(".edit-box .button.primary").disabled, true);
  fireEvent.keyDown(page.$(".edit-box textarea"), { key: "Enter", ctrlKey: true });
  assert.equal(page.socket.sentOfType("message").length, 1, "the running reply is not replaced");
});

test("a rejected edit reopens the edit box with the new text", async () => {
  const page = await boot();
  await page.open("Chat A");
  page.click(page.buttons("Edit")[0]);
  fireEvent.change(page.$(".edit-box textarea"), { target: { value: "reworded" } });
  page.click(page.$(".edit-box .button.primary"));
  const { request_id } = page.socket.sentOfType("message").at(-1);

  page.socket.serverSend({ type: "error", conversation_id: 1, request_id, error: "Too many messages." });
  await tick(20);

  assert.equal(page.$(".edit-box textarea").value, "reworded");
});

test("Stop stays usable when it cannot be sent", async () => {
  const page = await boot();
  await sendInChatA(page);
  page.socket.readyState = 3; // the connection died but no close event yet
  page.click(page.$("#send"));

  assert.equal(page.$("#send").disabled, false);
  assert.match(page.$(".notice.error").textContent, /Not connected/);
});

test("the wait notice goes away when the reply finishes", async () => {
  const page = await boot();
  const request_id = await sendInChatA(page);
  await page.send("too soon");
  assert.match(page.threadText(), /Wait for the reply to finish, or press Stop/);

  page.socket.serverSend({ type: "complete", conversation_id: 1, request_id });
  await tick(20);
  assert.doesNotMatch(page.threadText(), /Wait for the reply to finish, or press Stop/);
});

test("the not-connected notice goes away after reconnecting", async () => {
  const page = await boot();
  await page.open("Chat A");
  page.socket.readyState = 3;
  await page.send("hello");
  assert.match(page.threadText(), /Not connected/);

  await reconnect(page);
  assert.doesNotMatch(page.threadText(), /Not connected/);
});

/** Start sending from a blank chat while the server is still creating it. */
async function sendWhileCreating(text = "first message") {
  const create = deferred();
  const page = await boot({ holds: { "POST /api/conversations": create.promise } });
  await page.send(text);
  return { page, create };
}

const type = (page, value) => fireEvent.change(page.$("#input"), { target: { value } });

test("moving to another chat while a new one is created does not pull you back", async () => {
  const { page, create } = await sendWhileCreating();
  await page.open("Chat B");
  type(page, "draft for B");
  create.release();
  await tick(20);

  assert.equal(page.socket.sentOfType("message")[0].conversation_id, 3);
  assert.match(page.threadText(), /question B/);
  assert.equal(page.$("#input").value, "draft for B");

  page.click(page.$("#new-chat"));
  assert.equal(page.$("#input").value, "", "the sent text is not left as a draft");
});

test("text typed while a new chat is created is kept", async () => {
  const { page, create } = await sendWhileCreating();
  type(page, "next thought");
  create.release();
  await tick(20);

  assert.equal(page.socket.sentOfType("message")[0].content, "first message");
  assert.equal(page.$("#input").value, "next thought");
});

test("the unsent text of a new chat survives visiting another chat", async () => {
  const page = await boot();
  type(page, "an idea");
  await page.open("Chat A");
  assert.equal(page.$("#input").value, "");
  page.click(page.$("#new-chat"));
  assert.equal(page.$("#input").value, "an idea");
});

test("a suggestion card sends at once, also in an empty chat, and keeps typed text", async () => {
  const page = await boot();
  page.db.messages[1] = [];
  await page.open("Chat A");
  type(page, "half typed");
  page.click(page.$(".hint-button"));
  await tick(20);

  assert.equal(page.socket.sentOfType("message")[0].conversation_id, 1);
  assert.equal(page.$("#input").value, "half typed");
});

test("an older chat opened from search has its own title and joins the list when used", async () => {
  const now = new Date().toISOString();
  const chat = (id, title) => ({ conversation_id: id, title, created_at: now, updated_at: now });
  const page = await boot({
    db: {
      pageSize: 2,
      conversations: [chat(1, "Chat A"), chat(2, "Chat B"), chat(3, "Old chat")],
      messages: { 3: [{ message_id: 31, conversation_id: 3, role: "user", content: "long ago" }] },
      searchResults: [{ conversation_id: 3, title: "Old chat", excerpt: "long ago" }],
    },
  });
  fireEvent.change(page.$("#conversation-search"), { target: { value: "long ago" } });
  fireEvent.submit(page.$("#conversation-search-form"));
  await tick(20);
  page.click(page.$(".conversation-search-result"));
  await tick(20);
  assert.equal(page.$("#chat-title").textContent, "Old chat");

  await page.send("again");
  assert.ok(page.sidebarItem("Old chat"));
});

test("search does not claim there are no matches while searching or after an error", async () => {
  const search = deferred();
  const page = await boot({ holds: { "/api/conversations/search": search.promise } });
  page.db.failSearch = true;
  fireEvent.change(page.$("#conversation-search"), { target: { value: "x" } });
  fireEvent.submit(page.$("#conversation-search-form"));
  await tick();
  assert.doesNotMatch(page.$("#conversations").textContent, /No matching chats/);

  search.release();
  await tick(20);
  assert.match(page.$("#conversation-search-status").textContent, /Search is down/);
  assert.doesNotMatch(page.$("#conversations").textContent, /No matching chats/);
});

test("a chat that no longer exists is removed from the search results", async () => {
  const page = await boot();
  const request_id = await sendInChatA(page);
  await page.open("Chat B"); // the deleted chat is not the one on screen
  page.db.searchResults = [{ conversation_id: 1, title: "Chat A", excerpt: "gone" }];
  fireEvent.change(page.$("#conversation-search"), { target: { value: "gone" } });
  fireEvent.submit(page.$("#conversation-search-form"));
  await tick(20);

  page.db.missing = [1];
  page.socket.serverSend({ type: "complete", conversation_id: 1, request_id });
  await tick(20);
  assert.ok(!page.$(".conversation-search-result"));
});

test("a chat list that fails to load can be retried", async () => {
  const list = deferred();
  const page = await boot({ holds: { "/api/conversations": list.promise } });
  page.db.failList = true;
  list.release();
  await tick(20);

  assert.doesNotMatch(page.$("#sidebar").textContent, /No chats yet/);
  assert.match(page.$(".sidebar-error").textContent, /Could not load chats/);
  assert.equal(page.$("#connection").textContent, "Connected");

  page.db.failList = false;
  page.click(page.$(".sidebar-error .retry"));
  await tick(20);
  assert.ok(page.sidebarItem("Chat A"));
  assert.ok(!page.$(".sidebar-error"));
});

test("a failed delete does not change the connection status", async () => {
  const page = await boot();
  page.db.failDelete = true;
  page.click(page.sidebarItem("Chat A").querySelector(".conversation-delete"));
  await tick(20);

  assert.equal(page.$("#connection").textContent, "Connected");
  assert.match(page.$(".sidebar-error").textContent, /Could not delete/);
});

test("each date heading appears once", async () => {
  const today = new Date().toISOString();
  const old = "2020-01-01 00:00:00";
  const chat = (id, updated_at) => ({ conversation_id: id, title: `Chat ${id}`, updated_at });
  const page = await boot({
    db: { conversations: [chat(1, today), chat(2, old), chat(3, today)], messages: {} },
  });
  assert.deepEqual(page.$$(".conversation-group").map((g) => g.textContent), ["Today", "Older"]);
});

test("deleting the open chat keeps the phone drawer open", async () => {
  const page = await boot();
  await page.open("Chat A");
  page.click(page.$("#menu"));
  page.click(page.sidebarItem("Chat A").querySelector(".conversation-delete"));
  await tick(20);

  assert.ok(page.$("#chat").classList.contains("sidebar-open"));
  assert.match(page.threadText(), /question B/);
});

async function logOut(page) {
  page.click(page.$("#logout"));
  await tick(20);
}

test("logging out closes the phone drawer and forgets reminder status", async () => {
  const page = await boot();
  page.click(page.$("#menu"));
  assert.ok(page.store.state.reminders.status);
  await logOut(page);
  assert.equal(page.store.state.reminders.status, "");
  await logIn(page);

  assert.ok(!page.$("#chat").classList.contains("sidebar-open"));
});

test("a late reminder answer is ignored after logout", async () => {
  const timezone = deferred();
  const page = await boot({ holds: { "/api/timezone": timezone.promise } });
  await logOut(page);
  timezone.release();
  await tick(20);

  assert.equal(page.store.state.reminders.status, "");
});

test("logging out stops reminders for this device on the server", async () => {
  let unsubscribed = false;
  const subscription = {
    endpoint: "https://fcm.googleapis.com/x",
    toJSON() { return { endpoint: this.endpoint, keys: { p256dh: "p", auth: "a" } }; },
    unsubscribe: async () => { unsubscribed = true; },
  };
  vi.stubGlobal("PushManager", class {});
  Object.defineProperty(navigator, "serviceWorker", {
    configurable: true,
    value: {
      register: async () => ({ pushManager: { getSubscription: async () => subscription } }),
    },
  });
  const page = await boot();
  await logOut(page);

  const removed = page.db.calls.indexOf("DELETE /api/push/subscription");
  assert.ok(removed >= 0 && removed < page.db.calls.indexOf("POST /api/logout"));
  assert.equal(unsubscribed, false, "the next login can restore reminders");
});

test("too many login attempts are explained", async () => {
  const page = await boot({ signedOut: true });
  page.db.loginReply = [{ error: "Rate limit exceeded: 10 per 15 minute" }, 429];
  await logIn(page);

  assert.match(page.$("#auth-error").textContent, /Too many attempts/);
});

test("a rejected username is explained in plain words", async () => {
  const page = await boot({ signedOut: true });
  page.db.registerReply = [{ detail: [{
    type: "string_pattern_mismatch", loc: ["body", "username"],
    msg: "String should match pattern '^[A-Za-z0-9_]+$'",
  }] }, 422];
  page.click(page.$("#auth-switch"));
  await logIn(page);

  assert.equal(
    page.$("#auth-error").textContent,
    "Username can only use letters, numbers and underscores.",
  );
});

test("if the login after registering fails, the login form is shown", async () => {
  const page = await boot({ signedOut: true });
  page.db.loginReply = [{ detail: "Service temporarily unavailable." }, 503];
  page.click(page.$("#auth-switch"));
  await logIn(page);

  assert.equal(page.$("#auth-title").textContent, "Log in");
  assert.match(page.$("#auth-error").textContent, /account was created/i);
});

test("a sleeping server is retried instead of showing the login screen", async () => {
  const page = await boot({ db: { meFailures: 1, conversations: [], messages: {} }, holds: { "GET /none": Promise.resolve() } });
  assert.match(document.body.textContent, /Waking up the server…/);
  assert.ok(!page.$("#auth-form"));

  await tick(1300);
  assert.ok(page.$("#chat"));
});

test("if the server never wakes up, Retry is offered", async () => {
  vi.useFakeTimers();
  let up = false;
  vi.stubGlobal("fetch", () => (up
    ? Promise.resolve({ ok: false, status: 401, json: async () => ({}) })
    : Promise.reject(new TypeError("offline"))));
  const store = createChatStore();
  stores.push(store);
  render(<StoreContext.Provider value={store}><App /></StoreContext.Provider>);
  act(() => { void store.actions.checkSession(); });
  await act(() => vi.advanceTimersByTimeAsync(120000));

  assert.match(document.body.textContent, /Could not reach the server/);
  up = true;
  await act(async () => {
    document.querySelector(".retry").click();
    await vi.advanceTimersByTimeAsync(10);
  });
  assert.ok(document.querySelector("#auth-form"));
});

test("logging in again while a socket ticket is pending still connects", async () => {
  vi.stubEnv("VITE_WS_URL", "wss://api.example.onrender.com/ws");
  vi.resetModules();
  const modules = {
    App: (await import("./App.jsx")).default,
    createChatStore: (await import("./chatStore.js")).createChatStore,
    StoreContext: (await import("./StoreContext.js")).StoreContext,
  };
  try {
    const ticket = deferred();
    const page = await boot({ modules, holds: { "/api/ws-ticket": ticket.promise } });
    await logOut(page);
    await logIn(page);
    ticket.release();
    await tick(20);
    assert.equal(page.sockets.length, 1);
  } finally {
    vi.unstubAllEnvs();
    vi.resetModules();
  }
});

test("undoing a memory after the session ended shows the login screen", async () => {
  const page = await boot();
  const request_id = await sendInChatA(page);
  page.socket.serverSend(memoryEvent(request_id));
  page.db.sessionExpired = true;
  page.click(page.$(".tool-result button"));
  await tick(20);

  assert.match(page.$("#auth-error").textContent, /session has ended/);
});

test("a failed copy says so", async () => {
  const page = await boot();
  await page.open("Chat A");
  page.click(page.buttons("Copy")[0]);
  await tick();

  assert.equal(page.buttons("Copy failed").length, 1);
});

test("on a touch screen Enter adds a new line instead of sending", async () => {
  vi.stubGlobal("matchMedia", (query) => ({ matches: query === "(pointer: coarse)" }));
  const page = await boot();
  type(page, "line one");
  const notCancelled = fireEvent.keyDown(page.$("#input"), { key: "Enter" });
  await tick(20);

  assert.equal(notCancelled, true);
  assert.equal(page.socket.sentOfType("message").length, 0);
  assert.ok(!page.$(".composer-note"));
});
