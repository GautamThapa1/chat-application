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
});

const tick = (ms = 5) => act(() => new Promise((resolve) => setTimeout(resolve, ms)));

/** Boot the page with a fake backend. `db` plays the database. */
async function boot({ username = "Gautam", db, modules } = {}) {
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

  const json = (data, status = 200) => Promise.resolve({
    ok: status < 400, status, json: () => Promise.resolve(data),
  });
  vi.stubGlobal("fetch", (path, options = {}) => {
    const method = options.method || "GET";
    if (path === "/api/me") return json({ user_id: 1, username });
    if (path === "/api/ws-ticket" && method === "POST") {
      db.ticketsIssued = (db.ticketsIssued || 0) + 1;
      return json({ ticket: `t-${db.ticketsIssued}` });
    }
    if (path === "/api/conversations/search" && method === "POST") {
      return json({ results: db.searchResults || [] });
    }
    if (path.startsWith("/api/conversations?")) {
      return json({
        conversations: db.conversations,
        pagination: { limit: 20, offset: 0, total_count: db.conversations.length },
      });
    }
    const match = path.match(/^\/api\/conversations\/(\d+)$/);
    if (match && method === "GET") {
      const id = Number(match[1]);
      return json({
        conversation: db.conversations.find((c) => c.conversation_id === id),
        messages: db.messages[id] || [],
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
      if (db.deleteMemoryError) return json({ detail: db.deleteMemoryError }, 503);
      db.deletedMemoryIds.push(Number(memoryMatch[1]));
      return json({ ok: true });
    }
    return json({ detail: "not found" }, 404);
  });

  const store = makeStore();
  stores.push(store);
  render(
    <Context.Provider value={store}>
      <AppComponent />
    </Context.Provider>,
  );
  await act(() => store.actions.checkSession());
  await tick(20);

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const click = (element) => act(() => { element.click(); });
  const page = {
    window, db, sockets, $, $$, click,
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
  assert.equal(page.$(".tool-results"), null);
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
  assert.equal(page.$(".tool-results"), null);
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

  assert.equal(page.$(".tool-results"), null);
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
  assert.equal(page.$(".tool-result button"), null);
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
  assert.equal(page.$(".tool-results"), null);
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
  assert.equal(page.$(".tool-results"), null);
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
  assert.equal(page.$(".tool-results"), null);
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
  assert.match(page.$("#conversation-search-status").textContent, /1 matching chats/);

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
