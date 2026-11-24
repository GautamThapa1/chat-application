// Frontend tests: run the REAL frontend/app.js in jsdom with a fake server.
//
//   cd tests/frontend && npm install && npm test
//
// The fake WebSocket lets a test play the server: it records what the page
// sends and lets the test push events (chunk, complete, ...) back.

import { afterEach, test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "frontend");
const html = readFileSync(join(root, "index.html"), "utf8").replace(
  /<script src="app.js"><\/script>/,
  "",
);
const appJs = readFileSync(join(root, "app.js"), "utf8");
const openWindows = new Set();

afterEach(() => {
  for (const window of openWindows) window.close();
  openWindows.clear();
});

const tick = (ms = 5) => new Promise((resolve) => setTimeout(resolve, ms));

/** Boot the page with a fake backend. `db` plays the database. */
async function boot({ username = "Gautam", db } = {}) {
  const dom = new JSDOM(html, { runScripts: "outside-only", url: "http://localhost:8000/" });
  const { window } = dom;
  openWindows.add(window);
  const sockets = [];

  class FakeWebSocket {
    static OPEN = 1;
    constructor() {
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
    close() { this.closed = true; this.readyState = 3; this.onclose?.(); }
    serverSend(event) { this.onmessage?.({ data: JSON.stringify(event) }); }
    sentOfType(type) { return this.sent.filter((m) => m.type === type); }
  }
  window.WebSocket = FakeWebSocket;
  window.confirm = () => true;

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

  const json = (data, status = 200) => Promise.resolve({
    ok: status < 400, status, json: () => Promise.resolve(data),
  });
  window.fetch = (path, options = {}) => {
    const method = options.method || "GET";
    if (path === "/api/me") return json({ user_id: 1, username });
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
    return json({ detail: "not found" }, 404);
  };

  window.eval(appJs);
  await tick(20);

  const $ = (selector) => window.document.querySelector(selector);
  const $$ = (selector) => [...window.document.querySelectorAll(selector)];
  const page = {
    window, db, sockets, $, $$,
    get socket() { return sockets[sockets.length - 1]; },
    threadText: () => $("#thread").textContent,
    sidebarItem: (title) => $$(".conversation-item").find((i) => i.textContent.includes(title)),
    async open(title) {
      page.sidebarItem(title).querySelector(".conversation-select").click();
      await tick(20);
    },
    async send(text) {
      $("#input").value = text;
      $("#form").dispatchEvent(new window.Event("submit", { cancelable: true, bubbles: true }));
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
  await page.$("#new-chat").click();
  await tick();
  assert.match(page.threadText(), /Hello, Gautam/, "blank new chat screen");

  await page.open("Chat A");
  assert.match(page.threadText(), /question A/);
  assert.match(page.threadText(), /answer A/);
});

test("the first message in a blank chat creates a conversation and sends to it", async () => {
  const page = await boot();
  await page.$("#new-chat").click();
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

test("tool results stream separately from assistant text and remain visible on completion", async () => {
  const page = await boot();
  await page.open("Chat A");
  await page.send("create a task");
  const { request_id } = page.socket.sentOfType("message")[0];

  page.socket.serverSend({
    type: "tool_result",
    conversation_id: 1,
    request_id,
    index: 0,
    tool: "create_task",
    result: { task_id: 7, status: "📌 Task added" },
  });
  assert.match(page.$(".tool-result-content").textContent, /Task added/);
  assert.doesNotMatch(
    page.$$(".message.assistant .bubble").pop().textContent,
    /Task added/,
  );

  page.socket.serverSend({
    type: "chunk",
    conversation_id: 1,
    request_id,
    content: "Task created.",
    offset: 0,
  });
  assert.equal(page.$$(".message.assistant .bubble").pop().textContent, "Task created.");

  page.db.messages[1].push(
    { message_id: 13, conversation_id: 1, role: "user", content: "create a task" },
    { message_id: 14, conversation_id: 1, role: "assistant", content: "Task created." },
  );
  page.socket.serverSend({ type: "complete", conversation_id: 1, request_id });
  await tick(20);

  assert.match(page.$(".tool-result-content").textContent, /Task added/);
  assert.equal(page.$$(".message.assistant .bubble").at(-1).textContent, "Task created.");
});

test("tool results remain visible if the assistant response is cancelled or errors", async () => {
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
  assert.match(page.$(".tool-result-content").textContent, /Task added/);
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
  assert.match(page.$(".tool-result-content").textContent, /task_id.*8/s);
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

  page.buttons("Regenerate")[0].click();
  await tick();
  const regenerate = page.socket.sentOfType("message").pop();
  assert.equal(regenerate.mode, "regenerate");
  assert.equal(regenerate.assistant_message_id, 12);
  assert.equal(regenerate.conversation_id, 1);
});

test("inline edit sends the edited text for the right message", async () => {
  const page = await boot();
  await page.open("Chat A");

  page.buttons("Edit")[0].click();
  await tick();
  const textarea = page.$(".edit-box textarea");
  textarea.value = "question A, reworded";
  textarea.dispatchEvent(new page.window.Event("input", { bubbles: true }));
  page.$(".edit-box .button.primary").click();
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
  assert.match(page.$(".tool-result-content").textContent, /Task added/);
  assert.ok(page.sidebarItem("Chat A").querySelector(".spinner"));
});
