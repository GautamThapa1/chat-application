# Luma — AI Chat Assistant

A modern AI-powered chat application built with **Python, FastAPI, and Google Gemini**. Luma combines real-time AI conversations with persistent chat history, note management, and task management.

[GitHub repository](https://github.com/GautamThapa1/chat-application)

## ✨ Features

- **AI Conversations** — Stream real-time responses using Google Gemini.
- **User Authentication** — Secure registration, login, and session management.
- **Chat History** — Create, manage, delete, and semantically search conversations.
- **Notes Management** — Save, search, and retrieve personal notes.
- **Long-Term Memory** — Automatically retain useful, stable user context separately from notes, retrieve it when relevant, and undo a newly saved memory.
- **Task Management** — Create, list, reschedule, complete, and delete tasks; clearing all tasks requires explicit confirmation.
- **Task Reminders** — Optional Web Push reminders 15 minutes before task deadlines, including when the app is closed.
- **Gemini Function Calling** — Execute note, memory, and task actions; clearing notes or tasks requires explicit confirmation.
- **gRPC Communication** — Separate application logic from the backend service.
- **SQLite Storage** — Persistent data storage.

## 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.12+ | Backend |
| FastAPI | API and WebSocket server |
| Google Gemini | Streaming chat and native function calling |
| gRPC | Inter-service communication |
| SQLite | Database |
| HTML, CSS, JavaScript | Frontend |
| uv | Dependency management |

## 📁 Project Structure

```text
backend/
├── orchestrator/    # AI orchestration and WebSockets
├── service/         # gRPC service and database
├── proto/           # Protocol Buffer definitions
└── generated/       # Generated gRPC code

frontend/            # Web interface
tests/               # Automated tests

Makefile
pyproject.toml
uv.lock
```

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/GautamThapa1/chat-application.git
cd chat-application
```

**2. Install dependencies**

```bash
uv sync
```

**3. Configure environment variables**

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_MAX_INPUT_TOKENS=24000
DATABASE_KEY=your_long_random_database_key
VAPID_PRIVATE_KEY="your VAPID private key in PEM format"
VAPID_PUBLIC_KEY=your VAPID application server public key
VAPID_SUBJECT=mailto:you@example.com
```

Web Push requires HTTPS (localhost is allowed for development). Generate a
VAPID key pair with `uv run vapid --gen`, then set the private PEM as
`VAPID_PRIVATE_KEY` and the browser application-server key from
`uv run vapid --applicationServerKey --private-key <private-key-file>` as
`VAPID_PUBLIC_KEY`. Keep the private key secret and out of version control.
After signing in, use **Enable task reminders** and allow browser notifications.
The browser's timezone is saved with the account; date-only task deadlines use
9:00 AM in that timezone. A reminder is scheduled for 15 minutes before the
deadline, and its push-service TTL is limited to the remaining time before that
deadline.

Notes are indexed with Gemini embeddings. The gRPC service queues new and
updated notes, backfills existing notes at startup, and retries failed
embeddings without delaying or undoing note saves. The encrypted SQLite
database remains the source of truth for note content and indexing jobs;
Qdrant stores note vectors and performs owner-scoped nearest-neighbor search.
Existing SQLite vectors are moved to Qdrant on the next indexing pass without
re-embedding when their model and content hash are still current.

Memories are stored in a separate encrypted SQLite table and indexed in a
distinct Qdrant collection by the shared background worker. Conversation history
is indexed in its own Qdrant collection: each user prompt and its assistant
reply form a chunk, and long turns are split at natural text boundaries. The
worker backfills existing conversations, incrementally embeds new turns, and
reindexes edited branches. `/api/conversations/search` embeds each query with
the same Gemini model and returns owner-scoped, semantically ranked chunk
previews, with keyword-only fallback when query embedding is unavailable. The
`/api/memories` list, search, create, update, and delete endpoints provide a
separate management API for a future Settings page. Legacy notes titled
`Memory: ...` are moved into memory storage at startup. Search combines
semantic similarity with keyword matching and falls back to keyword-only
search if query embedding fails.

Start a local Qdrant instance before the gRPC service:

```bash
docker compose up -d qdrant
```

By default, the service connects to `http://localhost:6333`. Set `QDRANT_URL`
and, for authenticated deployments, `QDRANT_API_KEY` in `.env` to override the
connection. Qdrant data persists in the `qdrant_storage` Docker volume.

Chat history is trimmed to the newest complete turns that fit this input-token
budget. The current message is included in the count; messages exceeding the
budget are rejected. The default budget is `24000`.

`DATABASE_KEY` encrypts the SQLite database with SQLCipher. Keep it secret and
back it up securely: the database cannot be opened if the key is lost. On first
startup, an existing plaintext `app.db` is migrated to an encrypted database.

**4. Start the gRPC service**

```bash
make service
```

**5. Start the application**

In a separate terminal:

```bash
make back
```

**6. Open the application**

Visit [http://localhost:8000](http://localhost:8000).

## 🐳 Docker

Run the whole stack (Qdrant, gRPC service, web app) with one command:

```bash
cp .env.example .env        # then set GEMINI_API_KEY and DATABASE_KEY
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000).

- The SQLite database lives in the `app_data` volume and Qdrant vectors in
  `qdrant_storage`, so data survives `docker compose down`. Only
  `docker compose down -v` deletes it.
- Only port `8000` (web) is published. The gRPC port stays on the private
  compose network.
- Keep **one** app worker. Running replies are tracked in the app process's
  memory, so do not scale `app` or add uvicorn workers.
- Back up `DATABASE_KEY`: the encrypted database cannot be opened without it.

| Variable | Used by | Purpose |
|---|---|---|
| `GRPC_TARGET` | app | Address of the gRPC service (Compose sets `service:50051`; local default is `127.0.0.1:50051`) |
| `GRPC_BIND` | service | Interface the gRPC server listens on (default `127.0.0.1`) |
| `DB_PATH` | service | Database file path (default `app.db`) |
| `QDRANT_URL` | service | Qdrant address (default `http://localhost:6333`) |
| `QDRANT_CONVERSATION_COLLECTION_PREFIX` | service | Optional conversation-vector collection prefix (default `note_embeddings_conversations`, derived from `QDRANT_COLLECTION_PREFIX`) |

## 🔁 Continuous Integration

`.github/workflows/ci.yml` runs on every pull request and push to `main`:

1. **Backend** — installs locked dependencies, checks the generated gRPC code
   matches `chat.proto` (`make proto`), and runs `pytest`.
2. **Frontend** — runs the jsdom tests in `tests/frontend`.
3. **Docker** — builds the image, smoke-tests its imports, and validates
   `compose.yaml`. Pushes to `main` and `v*` tags also publish the image to
   `ghcr.io/<owner>/<repo>`.

CI needs no secrets of its own: tests use a throwaway database key and an
in-memory Qdrant.

## 🧪 Testing

Run the test suite (set any `DATABASE_KEY` first, or put one in `.env`):

```bash
DATABASE_KEY=test-key make test
```

Frontend tests (needs Node 22):

```bash
cd tests/frontend && npm ci && npm test
```

Regenerate gRPC files after modifying the Protocol Buffer definitions:

```bash
make proto
```

## 📄 License

This project is currently available for personal and educational use.
