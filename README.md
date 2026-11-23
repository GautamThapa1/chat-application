# Luma — AI Chat Assistant

A modern AI-powered chat application built with **Python, FastAPI, and Google Gemini**. Luma combines real-time AI conversations with persistent chat history, note management, and task management.

[GitHub repository](https://github.com/GautamThapa1/chat-application)

## ✨ Features

- **AI Conversations** — Stream real-time responses using Google Gemini.
- **User Authentication** — Secure registration, login, and session management.
- **Chat History** — Create, manage, and delete conversations.
- **Notes Management** — Save, search, and retrieve personal notes.
- **Task Management** — Create, list, and complete tasks.
- **gRPC Communication** — Separate application logic from the backend service.
- **SQLite Storage** — Persistent data storage.

## 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.12+ | Backend |
| FastAPI | API and WebSocket server |
| Google Gemini | AI responses and intent classification |
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
DATABASE_KEY=your_long_random_database_key
```

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

## 🧪 Testing

Run the test suite:

```bash
make test
```

Regenerate gRPC files after modifying the Protocol Buffer definitions:

```bash
make proto
```

## 📄 License

This project is currently available for personal and educational use.
