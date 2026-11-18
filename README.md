# Chat app

The orchestrator uses Gemini for chat and intent classification. It executes
note and task actions only through the gRPC service; the service owns SQLite
access.

Add your Google AI Studio API key to the project-root `.env` file:

```dotenv
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-3.5-flash-lite
```

`.env` is ignored by Git. Environment variables exported in the shell take
precedence over values in `.env`. Start the service and orchestrator in
separate terminals:

```bash
make service
make back
```