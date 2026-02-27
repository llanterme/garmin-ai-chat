# CLAUDE.md

## Project Overview

FastAPI backend that syncs activities from Garmin Connect and provides a conversational AI chat interface for querying fitness data. Users authenticate, sync their Garmin activities into MySQL + Pinecone, then ask natural language questions about their training.

## Architecture

- **Framework**: FastAPI (async)
- **Database**: MySQL via SQLAlchemy async ORM + Alembic migrations
- **Vector DB**: Pinecone — stores multi-vector embeddings (main, metrics, temporal, performance) per activity
- **Embeddings**: OpenAI `text-embedding-ada-002` via `src/services/embedding.py`
- **LLM**: OpenAI chat completions via `src/services/llm.py`
- **Auth**: JWT tokens, Garmin credentials encrypted with Fernet
- **Garmin**: `python-garminconnect` library

## Key Data Flow

1. **Sync**: Garmin API → `garmin.py` parses raw activities → stored in MySQL (`activities` table) + Pinecone (multi-vector embeddings via `vector_db.py`)
2. **Chat query**: User query → `temporal_processor.py` extracts filters (date ranges, activity types, metrics) → `embedding.py` generates query embeddings → `vector_db.py` hybrid search in Pinecone → `llm.py` formats context + generates response
3. **Activity types flow**: Garmin `typeKey` (e.g., `running`, `virtual_ride`) → stored verbatim in MySQL → lowercased in Pinecone metadata → matched via regex patterns + alias expansion in `temporal_processor.py`

## Development Commands

```bash
make install-dev    # Install dependencies (uses uv)
make run            # Dev server on :8000
make test           # Run pytest
make lint           # ruff + mypy
make format         # black + ruff format
make upgrade        # Run Alembic migrations
make downgrade      # Rollback last migration
```

## Configuration

Environment variables in `.env`:
- `DATABASE_URL` — MySQL connection string
- `SECRET_KEY` — JWT secret (min 32 chars)
- `GARMIN_ENCRYPTION_KEY` — Fernet key (exactly 32 chars)
- `OPENAI_API_KEY` — For embeddings and chat completions
- `PINECONE_API_KEY` — Vector database
- `PINECONE_INDEX_NAME` — Default: `garmin-fitness-activities`
- `DEBUG` — Enable debug mode

## API Endpoints

- `POST /api/v1/auth/register|login|garmin-credentials` — Auth
- `GET /api/v1/activities/` — List activities (paginated, filtered)
- `POST /api/v1/sync/activities` — Sync from Garmin
- `POST /api/v1/chat/query` — Conversational fitness query
- `GET /api/v1/chat/conversations` — List user conversations
- `GET /api/v1/chat/stats` — Vectorized activity stats
- `GET /health/` — Health check

## Critical Services

| Service | File | Role |
|---------|------|------|
| `TemporalQueryProcessor` | `src/services/temporal_processor.py` | Parses NL queries → temporal filters, activity types, metric filters |
| `VectorDBService` | `src/services/vector_db.py` | Pinecone CRUD, multi-vector hybrid search with recency boost |
| `ActivityIngestionService` | `src/services/activity_ingestion.py` | Orchestrates search: query processing → embedding → vector search → filtering |
| `LLMService` | `src/services/llm.py` | Formats activity context for LLM, generates chat responses |
| `ConversationService` | `src/services/conversation.py` | Manages conversation state, coordinates ingestion + LLM |
| `EmbeddingService` | `src/services/embedding.py` | OpenAI embedding generation |
| `GarminService` | `src/services/garmin.py` | Garmin Connect API integration |

## Temporal Processor Details

`temporal_processor.py` is the NL query parser. Key design decisions:
- **Pattern order matters**: patterns are checked in insertion order (first match wins). Broader patterns (week/month) come before narrow ones (time-of-day)
- **Tuple returns = date ranges**: handler lambdas returning `(start, end)` tuples are treated as ranges; single datetimes become specific_date filters
- **"last N activities" guard**: queries like "last 3 runs" bypass temporal filtering entirely so "today" in "suggest a workout today" doesn't filter out historical data
- **Activity type aliases return lists**: `activity_type_filter` is `Optional[Union[str, List[str]]]` — aliases like "workouts" expand to `["strength_training", "running", "cycling"]`
- **Pinecone filter expansion**: `create_pinecone_filter` maps extracted types to all stored variations via `activity_variations` dict using `$in` queries

## Database Schema

- `users` — Accounts + encrypted Garmin credentials
- `activities` — Structured metrics + full JSON payload from Garmin
- `sync_history` — Sync job tracking

Activity types are free-form strings (Garmin's `typeKey` values like `running`, `virtual_ride`, `trail_running`). No enum — stored verbatim in MySQL, lowercased in Pinecone.
