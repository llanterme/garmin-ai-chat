# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Garmin AI Chat Backend is a comprehensive FastAPI application that provides secure integration with Garmin Connect for activity synchronization and management. The system allows users to authenticate with their Garmin credentials, sync activities by date range, and store detailed activity metrics in a MySQL database.

## Architecture

- **Framework**: FastAPI with async support
- **Database**: MySQL with SQLAlchemy async ORM
- **Authentication**: JWT tokens with encrypted credential storage
- **Garmin Integration**: python-garminconnect library
- **Activity Storage**: Hybrid approach with structured columns + JSON for detailed metrics

## Development Commands

```bash
# Setup
make install-dev          # Install dependencies
mysql -u root -p          # Create database manually:
CREATE DATABASE garmin_ai_chat;

# Database
make upgrade              # Run Alembic migrations
make downgrade            # Rollback last migration
make revision             # Create new migration

# Development
make run                  # Start development server
make test                 # Run test suite
make lint                 # Run code linting
make format               # Format code

# Production
make install              # Install production dependencies only
```

## Configuration

Environment variables in `.env`:
- `DATABASE_URL`: MySQL connection string
- `SECRET_KEY`: JWT secret (min 32 chars)
- `GARMIN_ENCRYPTION_KEY`: Fernet encryption key (exactly 32 chars)
- `DEBUG`: Enable debug mode

## API Endpoints

**Authentication:**
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - Login with JWT
- `POST /api/v1/auth/garmin-credentials` - Set Garmin credentials

**Activities:**
- `GET /api/v1/activities/` - List activities (paginated, filtered)
- `GET /api/v1/activities/{id}` - Get activity details
- `GET /api/v1/activities/types/` - Get activity types

**Synchronization:**
- `POST /api/v1/sync/activities` - Start activity sync
- `GET /api/v1/sync/status/{sync_id}` - Monitor sync progress
- `GET /api/v1/sync/history` - View sync history

**Health:**
- `GET /health/` - Application health check
- `GET /docs` - Interactive API documentation

## Database Schema

**Key Tables:**
- `users` - User accounts and encrypted Garmin credentials
- `activities` - Activity data with structured metrics + JSON payloads
- `sync_history` - Synchronization tracking and status

**Activity Metrics Stored:**
- Distance, duration, elevation gain
- Heart rate (avg/max), power (avg/max), speed, cadence
- Activity-specific data (swimming strokes, cycling power zones, etc.)
- Full raw response from Garmin Connect API

## Garmin Integration

The system uses the `python-garminconnect` library to:
1. Authenticate with Garmin Connect using user credentials
2. Fetch activities by date range (configurable: 10, 20, 100+ days)
3. Retrieve detailed activity metrics and performance data
4. Handle different activity types (running, cycling, swimming, etc.)

## Security Features

- Encrypted storage of Garmin credentials using Fernet encryption
- JWT authentication with refresh tokens
- Password hashing with bcrypt
- Input validation with Pydantic
- SQL injection protection via SQLAlchemy ORM

## Development Setup

1. Ensure MySQL is running locally
2. Copy `.env.example` to `.env` and configure
3. Run `make install-dev` to install dependencies
4. Run `make upgrade` to create database tables
5. Start with `make run` - server runs on http://localhost:8000

## Testing

The server includes comprehensive health checks and can be tested via:
- Interactive docs at `/docs`
- Health endpoint at `/health/`
- All endpoints support proper error handling and validation