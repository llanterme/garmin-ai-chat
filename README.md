# Garmin AI Chat Backend

A FastAPI backend service for synchronizing and analyzing Garmin Connect activity data with AI-powered chat capabilities.

## Features

- **User Authentication**: JWT-based authentication with refresh tokens
- **Garmin Connect Integration**: Secure sync of activity data from Garmin Connect
- **Activity Management**: Store and retrieve detailed activity metrics
- **Database**: MySQL with SQLAlchemy ORM and async support
- **Data Security**: Encrypted storage of Garmin credentials
- **API Documentation**: Auto-generated OpenAPI/Swagger docs
- **Comprehensive Logging**: Structured logging with configurable levels
- **Error Handling**: Robust error handling with detailed responses

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **Python 3.11+**: Async/await support with type hints
- **MySQL**: Relational database with aiomysql driver
- **SQLAlchemy**: Async ORM with Alembic migrations
- **Pydantic**: Data validation and serialization
- **JWT**: Secure authentication tokens
- **python-garminconnect**: Official Garmin Connect API wrapper

## Quick Start

### Prerequisites

- Python 3.11 or higher
- uv (Python package manager)
- MySQL 8.0 or higher
- Docker (optional, for MySQL container)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd garmin-ai-chat
   ```

2. **Install dependencies**
   ```bash
   make install-dev
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start MySQL database**
   ```bash
   # Option 1: Using Docker
   make docker-up
   
   # Option 2: Use existing MySQL installation
   # Create database: garmin_ai_chat
   ```

5. **Run database migrations**
   ```bash
   make upgrade
   ```

6. **Start the development server**
   ```bash
   make run
   ```

The API will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=mysql+aiomysql://user:password@localhost:3306/garmin_ai_chat

# Authentication
SECRET_KEY=your-secret-key-at-least-32-characters-long
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Garmin Connect
GARMIN_ENCRYPTION_KEY=your-32-character-encryption-key

# Application
DEBUG=true
LOG_LEVEL=INFO
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/me` - Get current user info
- `POST /api/v1/auth/garmin-credentials` - Update Garmin credentials

### Activities
- `GET /api/v1/activities/` - List user activities (paginated)
- `GET /api/v1/activities/{id}` - Get specific activity
- `GET /api/v1/activities/types/` - Get activity types
- `DELETE /api/v1/activities/{id}` - Delete activity

### Synchronization
- `POST /api/v1/sync/activities` - Start activity sync from Garmin
- `GET /api/v1/sync/status/{sync_id}` - Get sync status
- `GET /api/v1/sync/history` - Get sync history
- `GET /api/v1/sync/stats` - Get sync statistics

### Health
- `GET /health/` - Health check
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe

## Usage Examples

### Register and Login

```bash
# Register new user
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword",
    "full_name": "John Doe"
  }'

# Login
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword"
  }'
```

### Set Garmin Credentials

```bash
curl -X POST "http://localhost:8000/api/v1/auth/garmin-credentials" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_garmin_username",
    "password": "your_garmin_password"
  }'
```

### Sync Activities

```bash
curl -X POST "http://localhost:8000/api/v1/sync/activities" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "days": 30,
    "force_resync": false
  }'
```

## Development

### Project Structure

```
garmin-ai-chat/
├── src/
│   ├── api/              # FastAPI route handlers
│   ├── core/             # Core configuration and utilities
│   ├── db/               # Database models and operations
│   ├── services/         # Business logic services
│   ├── schemas/          # Pydantic models
│   └── main.py           # FastAPI application
├── tests/                # Test suite
├── alembic/              # Database migrations
├── pyproject.toml        # Project configuration
├── Makefile              # Development commands
└── README.md
```

### Available Make Commands

```bash
make help           # Show available commands
make install        # Install production dependencies
make install-dev    # Install all dependencies
make test           # Run tests
make test-cov       # Run tests with coverage
make lint           # Run linting
make format         # Format code
make check          # Run all checks
make run            # Start development server
make clean          # Clean build artifacts
make migrate        # Generate new migration
make upgrade        # Apply database migrations
make downgrade      # Rollback migration
```

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test
uv run pytest tests/test_auth.py -v
```

### Database Migrations

```bash
# Generate new migration
make migrate MESSAGE="Add new field to users table"

# Apply migrations
make upgrade

# Rollback migration
make downgrade
```

## Data Models

### User
- Basic user information and authentication
- Encrypted Garmin Connect credentials
- Session management for Garmin API

### Activity
- Comprehensive activity metrics (distance, duration, calories, etc.)
- Performance data (heart rate, power, cadence)
- Elevation and location data
- Raw Garmin data storage for future analysis

### Sync History
- Track synchronization operations
- Success/failure status and error messages
- Performance metrics and statistics

## Security

- **Password Hashing**: bcrypt with salt
- **JWT Tokens**: Secure token-based authentication
- **Credential Encryption**: Garmin passwords encrypted at rest
- **Rate Limiting**: Built-in protection against abuse
- **CORS**: Configurable cross-origin resource sharing

## Deployment

### Production Configuration

1. **Set production environment variables**
   ```bash
   DEBUG=false
   ENVIRONMENT=production
   SECRET_KEY=your-production-secret-key
   ```

2. **Use production database**
   ```bash
   DATABASE_URL=mysql+aiomysql://user:pass@prod-host:3306/garmin_ai_chat
   ```

3. **Configure logging**
   ```bash
   LOG_LEVEL=INFO
   LOG_FILE=/var/log/garmin-ai-chat/app.log
   ```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync --no-dev

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `make check` to ensure code quality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs for error details