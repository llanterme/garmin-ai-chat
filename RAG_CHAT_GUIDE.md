# RAG Chat System - Implementation Guide

This guide explains the newly implemented RAG (Retrieval Augmented Generation) chat system that enables natural language conversations about Garmin fitness data.

## 🚀 Quick Start

### 1. Configure Environment Variables

Add these new variables to your `.env` file:

```bash
# AI/ML Configuration
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=garmin-fitness-activities
```

### 2. Install Dependencies

```bash
make install-dev  # Installs pinecone>=5.0.0, openai>=1.0.0, httpx>=0.28.1
```

### 3. Start the Server

```bash
make run  # Starts on http://localhost:8000
```

### 4. Process Your Activities

Before asking questions, ingest your Garmin activities into the vector database:

```bash
# POST /api/v1/chat/ingestion/start
curl -X POST "http://localhost:8000/api/v1/chat/ingestion/start" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"force_reingest": false, "batch_size": 10}'
```

### 5. Start Chatting

```bash
# POST /api/v1/chat/query
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was my longest run last week?",
    "search_limit": 15,
    "include_follow_ups": true
  }'
```

## 🏗️ Architecture Overview

### Multi-Vector Embedding Strategy

Each Garmin activity is stored as **4 different vector embeddings** to capture different aspects:

1. **Main Vector**: Comprehensive activity summary
2. **Metrics Vector**: Performance data focused (power, HR, pace, distance)
3. **Temporal Vector**: Date/time context with multiple formats
4. **Performance Vector**: Training analysis and intensity classification

### Core Components

```
src/services/
├── embedding.py          # Multi-vector embedding generation
├── vector_db.py          # Pinecone integration with hybrid search
├── temporal_processor.py # Natural language date/time parsing
├── llm.py               # GPT-4 conversational responses
├── activity_ingestion.py # Activity processing pipeline
└── conversation.py       # Chat session management

src/api/
└── chat.py              # REST API endpoints

src/schemas/
└── chat.py              # Request/response schemas
```

## 📊 Data Flow

```
Garmin Activity Data (MySQL)
    ↓
Multi-Vector Summary Generation
    ↓
Batch Embedding Generation (OpenAI)
    ↓
Vector Storage (Pinecone) with Metadata
    ↓
User Query → Temporal Processing
    ↓
Hybrid Vector Search + Metadata Filtering
    ↓
LLM Response Generation (GPT-4o-mini)
    ↓
Conversational Response + Follow-ups
```

## 🔍 Query Processing Pipeline

### 1. Temporal Processing
- **Input**: "What was my best run yesterday?"
- **Processing**: Extracts temporal context, activity type, performance indicators
- **Output**: Enhanced query with date filters and metadata

### 2. Multi-Vector Search
- Generates embeddings for all 4 vector types
- Applies metadata filters (date, activity type, metrics)
- Performs hybrid search with relevance scoring
- Applies recency and vector-type specific boosts

### 3. Context Enhancement
- Calculates user performance averages
- Adds performance comparisons
- Includes efficiency metrics and trends

### 4. LLM Generation
- Uses GPT-4o-mini for cost-effective responses
- Includes conversation history for context
- Generates 3 relevant follow-up questions

## 🎯 API Endpoints

### Chat Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/query` | POST | Process conversational query |
| `/api/v1/chat/conversations` | GET | List user conversations |
| `/api/v1/chat/conversations/{id}` | GET | Get conversation history |
| `/api/v1/chat/conversations/{id}` | DELETE | Clear conversation |
| `/api/v1/chat/suggestions` | GET | Get suggested questions |

### Ingestion Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/ingestion/start` | POST | Start activity ingestion |
| `/api/v1/chat/ingestion/status` | GET | Check ingestion status |
| `/api/v1/chat/stats` | GET | Get vectorization statistics |

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/health` | GET | Service health check |

## 💬 Example Conversations

### Temporal Queries
```
User: "What did I do yesterday morning?"
AI: "Yesterday morning you completed a 5.2km run in Sandton 🏃‍♂️..."

User: "Show me my cycling activities last week"
AI: "Last week you had 3 cycling sessions totaling 147km 🚴‍♂️..."
```

### Performance Analysis
```
User: "How's my running pace improving?"
AI: "Your average pace has improved from 5:45/km to 5:12/km over the past month 📈..."

User: "Compare my power output this month vs last month"
AI: "Your average cycling power increased from 245W to 267W (+9%) 💪..."
```

### Aggregation Queries
```
User: "How many calories did I burn this week?"
AI: "This week you burned 2,847 calories across 6 activities 🔥..."

User: "What's my total distance for all activities this month?"
AI: "Total distance this month: 234.5km (Running: 145km, Cycling: 89.5km) 📊..."
```

## 🔧 Configuration Options

### Vector Database Settings
```python
# Pinecone Configuration
PINECONE_INDEX_NAME = "garmin-fitness-activities"
PINECONE_ENVIRONMENT = "us-east-1"  # AWS region
```

### Embedding Settings
```python
# OpenAI Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions
LLM_MODEL = "gpt-4o-mini"  # Cost-effective chat model
```

### Search Parameters
```python
# Default search limits by query type
SEARCH_LIMITS = {
    "temporal": 20,      # Recent activities
    "aggregation": 100,  # Many activities for calculations
    "comparison": 50,    # Good sample for comparisons
    "analysis": 50,      # Substantial data for trends
    "general": 15        # Default limit
}
```

## 🧪 Testing

### Basic Functionality Test
```bash
python test_chat.py
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### Health Check
```bash
curl http://localhost:8000/api/v1/chat/health
```

## 📈 Performance Optimizations

### Embedding Generation
- **Batch Processing**: 4 summaries per API call
- **Concurrent Processing**: 10 activities in parallel
- **Rate Limiting**: 0.1s delay between activities

### Vector Search
- **Smart Limits**: Query-type based search limits
- **Vector Deduplication**: Single result per activity
- **Score Boosting**: Recency and vector-type boosts
- **Metadata Filtering**: Efficient Pinecone filters

### LLM Responses
- **Context Window**: Last 10 messages for continuity
- **Token Optimization**: Max 1000 tokens per response
- **Fallback Handling**: Graceful degradation on failures

## 🔐 Security Features

- **User Isolation**: Separate Pinecone namespaces per user
- **JWT Authentication**: Required for all endpoints
- **Input Validation**: Pydantic schemas for all requests
- **API Key Protection**: Environment variable configuration

## 🚨 Troubleshooting

### Common Issues

1. **"No activities found"**
   - Run activity ingestion first: `POST /api/v1/chat/ingestion/start`
   - Check ingestion status: `GET /api/v1/chat/ingestion/status`

2. **"OpenAI API Error"**
   - Verify `OPENAI_API_KEY` in `.env`
   - Check API quota and billing

3. **"Pinecone Connection Error"**
   - Verify `PINECONE_API_KEY` in `.env`
   - Ensure index exists or will be auto-created

4. **"Slow Response Times"**
   - Reduce `search_limit` parameter
   - Check Pinecone region configuration

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
make run
```

## 🚀 Next Steps

1. **Enhanced Features**:
   - Voice input/output
   - Activity recommendations
   - Training plan generation
   - Multi-user comparisons

2. **Performance**:
   - Redis caching for conversations
   - Database persistence for chat history
   - Background embedding processing

3. **Analytics**:
   - Query performance metrics
   - User engagement tracking
   - Response quality monitoring

## 📚 Technical Details

### Vector Storage Structure
```
Pinecone Index: garmin-fitness-activities
├── Namespace: user_{user_id}
│   ├── Vector ID: {activity_id}_main
│   ├── Vector ID: {activity_id}_metrics  
│   ├── Vector ID: {activity_id}_temporal
│   └── Vector ID: {activity_id}_performance
```

### Metadata Schema
```json
{
  "user_id": "string",
  "activity_id": "string", 
  "main_activity_id": "string",
  "vector_type": "main|metrics|temporal|performance",
  "data_source": "garmin",
  "date": "2025-08-27",
  "timestamp": 1693134000,
  "activity_type": "running",
  "distance_km": 5.2,
  "duration_minutes": 30.5,
  "average_speed_kmh": 10.2,
  "average_heart_rate": 150,
  "has_power_data": true,
  "efficiency_score": 0.14
}
```

---

🎉 **Congratulations!** You now have a fully functional RAG chat system for natural language conversations about Garmin fitness data. The system combines semantic search with LLM generation to provide accurate, contextual responses about training activities, performance trends, and fitness analytics.