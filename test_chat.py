#!/usr/bin/env python3
"""Simple test script for the chat functionality."""

import os
import sys
import asyncio
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock environment variables for testing
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('PINECONE_API_KEY', 'test-key')
os.environ.setdefault('DATABASE_URL', 'mysql+aiomysql://root:Passw0rd1@localhost:3306/garmin_ai_chat')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-only-not-secure')
os.environ.setdefault('GARMIN_ENCRYPTION_KEY', '12345678901234567890123456789012')  # Exactly 32 chars

async def test_temporal_processor():
    """Test temporal query processing."""
    print("=== Testing Temporal Query Processor ===")
    
    from services.temporal_processor import TemporalQueryProcessor
    
    processor = TemporalQueryProcessor()
    
    test_queries = [
        "What was my best run yesterday?",
        "Show me my cycling activities last week",
        "How many calories did I burn this month?",
        "What's my average pace for runs over 10km?",
        "Compare my performance today vs last month"
    ]
    
    for query in test_queries:
        context = processor.process_query(query)
        print(f"\nQuery: {query}")
        print(f"  Type: {context.query_type}")
        print(f"  Temporal: {context.has_temporal_context}")
        print(f"  Activity filter: {context.activity_type_filter}")
        print(f"  Metric filters: {len(context.metric_filters)}")

async def test_embedding_service():
    """Test embedding service (without real API calls)."""
    print("\n=== Testing Embedding Service (Structure) ===")
    
    try:
        from services.embedding import EmbeddingService
        
        # Test activity data structure
        sample_activity = {
            "garmin_activity_id": "12345",
            "activity_type": "running",
            "activity_name": "Morning Run",
            "start_time": datetime.now() - timedelta(days=1),
            "distance": 5000,  # meters
            "duration": 1800,  # seconds (30 minutes)
            "average_speed": 2.78,  # m/s (10 km/h)
            "average_heart_rate": 150,
            "calories": 300
        }
        
        service = EmbeddingService()
        
        # Test summary generation (doesn't require API)
        summaries = service.create_multi_vector_embeddings(sample_activity)
        
        print("✓ Successfully generated summaries:")
        for vector_type, summary in summaries.items():
            print(f"  {vector_type}: {summary[:80]}...")
            
    except ImportError as e:
        print(f"✗ Import failed (expected without OpenAI key): {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

async def test_chat_schemas():
    """Test chat request/response schemas."""
    print("\n=== Testing Chat Schemas ===")
    
    from schemas.chat import ChatRequest, ChatResponse, ActivitySummary
    
    # Test request schema
    request = ChatRequest(
        query="What was my best run this week?",
        search_limit=10,
        include_follow_ups=True
    )
    print(f"✓ Chat request: {request.query}")
    
    # Test response schema
    activity = ActivitySummary(
        garmin_activity_id="12345",
        activity_type="running",
        distance_km=5.0,
        duration_minutes=30.0,
        relevance_score=0.95
    )
    
    response = ChatResponse(
        response="Your best run this week was 5km in 30 minutes!",
        relevant_activities=[activity],
        follow_up_questions=["How does this compare to last week?"],
        conversation_id="test-conv-123",
        timestamp=datetime.now().isoformat(),
        activity_count=1
    )
    
    print(f"✓ Chat response generated with {response.activity_count} activities")

async def main():
    """Run all tests."""
    print("🚀 Testing Garmin AI Chat System\n")
    
    try:
        await test_temporal_processor()
        await test_embedding_service()
        await test_chat_schemas()
        
        print("\n✅ Basic functionality tests completed!")
        print("\n💡 Next steps:")
        print("1. Configure OpenAI API key in .env")
        print("2. Configure Pinecone API key in .env")
        print("3. Run: make run")
        print("4. Test endpoints at http://localhost:8000/docs")
        print("5. Start activity ingestion: POST /api/v1/chat/ingestion/start")
        print("6. Ask questions: POST /api/v1/chat/query")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())