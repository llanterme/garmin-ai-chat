#!/usr/bin/env python3
"""Test script to validate the Tuesday query fix."""

import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock environment variables
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('PINECONE_API_KEY', 'test-key')
os.environ.setdefault('DATABASE_URL', 'mysql+aiomysql://root:Passw0rd1@localhost:3306/garmin_ai_chat')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-only-not-secure')
os.environ.setdefault('GARMIN_ENCRYPTION_KEY', '12345678901234567890123456789012')

def test_temporal_processor():
    """Test that our temporal processor correctly handles Tuesday queries."""
    print("=== Testing Tuesday Query Fix ===")
    
    from src.services.temporal_processor import TemporalQueryProcessor
    
    processor = TemporalQueryProcessor()
    
    # Test the exact query that was failing
    test_queries = [
        "Give me a detailed breakdown of my ride on Tuesday this week",
        "Tuesday this week",
        "this week's Tuesday", 
        "on Tuesday",
        "last Tuesday",
        "this Tuesday"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing Query: '{query}' ---")
        context = processor.process_query(query)
        
        print(f"  Enhanced query: {context.enhanced_query}")
        print(f"  Has temporal context: {context.has_temporal_context}")
        
        if context.temporal_filter:
            tf = context.temporal_filter
            print(f"  Temporal filter:")
            print(f"    - Specific date: {tf.specific_date}")
            print(f"    - Is exact date: {tf.is_exact_date}")
            print(f"    - Start date: {tf.start_date}")
            print(f"    - End date: {tf.end_date}")
            
            if tf.specific_date:
                print(f"    - Weekday: {tf.specific_date.strftime('%A')} ({tf.specific_date.strftime('%Y-%m-%d')})")
        else:
            print("  No temporal filter found")


def test_date_matching():
    """Test date matching utilities."""
    print("\n=== Testing Date Matching Utilities ===")
    
    from src.services.activity_ingestion import ActivityIngestionService
    
    service = ActivityIngestionService()
    
    # Create test activities with the actual IDs from your database
    test_activities = [
        {
            'garmin_activity_id': 20327400687,
            'activity_name': 'Zwift - Untitled workout on Tempus Fugit in Watopia',
            'start_time': '2025-09-09 07:02:41',
            'activity_type': 'virtual_ride'
        },
        {
            'garmin_activity_id': 20316526058,
            'activity_name': 'Zwift - New Workout on Itza Party in Watopia',
            'start_time': '2025-09-08 06:34:51',
            'activity_type': 'virtual_ride'
        }
    ]
    
    # Tuesday this week should be September 9th, 2025 (based on current data)
    tuesday_date = datetime(2025, 9, 9, 0, 0, 0)  # Tuesday
    print(f"Target Tuesday date: {tuesday_date.strftime('%A, %Y-%m-%d')}")
    
    # Test filtering
    filtered = service._filter_activities_by_exact_date(test_activities, tuesday_date)
    
    print(f"\nOriginal activities: {len(test_activities)}")
    print(f"Filtered activities: {len(filtered)}")
    
    for activity in filtered:
        print(f"  Matched Activity ID: {activity['garmin_activity_id']}")
        print(f"  Activity Name: {activity['activity_name']}")
        print(f"  Start Time: {activity['start_time']}")
        if '_date_match_debug' in activity:
            debug = activity['_date_match_debug']
            print(f"  Debug Info:")
            print(f"    - Activity date: {debug['activity_date']}")
            print(f"    - Target date: {debug['target_date']}")
            print(f"    - Matched field: {debug['matched_field']}")


if __name__ == "__main__":
    try:
        test_temporal_processor()
        # Skip date matching test due to Pinecone API key requirement
        print("\n=== Test Complete ===")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()