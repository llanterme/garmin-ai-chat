"""Chat API endpoints for conversational fitness queries."""

import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.dependencies import get_current_user
from ..core.logging import get_logger
from ..db.base import get_db
from ..db.models import User
from ..schemas.chat import (
    ActivityStats,
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    ConversationSummary,
    EmbeddingsStatus,
    ErrorResponse,
    HealthCheckResponse,
    IngestionRequest,
    IngestionStatus,
    SuggestionRequest,
    SuggestionsResponse,
    UserConversationsResponse,
)
from ..services.activity_ingestion import ActivityIngestionService
from ..services.conversation import ConversationService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Process a conversational query about fitness activities."""
    try:
        logger.info(f"Processing chat query for user {current_user.id}: {request.query[:50]}...")
        
        # Initialize conversation service
        conversation_service = ConversationService(db)
        
        # Process the query
        result = await conversation_service.process_query(
            user_id=current_user.id,
            query=request.query,
            conversation_id=request.conversation_id,
            search_limit=request.search_limit,
            include_follow_ups=request.include_follow_ups
        )
        
        # Convert to response schema
        return ChatResponse(
            response=result["response"],
            relevant_activities=result["relevant_activities"],
            follow_up_questions=result["follow_up_questions"],
            conversation_id=result["conversation_id"],
            timestamp=result["timestamp"],
            activity_count=result["activity_count"],
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Chat query failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat query: {str(e)}"
        )


@router.get("/conversations", response_model=UserConversationsResponse)
async def get_user_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all conversations for the current user."""
    try:
        conversation_service = ConversationService(db)
        conversations = await conversation_service.get_user_conversations(current_user.id)
        
        return UserConversationsResponse(
            conversations=[
                ConversationSummary(**conv) for conv in conversations
            ],
            total_count=len(conversations)
        )
        
    except Exception as e:
        logger.error(f"Failed to get conversations for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get conversation history by ID."""
    try:
        conversation_service = ConversationService(db)
        history = await conversation_service.get_conversation_history(conversation_id)
        
        if not history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify user ownership
        if history["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied to this conversation")
        
        return ConversationHistory(**history)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Clear a conversation by ID."""
    try:
        conversation_service = ConversationService(db)
        
        # Verify ownership before deletion
        history = await conversation_service.get_conversation_history(conversation_id)
        if history and history["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied to this conversation")
        
        success = await conversation_service.clear_conversation(conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": "Conversation cleared successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.post("/ingestion/start", response_model=IngestionStatus)
async def start_activity_ingestion(
    request: IngestionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Start ingesting user's activities into vector database."""
    try:
        logger.info(f"Starting activity ingestion for user {current_user.id}")
        
        ingestion_service = ActivityIngestionService(db)
        
        result = await ingestion_service.ingest_user_activities(
            user_id=current_user.id,
            session=db,
            batch_size=request.batch_size,
            force_reingest=request.force_reingest
        )
        
        return IngestionStatus(**result)
        
    except Exception as e:
        logger.error(f"Activity ingestion failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start activity ingestion: {str(e)}"
        )


@router.get("/ingestion/status", response_model=EmbeddingsStatus)
async def get_ingestion_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get status of embeddings vs database activities."""
    try:
        ingestion_service = ActivityIngestionService(db)
        
        status = await ingestion_service.get_activity_embeddings_status(
            user_id=current_user.id,
            session=db
        )
        
        return EmbeddingsStatus(**status)
        
    except Exception as e:
        logger.error(f"Failed to get ingestion status for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ingestion status: {str(e)}"
        )


@router.get("/stats", response_model=ActivityStats)
async def get_activity_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get statistics about user's vectorized activities."""
    try:
        ingestion_service = ActivityIngestionService(db)
        
        stats = await ingestion_service.get_user_activity_stats(current_user.id)
        
        return ActivityStats(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get activity stats for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get activity statistics: {str(e)}"
        )


@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(
    recent_query: str = Query(None, description="Recent user query for context"),
    activity_type: str = Query(None, description="Focus on specific activity type"),
    current_user: User = Depends(get_current_user)
):
    """Get suggested questions for the user."""
    try:
        # Generate contextual suggestions based on activity type and recent query
        suggestions = []
        category = "general"
        
        if activity_type:
            category = activity_type.lower()
            if activity_type.lower() in ["running", "treadmill_running"]:
                suggestions = [
                    "What's my average running pace this month?",
                    "How many kilometers did I run last week?",
                    "What's my best 5K time recently?",
                    "Show me my heart rate trends during runs",
                    "How consistent has my running been?"
                ]
            elif activity_type.lower() in ["cycling", "virtual_ride"]:
                suggestions = [
                    "What's my average power output this month?",
                    "How far did I cycle last week?",
                    "What's my longest ride recently?",
                    "Show me my cycling speed improvements",
                    "How's my cycling training load?"
                ]
            elif activity_type.lower() == "swimming":
                suggestions = [
                    "How many laps did I swim this week?",
                    "What's my average swimming pace?",
                    "Show me my swimming distance trends",
                    "How often do I swim per week?",
                    "What's my longest swim session?"
                ]
        
        if not suggestions:
            # General suggestions
            suggestions = [
                "What activities did I do yesterday?",
                "Show me my workout summary for this week",
                "What's my most frequent activity type?",
                "How many calories did I burn last week?",
                "What's my average heart rate during workouts?",
                "Show me my longest activities this month",
                "How has my fitness improved recently?",
                "What's my weekly training volume?"
            ]
        
        return SuggestionsResponse(
            suggestions=suggestions[:5],  # Return top 5 suggestions
            category=category
        )
        
    except Exception as e:
        logger.error(f"Failed to generate suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check for chat services."""
    try:
        services = {}
        overall_status = "healthy"
        
        # Check embedding service
        try:
            from ..services.embedding import EmbeddingService
            embedding_service = EmbeddingService()
            services["embedding_service"] = "healthy"
        except Exception as e:
            services["embedding_service"] = f"unhealthy: {str(e)}"
            overall_status = "degraded"
        
        # Check vector database
        try:
            from ..services.vector_db import VectorDBService
            vector_db = VectorDBService()
            services["vector_database"] = "healthy"
        except Exception as e:
            services["vector_database"] = f"unhealthy: {str(e)}"
            overall_status = "degraded"
        
        # Check LLM service
        try:
            from ..services.llm import LLMService
            llm_service = LLMService()
            services["llm_service"] = "healthy"
        except Exception as e:
            services["llm_service"] = f"unhealthy: {str(e)}"
            overall_status = "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            services=services,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            services={"error": str(e)},
            timestamp=datetime.now().isoformat()
        )