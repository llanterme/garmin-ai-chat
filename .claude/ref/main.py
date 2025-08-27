"""
FastAPI application for Athlete IQ backend.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Path, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from .services.activity_ingestion import ActivityIngestionService, IngestionStatus
from .services.conversation_service import ConversationService
from .services.mysql_service import mysql_service
from .services.training_plan_service import TrainingPlanService
from .services.training_plan_job_service import TrainingPlanJobService
from .auth.garmin_oauth import GarminOAuth
from .auth.webhook_verification import webhook_verifier
from .services.webhook_service import webhook_service
from .middleware.rate_limit import webhook_rate_limit_middleware
from .models.race import RaceCreateRequest, RaceUpdateRequest, RaceResponse, RaceListResponse, RaceEntry
from .models.training_plan import TrainingPlanRequest, TrainingPlanResponse, TrainingPlanSummary
from .models.webhook import GarminWebhookPayload, WebhookHealthStatus, WebhookMetrics
from .models.training_plan_job import (
    JobStatus, TrainingPlanJobResponse, TrainingPlanJobStatusResponse,
    TrainingPlanJobResult, TrainingPlanJobListResponse
)

# Load environment variables
load_dotenv()

# Configure logging - keep DEBUG for troubleshooting
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global services
ingestion_service = ActivityIngestionService()
conversation_service = ConversationService()
training_plan_service = TrainingPlanService()
training_plan_job_service = TrainingPlanJobService()
garmin_oauth = GarminOAuth()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Athlete IQ backend")
    
    # Initialize MySQL service
    try:
        await mysql_service.initialize()
        logger.info("MySQL service initialized successfully")
        
        # Reset any orphaned training plan jobs
        reset_count = await training_plan_job_service.reset_orphaned_jobs()
        if reset_count > 0:
            logger.info(f"Reset {reset_count} orphaned training plan jobs on startup")
        
        # Webhook processing (simplified - no queue workers needed)
        
    except Exception as e:
        logger.error(f"Failed to initialize MySQL service: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Application shutdown complete")
    
    try:
        await mysql_service.close()
        logger.info("MySQL service closed successfully")
    except Exception as e:
        logger.error(f"Error closing MySQL service: {str(e)}")
    
    logger.info("Shutting down Athlete IQ backend")


app = FastAPI(
    title="Athlete IQ API",
    description="Backend API for Strava Fitness AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS - Disabled when using API Gateway
# API Gateway handles CORS, so we only enable this for local development
if os.getenv("DISABLE_CORS_MIDDLEWARE", "false").lower() != "true":
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3002").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Pydantic models
class IngestionRequest(BaseModel):
    user_id: str
    access_token: str
    refresh_token: str
    full_sync: bool = False
    sync_days: Optional[int] = None  # Number of days to sync, defaults to env var if not provided
    
    @field_validator('sync_days')
    @classmethod
    def validate_sync_days(cls, v):
        if v is not None and (v < 1 or v > 1095):  # Max 3 years
            raise ValueError('sync_days must be between 1 and 1095 days')
        return v


class SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 10
    filter_metadata: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    activity_id: str
    summary: str
    score: float
    activity_type: str
    start_date: str
    distance: float
    moving_time: int
    metadata: Dict[str, Any]


class UserStatsResponse(BaseModel):
    total_activities: int
    ingestion_status: Optional[IngestionStatus]


class ChatRequest(BaseModel):
    user_id: str
    query: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    relevant_activities: List[Dict[str, Any]]
    follow_up_questions: List[str]
    conversation_id: str
    timestamp: str


class InsightsRequest(BaseModel):
    user_id: str
    time_period: str = "month"


# Garmin OAuth models
class GarminAuthRequest(BaseModel):
    user_id: Optional[str] = None

class GarminAuthResponse(BaseModel):
    authorization_url: str
    state: str
    expires_at: str


class GarminCallbackRequest(BaseModel):
    code: str
    state: str
    user_id: Optional[str] = None


class GarminAuthStatusResponse(BaseModel):
    user_id: str
    provider: str
    authenticated: bool
    garmin_user_id: Optional[str] = None
    token_expires_at: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Athlete IQ API is running"}


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle OPTIONS requests for all paths."""
    return {"message": "OK"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "athlete-iq-backend"
    }


# ===== GARMIN OAUTH ENDPOINTS =====

# Security: Rate limiting storage (in-memory for demo, use Redis in production)
_rate_limit_store = {}

def check_rate_limit(client_ip: str, endpoint: str, max_requests: int = 10, window_minutes: int = 5) -> bool:
    """Simple rate limiting check."""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    key = f"{client_ip}:{endpoint}"
    
    if key not in _rate_limit_store:
        _rate_limit_store[key] = []
    
    # Remove old requests outside the window
    window_start = now - timedelta(minutes=window_minutes)
    _rate_limit_store[key] = [req_time for req_time in _rate_limit_store[key] if req_time > window_start]
    
    # Check if under limit
    if len(_rate_limit_store[key]) >= max_requests:
        return False
    
    # Add current request
    _rate_limit_store[key].append(now)
    return True

def log_auth_event(event_type: str, user_id: Optional[str], success: bool, details: Optional[str] = None):
    """Log authentication events for audit purposes."""
    logger.info(f"AUTH_AUDIT: {event_type} | User: {user_id or 'unknown'} | Success: {success} | Details: {details or 'none'}")

@app.post("/auth/garmin/authorize", response_model=GarminAuthResponse)
async def garmin_authorize(auth_request: GarminAuthRequest, request: Request):
    """
    Initiate Garmin OAuth2 PKCE authorization flow.
    
    Returns authorization URL and state for frontend to redirect user to Garmin.
    """
    # Security: Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, "garmin_authorize", max_requests=5, window_minutes=5):
        log_auth_event("RATE_LIMIT_EXCEEDED", None, False, f"IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Too many authorization requests. Please try again later.")
    
    try:
        auth_url, state, code_verifier, code_challenge = garmin_oauth.get_authorization_url()
        
        # Store PKCE parameters in database with expiration
        from datetime import datetime, timedelta
        expires_at = datetime.now() + timedelta(minutes=10)
        
        # Store state and code_verifier for later verification
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            
            # If user_id is provided, ensure the user exists
            if auth_request.user_id:
                # Check if user exists, create if not
                user_check_query = text("SELECT id FROM users WHERE id = :user_id")
                user_result = await session.execute(user_check_query, {'user_id': auth_request.user_id})
                
                if not user_result.fetchone():
                    # Create the user record
                    create_user_query = text("""
                        INSERT INTO users (id, activity_provider, created_at, updated_at)
                        VALUES (:user_id, 'garmin', NOW(), NOW())
                    """)
                    await session.execute(create_user_query, {'user_id': auth_request.user_id})
            
            # Now create the OAuth state
            query = text("""
                INSERT INTO garmin_oauth_state (state, code_verifier, code_challenge, user_id, created_at, expires_at)
                VALUES (:state, :code_verifier, :code_challenge, :user_id, NOW(), :expires_at)
            """)
            
            await session.execute(query, {
                'state': state,
                'code_verifier': code_verifier,
                'code_challenge': code_challenge,
                'user_id': auth_request.user_id,
                'expires_at': expires_at
            })
            await session.commit()
        
        logger.info(f"Created Garmin OAuth state {state} with PKCE challenge")
        log_auth_event("AUTHORIZATION_INITIATED", auth_request.user_id, True, f"State: {state[:8]}...")
        
        return GarminAuthResponse(
            authorization_url=auth_url,
            state=state,
            expires_at=expires_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to create Garmin authorization URL: {str(e)}")
        log_auth_event("AUTHORIZATION_FAILED", None, False, str(e))
        raise HTTPException(status_code=500, detail=f"Authorization failed: {str(e)}")


@app.get("/auth/garmin/callback")
async def garmin_callback_get(request: Request):
    """
    Handle Garmin OAuth2 callback via GET request (standard OAuth redirect).
    
    Extracts parameters from query string and processes the callback.
    """
    # Extract parameters from query string
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    
    if error:
        log_auth_event("CALLBACK_ERROR", None, False, f"OAuth error: {error}")
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    if not code or not state:
        log_auth_event("CALLBACK_MISSING_PARAMS", None, False, "Missing code or state parameter")
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
    # Create callback request object and call main handler
    callback_request = GarminCallbackRequest(code=code, state=state)
    return await garmin_callback_handler(callback_request, request)


@app.post("/auth/garmin/callback")
async def garmin_callback_post(callback_request: GarminCallbackRequest, request: Request):
    """
    Handle Garmin OAuth2 callback via POST request.
    
    Accepts callback data in request body.
    """
    return await garmin_callback_handler(callback_request, request)


async def garmin_callback_handler(callback_request: GarminCallbackRequest, request: Request):
    """
    Handle Garmin OAuth2 callback and exchange code for tokens.
    
    Validates state and PKCE parameters, then stores tokens.
    """
    # Security: Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, "garmin_callback", max_requests=10, window_minutes=5):
        log_auth_event("RATE_LIMIT_EXCEEDED", getattr(callback_request, 'user_id', None), False, f"IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Too many callback requests. Please try again later.")
    
    try:
        # Retrieve and validate PKCE parameters from database
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            query = text("""
                SELECT code_verifier, user_id, expires_at 
                FROM garmin_oauth_state 
                WHERE state = :state AND expires_at > NOW()
            """)
            
            result = await session.execute(query, {'state': callback_request.state})
            row = result.fetchone()
            
            if not row:
                log_auth_event("INVALID_STATE", callback_request.user_id, False, f"State: {callback_request.state[:8]}...")
                raise HTTPException(status_code=400, detail="Invalid or expired state parameter")
            
            code_verifier = row.code_verifier
            stored_user_id = row.user_id
            
            # Clean up used state
            delete_query = text("DELETE FROM garmin_oauth_state WHERE state = :state")
            await session.execute(delete_query, {'state': callback_request.state})
            await session.commit()
        
        # Exchange authorization code for tokens
        tokens = await garmin_oauth.exchange_code_for_token(callback_request.code, code_verifier)
        
        # CRITICAL: Get Garmin user ID from profile
        try:
            profile_data = await garmin_oauth.get_user_profile(tokens.access_token)
            garmin_user_id = profile_data.get('userId') or profile_data.get('userAccessToken')
            if not garmin_user_id:
                logger.warning(f"Could not extract Garmin user ID from profile: {profile_data}")
                # Try alternative field names that Garmin might use
                garmin_user_id = profile_data.get('id') or profile_data.get('user_id')
            
            logger.info(f"Retrieved Garmin user ID: {garmin_user_id}")
        except Exception as e:
            logger.error(f"Failed to get Garmin user profile: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve Garmin user profile")
        
        # Use provided user_id or stored user_id, or generate a default for testing
        user_id = callback_request.user_id or stored_user_id
        if not user_id:
            # For testing purposes, generate a default user_id based on the tokens
            # In production, this should require proper user authentication
            import uuid
            user_id = f"garmin_user_{str(uuid.uuid4())[:8]}"
            logger.warning(f"No user_id provided, generated temporary ID: {user_id}")
        
        # Store tokens in users table
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            
            # Calculate token expiration timestamp if provided
            token_expires_at = None
            if tokens.expires_at:
                from datetime import datetime, timedelta
                token_expires_at = datetime.now() + timedelta(seconds=tokens.expires_at)
            
            update_query = text("""
                UPDATE users SET 
                    garmin_access_token = :access_token,
                    garmin_refresh_token = :refresh_token,
                    garmin_token_expires_at = :expires_at,
                    garmin_user_id = :garmin_user_id,
                    activity_provider = 'garmin'
                WHERE id = :user_id
            """)
            
            result = await session.execute(update_query, {
                'user_id': user_id,
                'access_token': tokens.access_token,
                'refresh_token': tokens.refresh_token,
                'expires_at': token_expires_at,
                'garmin_user_id': garmin_user_id
            })
            
            if result.rowcount == 0:
                # User doesn't exist, create new user record
                insert_query = text("""
                    INSERT INTO users (id, garmin_access_token, garmin_refresh_token, 
                                     garmin_token_expires_at, garmin_user_id, activity_provider)
                    VALUES (:user_id, :access_token, :refresh_token, :expires_at, :garmin_user_id, 'garmin')
                """)
                
                await session.execute(insert_query, {
                    'user_id': user_id,
                    'access_token': tokens.access_token,
                    'refresh_token': tokens.refresh_token,
                    'expires_at': token_expires_at,
                    'garmin_user_id': garmin_user_id
                })
            
            await session.commit()
        
        logger.info(f"Successfully stored Garmin tokens for user {user_id}")
        log_auth_event("TOKEN_EXCHANGE_SUCCESS", user_id, True, "Garmin authentication completed")
        
        return {
            "success": True,
            "user_id": user_id,
            "message": "Garmin authentication successful",
            "provider": "garmin",
            "token_expires_at": token_expires_at.isoformat() if token_expires_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process Garmin callback: {str(e)}")
        log_auth_event("CALLBACK_FAILED", callback_request.user_id, False, str(e))
        raise HTTPException(status_code=500, detail=f"Callback processing failed: {str(e)}")


@app.post("/auth/garmin/refresh")
async def garmin_refresh_token(user_id: str):
    """
    Refresh Garmin access token for a user.
    Note: Garmin may not support refresh tokens - check their documentation.
    """
    try:
        # Get current refresh token from database
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            query = text("""
                SELECT garmin_refresh_token 
                FROM users 
                WHERE id = :user_id AND garmin_refresh_token IS NOT NULL
            """)
            
            result = await session.execute(query, {'user_id': user_id})
            row = result.fetchone()
            
            if not row or not row.garmin_refresh_token:
                raise HTTPException(status_code=404, detail="No refresh token found for user")
            
            refresh_token = row.garmin_refresh_token
        
        # Attempt to refresh token
        new_tokens = await garmin_oauth.refresh_token(refresh_token)
        
        # Update stored tokens
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            from datetime import datetime, timedelta
            
            token_expires_at = None
            if new_tokens.expires_at:
                token_expires_at = datetime.now() + timedelta(seconds=new_tokens.expires_at)
            
            update_query = text("""
                UPDATE users SET 
                    garmin_access_token = :access_token,
                    garmin_refresh_token = :refresh_token,
                    garmin_token_expires_at = :expires_at
                WHERE id = :user_id
            """)
            
            await session.execute(update_query, {
                'user_id': user_id,
                'access_token': new_tokens.access_token,
                'refresh_token': new_tokens.refresh_token,
                'expires_at': token_expires_at
            })
            await session.commit()
        
        logger.info(f"Successfully refreshed Garmin tokens for user {user_id}")
        
        return {
            "success": True,
            "message": "Token refreshed successfully",
            "token_expires_at": token_expires_at.isoformat() if token_expires_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh Garmin token for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {str(e)}")


@app.post("/auth/garmin/revoke")
async def garmin_revoke_access(user_id: str):
    """
    Revoke Garmin access for a user and clean up stored tokens.
    """
    try:
        # Get current access token
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            query = text("""
                SELECT garmin_access_token 
                FROM users 
                WHERE id = :user_id AND garmin_access_token IS NOT NULL
            """)
            
            result = await session.execute(query, {'user_id': user_id})
            row = result.fetchone()
            
            access_token = row.garmin_access_token if row else None
        
        # Attempt to revoke token with Garmin (if token exists)
        revoke_success = True
        if access_token:
            try:
                revoke_success = await garmin_oauth.revoke_access(access_token)
            except Exception as e:
                logger.warning(f"Failed to revoke token with Garmin for user {user_id}: {str(e)}")
                # Continue with local cleanup even if remote revocation fails
        
        # Clean up local tokens regardless of remote revocation result
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            update_query = text("""
                UPDATE users SET 
                    garmin_access_token = NULL,
                    garmin_refresh_token = NULL,
                    garmin_token_expires_at = NULL,
                    garmin_user_id = NULL,
                    activity_provider = 'strava'
                WHERE id = :user_id
            """)
            
            await session.execute(update_query, {'user_id': user_id})
            await session.commit()
        
        logger.info(f"Revoked Garmin access for user {user_id}")
        
        return {
            "success": True,
            "message": "Garmin access revoked successfully",
            "remote_revocation": revoke_success
        }
        
    except Exception as e:
        logger.error(f"Failed to revoke Garmin access for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Revocation failed: {str(e)}")


@app.get("/auth/garmin/status/{user_id}", response_model=GarminAuthStatusResponse)
async def garmin_auth_status(user_id: str):
    """
    Get Garmin authentication status for a user.
    """
    try:
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            query = text("""
                SELECT garmin_user_id, garmin_access_token, garmin_token_expires_at, activity_provider
                FROM users 
                WHERE id = :user_id
            """)
            
            result = await session.execute(query, {'user_id': user_id})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="User not found")
            
            has_token = bool(row.garmin_access_token)
            provider = row.activity_provider or 'strava'
            
            return GarminAuthStatusResponse(
                user_id=user_id,
                provider=provider,
                authenticated=has_token,
                garmin_user_id=row.garmin_user_id,
                token_expires_at=row.garmin_token_expires_at.isoformat() if row.garmin_token_expires_at else None
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get Garmin auth status for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.get("/api/config")
async def get_config():
    """Get current system configuration."""
    try:
        sync_days = int(os.getenv('STRAVA_SYNC_DAYS', '365'))
        return {
            "sync_period_days": sync_days,
            "sync_period_description": f"Last {sync_days} days" if sync_days < 365 else "Full year",
            "is_test_mode": sync_days < 365
        }
    except Exception as e:
        logger.error(f"Failed to get config: {str(e)}")
        return {"error": str(e)}


@app.get("/api/debug/garmin-config")
async def debug_garmin_config():
    """Debug endpoint to check Garmin OAuth configuration."""
    try:
        client_id = os.getenv("GARMIN_CLIENT_ID")
        client_secret = os.getenv("GARMIN_CLIENT_SECRET")
        redirect_uri = os.getenv("GARMIN_REDIRECT_URI", "http://localhost:8000/auth/garmin/callback")
        
        return {
            "client_id": client_id[:8] + "..." if client_id else "NOT SET",
            "client_secret": "SET" if client_secret else "NOT SET",
            "redirect_uri": redirect_uri,
            "auth_base_url": "https://connect.garmin.com/oauth2Confirm",
            "token_url": "https://diauth.garmin.com/di-oauth2-service/oauth/token",
            "configuration_valid": bool(client_id and client_secret)
        }
    except Exception as e:
        logger.error(f"Failed to get Garmin config: {str(e)}")
        return {"error": str(e)}


@app.get("/api/debug/garmin-token/{user_id}")
async def debug_garmin_token(user_id: str):
    """DEBUG ONLY: Display Garmin token for testing. REMOVE IN PRODUCTION!"""
    try:
        async with mysql_service.async_session() as session:
            from sqlalchemy import text
            query = text("""
                SELECT garmin_access_token, garmin_refresh_token, garmin_token_expires_at, 
                       activity_provider, created_at, updated_at
                FROM users 
                WHERE id = :user_id
            """)
            
            result = await session.execute(query, {'user_id': user_id})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="User not found")
            
            return {
                "user_id": user_id,
                "access_token": row.garmin_access_token,
                "refresh_token": row.garmin_refresh_token,
                "token_expires_at": row.garmin_token_expires_at.isoformat() if row.garmin_token_expires_at else None,
                "activity_provider": row.activity_provider,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "warning": "⚠️ THIS IS A DEBUG ENDPOINT - REMOVE IN PRODUCTION!"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get Garmin token for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get token: {str(e)}")


@app.post("/api/test/strava-auth")
async def test_strava_auth(request: Dict[str, str]):
    """Test Strava authentication by fetching athlete profile."""
    try:
        access_token = request.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token provided")
        
        strava_client = ingestion_service.strava_client
        athlete = await strava_client.get_athlete_profile(access_token)
        
        return {
            "success": True,
            "athlete_id": athlete.get("id"),
            "athlete_name": f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}",
            "message": "Authentication working correctly"
        }
    except Exception as e:
        logger.error(f"Test auth failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Authentication test failed"
        }


@app.post("/api/test/strava-activities")
async def test_strava_activities(request: Dict[str, str]):
    """Test fetching a small number of activities from Strava."""
    try:
        access_token = request.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token provided")
        
        strava_client = ingestion_service.strava_client
        
        # Test with a small number of activities
        logger.info("Testing Strava activities fetch with limited scope...")
        activities = await strava_client.get_activities(access_token)
        
        # Just return first 3 activities for testing
        test_activities = activities[:3] if len(activities) > 3 else activities
        
        return {
            "success": True,
            "total_activities": len(activities),
            "sample_activities": len(test_activities),
            "activities": test_activities,
            "message": f"Successfully fetched {len(activities)} activities"
        }
    except Exception as e:
        logger.error(f"Test activities failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": "Activities test failed"
        }


@app.get("/api/test/garmin-api/{user_id}/{activity_id}")
async def test_garmin_api(user_id: str, activity_id: str):
    """Test fetching detailed activity data from Garmin API."""
    try:
        from app.services.garmin_api import garmin_api_client
        
        # Get user's access token
        user_token = await garmin_api_client._get_user_token(user_id)
        if not user_token:
            return {"error": "No access token found for user"}
        
        # Test API call to get detailed activity data
        url = f"https://apis.garmin.com/wellness-api/rest/activityDetails/{activity_id}"
        headers = {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/json"
        }
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            
        return {
            "success": True,
            "status_code": response.status_code,
            "activity_id": activity_id,
            "response_data": response.json() if response.status_code == 200 else response.text,
            "has_power_data": "averagePower" in (response.json() if response.status_code == 200 else {})
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/test/connections")
async def test_connections():
    """Test all system connections (Pinecone, OpenAI, etc.)."""
    try:
        results = {}
        
        # Test Pinecone connection
        try:
            from pinecone import Pinecone
            import os
            
            api_key = os.getenv("PINECONE_API_KEY")
            if api_key:
                pc = Pinecone(api_key=api_key)
                indexes = pc.list_indexes()
                results["pinecone"] = {
                    "status": "connected",
                    "indexes": [idx.name for idx in indexes]
                }
            else:
                results["pinecone"] = {"status": "no_api_key"}
        except Exception as e:
            results["pinecone"] = {"status": "error", "error": str(e)}
        
        # Test OpenAI connection
        try:
            import openai
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input="test"
                )
                results["openai"] = {
                    "status": "connected",
                    "embedding_dimension": len(response.data[0].embedding)
                }
            else:
                results["openai"] = {"status": "no_api_key"}
        except Exception as e:
            results["openai"] = {"status": "error", "error": str(e)}
        
        return {
            "success": True,
            "connections": results,
            "message": "Connection test completed"
        }
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Connection test failed"
        }


@app.post("/api/activities/ingest")
async def ingest_activities(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Start activity ingestion for a user.
    This runs in the background to avoid timeout issues.
    """
    try:
        # Start ingestion in background
        background_tasks.add_task(
            ingestion_service.ingest_user_activities,
            request.user_id,
            request.access_token,
            request.refresh_token,
            request.full_sync,
            request.sync_days
        )
        
        return {
            "message": "Activity ingestion started",
            "user_id": request.user_id,
            "status": "in_progress"
        }
        
    except Exception as e:
        logger.error(f"Failed to start ingestion for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {str(e)}")


@app.get("/api/activities/status/{user_id}")
async def get_ingestion_status(user_id: str):
    """Get current ingestion status for a user."""
    try:
        status = await ingestion_service.get_ingestion_status(user_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="No ingestion status found for user")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ingestion status for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.post("/api/activities/search", response_model=List[SearchResult])
async def search_activities(request: SearchRequest):
    """Search user's activities using natural language."""
    try:
        results = await ingestion_service.search_user_activities(
            user_id=request.user_id,
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Search failed for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/users/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(user_id: str):
    """Get statistics about user's stored activities."""
    try:
        stats = await ingestion_service.get_user_activity_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/api/users/{user_id}/data")
async def delete_user_data(user_id: str):
    """Delete all data for a user."""
    try:
        await ingestion_service.delete_user_data(user_id)
        
        return {"message": f"All data deleted for user {user_id}"}
        
    except Exception as e:
        logger.error(f"Failed to delete data for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Process a conversational query about fitness activities."""
    try:
        response = await conversation_service.process_query(
            user_id=request.user_id,
            query=request.query,
            conversation_id=request.conversation_id
        )
        
        return ChatResponse(
            response=response.response,
            relevant_activities=response.relevant_activities,
            follow_up_questions=response.follow_up_questions,
            conversation_id=response.conversation_id,
            timestamp=response.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat failed for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/insights")
async def get_fitness_insights(request: InsightsRequest):
    """Generate comprehensive fitness insights for a user."""
    try:
        insights = await conversation_service.get_fitness_insights(
            user_id=request.user_id,
            time_period=request.time_period
        )
        
        return {"insights": insights, "time_period": request.time_period}
        
    except Exception as e:
        logger.error(f"Insights generation failed for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")


@app.get("/api/chat/history/{user_id}")
async def get_chat_history(user_id: str, conversation_id: Optional[str] = None, limit: int = 50):
    """Get conversation history for a user."""
    try:
        history = conversation_service.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit
        )
        
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Failed to get chat history for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")


@app.delete("/api/chat/history/{user_id}")
async def clear_chat_history(user_id: str, conversation_id: Optional[str] = None):
    """Clear conversation history for a user."""
    try:
        conversation_service.clear_conversation(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        return {"message": "Chat history cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear chat history for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")


@app.get("/api/fitness/{user_id}/tss")
async def get_user_tss_data(user_id: str):
    """Get TSS calculations for a user."""
    try:
        tss_calculations = await ingestion_service.get_user_tss_calculations(user_id)
        
        if not tss_calculations:
            raise HTTPException(status_code=404, detail="No TSS data found. Please complete a Full Sync first.")
        
        return {
            "user_id": user_id,
            "total_calculations": len(tss_calculations),
            "tss_data": [calc.dict() for calc in tss_calculations]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get TSS data for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get TSS data: {str(e)}")


@app.get("/api/fitness/{user_id}/metrics")
async def get_user_fitness_metrics(user_id: str):
    """Get fitness metrics (CTL, ATL, TSB) for a user."""
    try:
        fitness_metrics = await ingestion_service.get_user_fitness_metrics(user_id)
        
        if not fitness_metrics:
            raise HTTPException(status_code=404, detail="No fitness data found. Please complete a Full Sync first.")
        
        return {
            "user_id": user_id,
            "total_days": len(fitness_metrics),
            "fitness_data": [metrics.dict() for metrics in fitness_metrics]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fitness metrics for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get fitness metrics: {str(e)}")


@app.get("/api/fitness/{user_id}/status")
async def get_user_fitness_status(user_id: str):
    """Get current fitness status for a user."""
    try:
        fitness_status = await ingestion_service.get_user_fitness_status(user_id)
        
        return {
            "user_id": user_id,
            "fitness_status": fitness_status
        }
        
    except Exception as e:
        logger.error(f"Failed to get fitness status for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get fitness status: {str(e)}")


@app.get("/api/fitness/{user_id}/thresholds")
async def get_user_thresholds(user_id: str, force_recalculate: bool = False):
    """Get threshold estimates for a user using MySQL activities data."""
    try:
        logger.info(f"🚀 THRESHOLDS REQUEST: user_id={user_id}, force_recalculate={force_recalculate}")
        
        # Use MySQL-based threshold estimation instead of vector DB approach
        if force_recalculate:
            logger.info(f"🔄 FORCE RECALCULATE: Starting threshold recalculation for user {user_id}")
            
            # Trigger threshold recalculation using our working approach
            from app.services.threshold_estimator import ThresholdEstimator
            from sqlalchemy import text
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=180)
            logger.info(f"📅 CUTOFF DATE: {cutoff_date}")
            
            async with mysql_service.async_session() as session:
                logger.info(f"🗄️  DATABASE SESSION: Starting query for user {user_id}")
                
                # Get activities from MySQL (not vector DB)
                activities_query = text("""
                    SELECT activity_data, processed_at
                    FROM activities 
                    WHERE user_id = :user_id 
                    AND data_source = 'garmin'
                    AND processed_at >= :cutoff_date
                    ORDER BY processed_at DESC
                    LIMIT 200
                """)
                
                logger.info(f"🔍 EXECUTING QUERY: About to execute database query")
                result = await session.execute(activities_query, {
                    "user_id": user_id,
                    "cutoff_date": cutoff_date
                })
                logger.info(f"✅ QUERY EXECUTED: Database query completed successfully")
                
                activities = []
                logger.info(f"🔄 PARSING ACTIVITIES: Starting to parse activity data")
                row_count = 0
                for row in result.fetchall():
                    row_count += 1
                    logger.debug(f"  Processing row {row_count}")
                    if row.activity_data:
                        try:
                            # Handle JSON data properly
                            if isinstance(row.activity_data, str):
                                import json
                                activity_dict = json.loads(row.activity_data)
                            else:
                                activity_dict = row.activity_data
                            activities.append(activity_dict)
                        except Exception as parse_error:
                            logger.error(f"❌ JSON PARSE ERROR row {row_count}: {str(parse_error)}")
                            continue
                
                logger.info(f"📊 ACTIVITIES PARSED: Found {len(activities)} valid activities from {row_count} rows")
                
                if len(activities) >= 10:
                    logger.info(f"✅ SUFFICIENT DATA: {len(activities)} activities >= 10, proceeding with threshold estimation")
                    
                    # Normalize Garmin data format for threshold estimator
                    logger.info(f"🔄 DATA TRANSFORMATION: Converting Garmin format to threshold estimator format")
                    normalized_activities = []
                    
                    for activity in activities:
                        # Extract power and duration from Garmin format
                        power_avg = None
                        duration_seconds = None
                        activity_type = None
                        
                        # Try different power data locations
                        if activity.get('performance_data', {}).get('power_avg'):
                            power_avg = float(activity['performance_data']['power_avg'])
                        elif activity.get('averagePower'):
                            power_avg = float(activity['averagePower'])
                        
                        # Try different duration locations
                        if activity.get('durationInSeconds'):
                            duration_seconds = int(activity['durationInSeconds'])
                        elif activity.get('duration'):
                            duration_seconds = int(activity['duration'])
                        
                        # Get activity type
                        activity_type = activity.get('activityType', 'UNKNOWN')
                        
                        # Only include activities with power data
                        if power_avg and power_avg > 0 and duration_seconds:
                            normalized_activity = {
                                'average_watts': power_avg,
                                'moving_time': duration_seconds,  # Threshold estimator expects seconds, converts internally
                                'type': activity_type,
                                'sport_type': activity_type.lower() if activity_type else 'unknown',
                                # Include other fields that might be useful
                                'distance': activity.get('distanceInMeters', 0) / 1000,  # Convert to km
                                'total_elevation_gain': activity.get('totalElevationGainInMeters', 0),
                                'average_heartrate': activity.get('averageHeartRateInBeatsPerMinute'),
                                'max_heartrate': activity.get('maxHeartRateInBeatsPerMinute'),
                                'start_date': activity.get('startTimeInSeconds')
                            }
                            normalized_activities.append(normalized_activity)
                            logger.debug(f"  Normalized activity: {activity_type}, power={power_avg}W, duration={duration_seconds}s")
                    
                    logger.info(f"✅ DATA TRANSFORMATION COMPLETE: {len(normalized_activities)} activities with power data normalized")
                    
                    if len(normalized_activities) < 3:
                        logger.warning(f"⚠️ INSUFFICIENT POWER DATA: Only {len(normalized_activities)} activities have power data")
                        # Continue with what we have, but threshold estimation may return None
                    
                    # Estimate thresholds using normalized data
                    logger.info(f"🧮 THRESHOLD ESTIMATOR: Creating ThresholdEstimator instance")
                    estimator = ThresholdEstimator()
                    
                    logger.info(f"🔥 THRESHOLD CALCULATION: About to call estimate_all_thresholds() with {len(normalized_activities)} power activities")
                    
                    # DEBUG: Log sample normalized activity for verification
                    if normalized_activities:
                        sample = normalized_activities[0]
                        logger.info(f"🔍 SAMPLE NORMALIZED ACTIVITY: power={sample.get('average_watts')}W, duration={sample.get('moving_time')}s, type={sample.get('type')}")
                    
                    estimated_thresholds = estimator.estimate_all_thresholds(normalized_activities, data_source="garmin")
                    
                    logger.info(f"🎉 THRESHOLD CALCULATION COMPLETE: Results - FTP: {estimated_thresholds.ftp}, LTHR: {estimated_thresholds.lthr}")
                    logger.info(f"📊 THRESHOLD DETAILS: Running pace: {estimated_thresholds.running_threshold_pace}, Swimming pace: {estimated_thresholds.swimming_threshold_pace}")
                    
                    # Save thresholds
                    logger.info(f"💾 SAVING THRESHOLDS: Preparing to save to database")
                    thresholds_dict = {
                        'ftp': estimated_thresholds.ftp,
                        'running_threshold_pace': estimated_thresholds.running_threshold_pace,
                        'swimming_threshold_pace': estimated_thresholds.swimming_threshold_pace,
                        'lthr': estimated_thresholds.lthr,
                        'max_hr': estimated_thresholds.max_hr
                    }
                    logger.info(f"📋 THRESHOLDS PREPARED: {thresholds_dict}")
                    
                    # Calculate confidence
                    confidence = {
                        'ftp': 'medium' if estimated_thresholds.ftp else 'estimated',
                        'running_threshold_pace': 'medium' if estimated_thresholds.running_threshold_pace else 'estimated',
                        'swimming_threshold_pace': 'medium' if estimated_thresholds.swimming_threshold_pace else 'estimated',
                        'lthr': 'medium' if estimated_thresholds.lthr else 'estimated',
                        'max_hr': 'medium' if estimated_thresholds.max_hr else 'estimated'
                    }
                    logger.info(f"🎯 CONFIDENCE CALCULATED: {confidence}")
                    
                    logger.info(f"💾 DATABASE SAVE: About to save thresholds to database")
                    await mysql_service.save_user_thresholds(user_id, thresholds_dict, confidence)
                    logger.info(f"✅ DATABASE SAVE COMPLETE: Thresholds saved successfully")
                    
                    # Return the estimated thresholds
                    logger.info(f"🚀 RETURNING RESULTS: Preparing response")
                    response = {
                        "user_id": user_id,
                        "thresholds": {
                            "ftp": estimated_thresholds.ftp,
                            "running_threshold_pace": estimated_thresholds.running_threshold_pace,
                            "swimming_threshold_pace": estimated_thresholds.swimming_threshold_pace,
                            "lthr": estimated_thresholds.lthr,
                            "confidence": confidence
                        }
                    }
                    logger.info(f"🎉 SUCCESS: Threshold calculation completed successfully for user {user_id}")
                    return response
                else:
                    logger.warning(f"⚠️  INSUFFICIENT DATA: Only {len(activities)} activities found, need >= 10")
        
        # Get existing thresholds from database
        logger.info(f"📖 FALLBACK: Getting existing thresholds from database for user {user_id}")
        existing_thresholds = await mysql_service.get_user_thresholds(user_id)
        logger.info(f"📖 EXISTING THRESHOLDS: {existing_thresholds}")
        
        response = {
            "user_id": user_id,
            "thresholds": existing_thresholds
        }
        logger.info(f"📤 RETURNING EXISTING: {response}")
        return response
        
    except Exception as e:
        logger.error(f"❌ EXCEPTION in get_user_thresholds for user {user_id}: {str(e)}")
        import traceback
        logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get thresholds: {str(e)}")


@app.post("/api/chat/follow-up", response_model=ChatResponse)
async def ask_follow_up(request: ChatRequest):
    """Process a follow-up question."""
    try:
        response = await conversation_service.ask_follow_up(
            user_id=request.user_id,
            follow_up_question=request.query,
            conversation_id=request.conversation_id
        )
        
        return ChatResponse(
            response=response.response,
            relevant_activities=response.relevant_activities,
            follow_up_questions=response.follow_up_questions,
            conversation_id=response.conversation_id,
            timestamp=response.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Follow-up failed for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Follow-up failed: {str(e)}")


@app.get("/api/activities/{activity_id}/power-metrics")
async def get_power_metrics(activity_id: int = Path(..., description="The Strava activity ID")):
    """Get detailed power metrics for a cycling activity."""
    try:
        # Get activity from ingestion service (this would need to be implemented)
        activity = await ingestion_service.get_activity_by_id(activity_id)
        
        if not activity:
            raise HTTPException(status_code=404, detail="Activity not found")
        
        # Calculate power metrics
        metrics_service = ingestion_service.metrics_service
        thresholds = await ingestion_service.get_user_threshold_estimates(str(activity.get('athlete', {}).get('id', '')))
        
        np = metrics_service.calculate_normalized_power(activity)
        if_value = metrics_service.calculate_intensity_factor(np, thresholds.get('ftp', 200)) if np else None
        vi = metrics_service.calculate_variability_index(activity)
        ef = metrics_service.calculate_efficiency_factor(activity)
        
        return {
            "activity_id": activity_id,
            "activity_name": activity.get('name', ''),
            "metrics": {
                "average_power": activity.get('average_watts'),
                "normalized_power": np,
                "intensity_factor": if_value,
                "variability_index": vi,
                "efficiency_factor": ef,
                "ftp": thresholds.get('ftp', 200),
                "tss": None  # Would calculate TSS here
            },
            "has_power_data": bool(activity.get('average_watts'))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get power metrics for activity {activity_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get power metrics: {str(e)}")


@app.get("/api/activities/{activity_id}/running-metrics")
async def get_running_metrics(activity_id: int = Path(..., description="The Strava activity ID")):
    """Get detailed running metrics for a running activity."""
    try:
        # Get activity from ingestion service
        activity = await ingestion_service.get_activity_by_id(activity_id)
        
        if not activity:
            raise HTTPException(status_code=404, detail="Activity not found")
        
        # Calculate running metrics
        metrics_service = ingestion_service.metrics_service
        
        gap = metrics_service.calculate_grade_adjusted_pace(activity)
        ef = metrics_service.calculate_efficiency_factor(activity)
        
        # Calculate base pace
        distance = activity.get('distance', 0)
        moving_time = activity.get('moving_time', 0)
        pace_min_per_km = None
        if distance > 0 and moving_time > 0:
            pace_min_per_km = (moving_time / 60) / (distance / 1000)
        
        return {
            "activity_id": activity_id,
            "activity_name": activity.get('name', ''),
            "metrics": {
                "distance_km": distance / 1000 if distance else 0,
                "pace_min_per_km": round(pace_min_per_km, 2) if pace_min_per_km else None,
                "grade_adjusted_pace": gap,
                "elevation_gain": activity.get('total_elevation_gain', 0),
                "average_cadence": activity.get('average_cadence'),
                "average_heartrate": activity.get('average_heartrate'),
                "efficiency_factor": ef
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get running metrics for activity {activity_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get running metrics: {str(e)}")


@app.get("/api/athlete/{user_id}/performance-trends")
async def get_performance_trends(
    user_id: str,
    sport_type: Optional[str] = None,
    days: int = 90
):
    """Get performance trends over time for an athlete."""
    try:
        # Get user's activities
        activities = await ingestion_service.get_user_activities(user_id, days=days, sport_type=sport_type)
        
        if not activities:
            return {
                "user_id": user_id,
                "sport_type": sport_type,
                "period_days": days,
                "trends": {
                    "power": [],
                    "pace": [],
                    "efficiency": [],
                    "heart_rate": []
                }
            }
        
        metrics_service = ingestion_service.metrics_service
        thresholds = await ingestion_service.get_user_threshold_estimates(user_id)
        
        # Calculate trends
        power_trend = []
        pace_trend = []
        efficiency_trend = []
        hr_trend = []
        
        for activity in activities:
            date = activity.get('start_date_local', '').split('T')[0]
            sport = metrics_service._categorize_sport(activity.get('type', ''))
            
            if sport == 'cycling' and activity.get('average_watts'):
                np = metrics_service.calculate_normalized_power(activity)
                if_value = metrics_service.calculate_intensity_factor(np, thresholds.get('ftp', 200)) if np else None
                power_trend.append({
                    "date": date,
                    "normalized_power": np,
                    "intensity_factor": if_value,
                    "activity_name": activity.get('name', '')
                })
            
            if sport == 'running':
                gap = metrics_service.calculate_grade_adjusted_pace(activity)
                if gap:
                    pace_trend.append({
                        "date": date,
                        "grade_adjusted_pace": gap,
                        "activity_name": activity.get('name', '')
                    })
            
            ef = metrics_service.calculate_efficiency_factor(activity)
            if ef:
                efficiency_trend.append({
                    "date": date,
                    "efficiency_factor": ef,
                    "sport": sport,
                    "activity_name": activity.get('name', '')
                })
            
            if activity.get('average_heartrate'):
                hr_trend.append({
                    "date": date,
                    "average_hr": activity.get('average_heartrate'),
                    "max_hr": activity.get('max_heartrate'),
                    "activity_name": activity.get('name', '')
                })
        
        return {
            "user_id": user_id,
            "sport_type": sport_type,
            "period_days": days,
            "total_activities": len(activities),
            "trends": {
                "power": power_trend,
                "pace": pace_trend,
                "efficiency": efficiency_trend,
                "heart_rate": hr_trend
            },
            "thresholds": thresholds
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance trends for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance trends: {str(e)}")


@app.get("/api/activities")
async def get_activities(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    sport_type: Optional[str] = None,
    include_metrics: bool = True
):
    """Get user's activities with computed metrics."""
    try:
        # Get activities from ingestion service
        all_activities = await ingestion_service.get_user_activities(
            user_id=user_id,
            days=365,  # Get last year by default
            sport_type=sport_type
        )
        
        # Apply pagination
        paginated_activities = all_activities[offset:offset + limit]
        
        # Format response
        activities_response = []
        for activity in paginated_activities:
            activity_data = {
                "id": activity.get('id'),
                "name": activity.get('name'),
                "type": activity.get('type'),
                "sport_type": activity.get('sport_type', activity.get('type')),
                "start_date": activity.get('start_date_local'),
                "distance": activity.get('distance'),
                "moving_time": activity.get('moving_time'),
                "elapsed_time": activity.get('elapsed_time'),
                "total_elevation_gain": activity.get('total_elevation_gain'),
                "average_speed": activity.get('average_speed'),
                "max_speed": activity.get('max_speed'),
                "average_watts": activity.get('average_watts'),
                "average_heartrate": activity.get('average_heartrate'),
                "average_cadence": activity.get('average_cadence'),
                "achievement_count": activity.get('achievement_count', 0),
                "kudos_count": activity.get('kudos_count', 0),
                "comment_count": activity.get('comment_count', 0),
                "athlete_count": activity.get('athlete_count', 1),
                "trainer": activity.get('trainer', False),
                "commute": activity.get('commute', False),
                "manual": activity.get('manual', False),
                "private": activity.get('private', False)
            }
            
            # Include computed metrics if requested
            if include_metrics:
                activity_data.update({
                    "normalized_power": activity.get('normalized_power'),
                    "intensity_factor": activity.get('intensity_factor'),
                    "efficiency_factor": activity.get('efficiency_factor'),
                    "grade_adjusted_pace": activity.get('grade_adjusted_pace'),
                    "variability_index": activity.get('variability_index')
                })
            
            activities_response.append(activity_data)
        
        return {
            "user_id": user_id,
            "total_activities": len(all_activities),
            "limit": limit,
            "offset": offset,
            "activities": activities_response
        }
        
    except Exception as e:
        logger.error(f"Failed to get activities for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get activities: {str(e)}")


@app.get("/api/debug/activity-types/{user_id}")
async def debug_activity_types(user_id: str):
    """Debug endpoint to show what activity types are in the user's TSS data."""
    try:
        tss_calculations = await ingestion_service.get_user_tss_calculations(user_id)
        
        if not tss_calculations:
            return {"message": "No TSS calculations found. Please complete a Full Sync first."}
        
        # Count activity types
        original_activity_type_counts = {}
        sport_type_counts = {}
        
        for calc in tss_calculations:
            # Count by original activity type
            original_type = getattr(calc, 'original_activity_type', 'unknown')
            original_activity_type_counts[original_type] = original_activity_type_counts.get(original_type, 0) + 1
            
            # Count by categorized sport type
            sport_type = calc.sport_type
            sport_type_counts[sport_type] = sport_type_counts.get(sport_type, 0) + 1
        
        # Get unique original activity types
        unique_original_types = list(set(original_activity_type_counts.keys()))
        
        return {
            "user_id": user_id,
            "total_activities": len(tss_calculations),
            "original_activity_types": original_activity_type_counts,
            "unique_activity_types": unique_original_types,
            "categorized_sport_breakdown": sport_type_counts,
            "categorization_issues": {
                "activities_categorized_as_other": sum(1 for calc in tss_calculations if calc.sport_type == 'other'),
                "activities_with_unknown_type": original_activity_type_counts.get('unknown', 0)
            },
            "sample_calculations": [
                {
                    "activity_id": calc.activity_id,
                    "original_activity_type": getattr(calc, 'original_activity_type', 'unknown'),
                    "sport_type": calc.sport_type,
                    "tss_value": calc.tss_value,
                    "calculation_method": calc.calculation_method,
                    "notes": calc.notes,
                    "date": getattr(calc, 'activity_date', 'unknown')
                }
                for calc in tss_calculations[:20]  # Show first 20 as sample
            ]
        }
        
    except Exception as e:
        logger.error(f"Debug activity types failed for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


@app.post("/api/races", response_model=RaceResponse)
async def create_race_entry(request: RaceCreateRequest):
    """Create a new race entry for a user."""
    try:
        race_entry = RaceEntry(
            user_id=request.user_id,
            race_type=request.race_type,
            race_date=request.race_date
        )
        
        created_race = await mysql_service.create_race_entry(request.user_id, race_entry)
        
        if not created_race:
            raise HTTPException(status_code=500, detail="Failed to create race entry")
        
        return created_race
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create race entry for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create race entry: {str(e)}")


@app.get("/api/races/{user_id}", response_model=RaceListResponse)
async def get_user_races(user_id: str, upcoming_only: bool = False):
    """Get all race entries for a user."""
    try:
        races = await mysql_service.get_user_race_entries(user_id, upcoming_only=upcoming_only)
        
        return RaceListResponse(
            user_id=user_id,
            total_races=len(races),
            races=races
        )
        
    except Exception as e:
        logger.error(f"Failed to get race entries for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get race entries: {str(e)}")


@app.get("/api/races/{user_id}/{race_id}", response_model=RaceResponse)
async def get_race_entry(user_id: str, race_id: int):
    """Get a specific race entry for a user."""
    try:
        race = await mysql_service.get_race_entry(user_id, race_id)
        
        if not race:
            raise HTTPException(status_code=404, detail="Race entry not found")
        
        return race
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get race entry {race_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get race entry: {str(e)}")


@app.put("/api/races/{user_id}/{race_id}", response_model=RaceResponse)
async def update_race_entry(user_id: str, race_id: int, request: RaceUpdateRequest):
    """Update a race entry for a user."""
    try:
        # Convert request to dict, excluding None values
        updates = {}
        if request.race_type is not None:
            updates['race_type'] = request.race_type
        if request.race_date is not None:
            updates['race_date'] = request.race_date
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        updated_race = await mysql_service.update_race_entry(user_id, race_id, updates)
        
        if not updated_race:
            raise HTTPException(status_code=404, detail="Race entry not found")
        
        return updated_race
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update race entry {race_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update race entry: {str(e)}")


@app.delete("/api/races/{user_id}/{race_id}")
async def delete_race_entry(user_id: str, race_id: int):
    """Delete a race entry for a user."""
    try:
        success = await mysql_service.delete_race_entry(user_id, race_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Race entry not found")
        
        return {"message": f"Race entry {race_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete race entry {race_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete race entry: {str(e)}")


@app.post("/api/training-plans/generate", response_model=TrainingPlanJobResponse)
async def generate_training_plan(request: TrainingPlanRequest):
    """
    Start training plan generation as an async job.
    
    This endpoint immediately returns a job ID and starts processing in the background.
    Use the job endpoints to poll for status and retrieve the completed plan.
    """
    try:
        logger.info(f"Creating training plan job for user {request.user_id}, race ID: {request.race_id}")
        
        # Gather enriched data that will be sent to OpenAI (same as background task does)
        logger.info(f"Pre-gathering enriched training context for job creation...")
        
        # Get and validate race entry
        race_entry = await training_plan_service._get_and_validate_race(request.user_id, request.race_id)
        
        # Gather fitness data
        fitness_data = await training_plan_service._gather_user_fitness_data(request.user_id)
        
        # Build the complete prompt that would be sent to OpenAI
        prompt = training_plan_service._build_training_plan_prompt(request, race_entry, fitness_data)
        
        # Helper function to serialize dates/datetimes to ISO format
        def serialize_for_json(obj):
            """Recursively convert date/datetime objects to ISO strings for JSON serialization."""
            if isinstance(obj, (date, datetime)):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {key: serialize_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [serialize_for_json(item) for item in obj]
            else:
                return obj
        
        # Create enriched request data that matches what goes to OpenAI
        enriched_request_data = {
            # Original request fields
            "race_id": request.race_id,
            "user_id": request.user_id,
            "days_per_week": request.days_per_week,
            "years_experience": request.years_experience.value if hasattr(request.years_experience, 'value') else str(request.years_experience),
            "injury_limitations": request.injury_limitations or [],
            "max_hours_per_week": request.max_hours_per_week,
            "available_equipment": [eq.value if hasattr(eq, 'value') else str(eq) for eq in (request.available_equipment or [])],
            "preferred_rest_days": request.preferred_rest_days or [],
            "safe_outdoor_routes": request.safe_outdoor_routes,
            "upcoming_disruptions": request.upcoming_disruptions or [],
            "include_cross_training": request.include_cross_training,
            "preferred_training_days": request.preferred_training_days or [],
            "preferred_training_time": request.preferred_training_time,
            "include_strength_training": request.include_strength_training,
            
            # Enriched context data that gets sent to OpenAI (JSON serialized)
            "race_context": serialize_for_json(race_entry),
            "fitness_context": serialize_for_json(fitness_data),
            "openai_prompt": prompt,
            "enriched_at": datetime.now().isoformat()
        }
        
        # Create the job with enriched data
        job_response = await training_plan_job_service.create_job_with_enriched_data(
            request.user_id, 
            request, 
            enriched_request_data
        )
        
        # Start background processing with asyncio.create_task for true concurrency
        asyncio.create_task(process_training_plan_job_with_context(
            job_response.job_id, 
            request, 
            race_entry, 
            fitness_data, 
            prompt
        ))
        
        return job_response
        
    except Exception as e:
        logger.error(f"Failed to create training plan job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create training plan job: {str(e)}")


@app.get("/api/training-plans/{user_id}")
async def get_user_training_plans(
    user_id: str,
    active_only: bool = False
):
    """Get all training plans for a user."""
    try:
        plans = await training_plan_service.get_user_training_plans(
            user_id=user_id,
            active_only=active_only
        )
        
        # Convert to summaries for listing
        summaries = []
        for plan in plans:
            summary = TrainingPlanSummary(
                plan_id=plan.plan_id,
                race_id=plan.race_id,
                race_type=plan.race_type,
                race_date=plan.race_date,
                created_at=plan.created_at,
                plan_duration_weeks=plan.plan_duration_weeks,
                total_training_hours=plan.total_training_hours,
                status="active" if plan.race_date >= date.today() else "completed"
            )
            summaries.append(summary)
        
        return {
            "user_id": user_id,
            "total_plans": len(summaries),
            "plans": summaries
        }
        
    except Exception as e:
        logger.error(f"Failed to get training plans for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get training plans: {str(e)}")


@app.get("/api/training-plans/{user_id}/{plan_id}")
async def get_training_plan_details(user_id: str, plan_id: str):
    """Get detailed training plan by ID."""
    try:
        # Get the full plan details including weekly plans and workouts
        plan = await training_plan_service.get_training_plan_details(user_id=user_id, plan_id=plan_id)
        
        if not plan:
            raise HTTPException(status_code=404, detail="Training plan not found")
        
        return plan
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training plan {plan_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get training plan: {str(e)}")


@app.delete("/api/training-plans/{user_id}/{plan_id}")
async def delete_training_plan(user_id: str, plan_id: str):
    """Delete a training plan."""
    try:
        # Attempt to delete the training plan
        success = await training_plan_service.delete_training_plan(user_id=user_id, plan_id=plan_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Training plan not found")
        
        return {"message": f"Training plan {plan_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training plan {plan_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete training plan: {str(e)}")


# Training Plan Job Endpoints

@app.get("/api/training-plans/jobs/{job_id}/status", response_model=TrainingPlanJobStatusResponse)
async def get_job_status(job_id: str, user_id: str):
    """Get the current status of a training plan generation job."""
    logger.info(f"🔍 STATUS REQUEST: Getting status for job {job_id}, user {user_id}")
    
    try:
        status = await training_plan_job_service.get_job_status(job_id, user_id)
        
        if not status:
            logger.info(f"🔍 STATUS RESPONSE: Job {job_id} not found - returning 404")
            raise HTTPException(status_code=404, detail="Job not found")
        
        logger.info(f"🔍 STATUS RESPONSE: Job {job_id} status={status.status}, progress={status.progress}%")
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔍 STATUS ERROR: Failed to get job status for {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@app.get("/api/training-plans/jobs/{job_id}/result", response_model=TrainingPlanJobResult)
async def get_job_result(job_id: str, user_id: str):
    """Get the result of a completed training plan generation job."""
    try:
        result = await training_plan_job_service.get_job_result(job_id, user_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job result for {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job result: {str(e)}")


@app.post("/api/training-plans/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user_id: str):
    """Cancel a pending or processing training plan generation job."""
    try:
        success = await training_plan_job_service.cancel_job(job_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@app.get("/api/training-plans/jobs/{user_id}", response_model=TrainingPlanJobListResponse)
async def get_user_jobs(user_id: str, status_filter: Optional[JobStatus] = None):
    """Get all training plan generation jobs for a user."""
    try:
        jobs = await training_plan_job_service.get_user_jobs(user_id, status_filter)
        return jobs
        
    except Exception as e:
        logger.error(f"Failed to get jobs for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get jobs: {str(e)}")


@app.post("/api/admin/training-plans/jobs/cleanup")
async def cleanup_old_jobs(days_old: int = 7):
    """Admin endpoint to cleanup completed jobs older than specified days."""
    try:
        deleted_count = await training_plan_job_service.cleanup_old_jobs(days_old)
        
        return {
            "message": f"Cleaned up {deleted_count} old jobs",
            "deleted_count": deleted_count,
            "days_old": days_old
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old jobs: {str(e)}")


@app.post("/api/debug/test-job")
async def test_job_creation():
    """Debug endpoint to test job creation and immediate status check."""
    try:
        from uuid import uuid4
        job_id = str(uuid4())
        user_id = "test_user"
        
        # Create a test job directly
        success = await mysql_service.create_training_plan_job(
            job_id=job_id,
            user_id=user_id,
            race_id=1,
            request_data={"test": "data"}
        )
        
        logger.info(f"Test job creation success: {success}")
        
        # Immediately check status
        job_data = await mysql_service.get_job_by_id(job_id)
        
        return {
            "job_created": success,
            "job_id": job_id,
            "job_data": job_data
        }
        
    except Exception as e:
        logger.error(f"Test job creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


@app.get("/api/debug/direct-db-test/{job_id}")
async def direct_db_test(job_id: str):
    """Super simple direct database test."""
    logger.info(f"🚨 DIRECT DB TEST: Starting for job {job_id}")
    
    try:
        # Use the read session directly - bypass all service layers
        async with mysql_service.read_session() as session:
            logger.info(f"🚨 DIRECT DB TEST: Got session")
            from sqlalchemy import text
            result = await session.execute(text("SELECT * FROM training_plan_jobs WHERE id = :job_id"), {"job_id": job_id})
            logger.info(f"🚨 DIRECT DB TEST: Query executed")
            row = result.fetchone()
            logger.info(f"🚨 DIRECT DB TEST: Row fetched: {row is not None}")
            
            if row:
                return {"found": True, "status": row.status, "progress": row.progress}
            else:
                return {"found": False}
                
    except Exception as e:
        logger.error(f"🚨 DIRECT DB TEST ERROR: {str(e)}")
        return {"error": str(e)}


# Background Task Processing

async def process_training_plan_job(job_id: str, request: TrainingPlanRequest):
    """
    Background task to process training plan generation.
    
    This function runs the actual training plan generation with progress tracking
    and handles completion or failure scenarios.
    """
    logger.info(f"Starting background processing for job {job_id}")
    
    try:
        # Update job status to processing
        logger.info(f"Job {job_id}: Updating status to PROCESSING")
        success = await training_plan_job_service.update_progress(
            job_id=job_id,
            progress=0,
            current_step="Starting training plan generation...",
            status=JobStatus.PROCESSING
        )
        logger.info(f"Job {job_id}: Status update success: {success}")
        
        # Create progress callback and cancellation check
        progress_callback = training_plan_job_service.create_progress_callback(job_id)
        
        async def cancellation_check():
            return await training_plan_job_service.is_job_cancelled(job_id)
        
        # Generate the training plan with callbacks
        response = await training_plan_service.generate_training_plan(
            request=request,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check
        )
        
        if response.success and response.plan:
            # Complete the job successfully
            await training_plan_job_service.complete_job(job_id, response.plan)
            logger.info(f"Job {job_id} completed successfully with plan {response.plan.plan_id}")
        else:
            # Handle failure
            error_message = response.error or "Training plan generation failed"
            await training_plan_job_service.handle_job_failure(job_id, error_message)
            logger.error(f"Job {job_id} failed: {error_message}")
            
    except Exception as e:
        # Handle unexpected errors
        error_message = f"Unexpected error in job processing: {str(e)}"
        logger.error(f"Job {job_id} failed with exception: {error_message}")
        
        try:
            await training_plan_job_service.handle_job_failure(job_id, error_message)
        except Exception as cleanup_error:
            logger.error(f"Failed to handle job failure for {job_id}: {str(cleanup_error)}")


async def process_training_plan_job_with_context(
    job_id: str, 
    request: TrainingPlanRequest,
    race_entry: Dict[str, Any],
    fitness_data: Dict[str, Any],
    prompt: str
):
    """
    Background task to process training plan generation with pre-gathered context.
    
    This optimized version uses pre-gathered race and fitness data to avoid
    re-querying the database and ensure consistency with stored job data.
    """
    logger.info(f"Starting optimized background processing for job {job_id}")
    
    try:
        # Update job status to processing
        logger.info(f"Job {job_id}: Updating status to PROCESSING")
        success = await training_plan_job_service.update_progress(
            job_id=job_id,
            progress=0,
            current_step="Starting training plan generation...",
            status=JobStatus.PROCESSING
        )
        logger.info(f"Job {job_id}: Status update success: {success}")
        
        # Create progress callback and cancellation check
        progress_callback = training_plan_job_service.create_progress_callback(job_id)
        
        async def cancellation_check():
            return await training_plan_job_service.is_job_cancelled(job_id)
        
        # Generate the training plan using the optimized method with pre-gathered context
        response = await training_plan_service.generate_training_plan_with_context(
            request=request,
            race_entry=race_entry, 
            fitness_data=fitness_data,
            prompt=prompt,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check
        )
        
        if response.success and response.plan:
            # Complete the job successfully
            await training_plan_job_service.complete_job(job_id, response.plan)
            logger.info(f"Job {job_id} completed successfully with plan {response.plan.plan_id}")
        else:
            # Handle failure
            error_message = response.error or "Training plan generation failed"
            await training_plan_job_service.handle_job_failure(job_id, error_message)
            logger.error(f"Job {job_id} failed: {error_message}")
            
    except Exception as e:
        # Handle unexpected errors
        error_message = f"Unexpected error in job processing: {str(e)}"
        logger.error(f"Job {job_id} failed with exception: {error_message}")
        
        try:
            await training_plan_job_service.handle_job_failure(job_id, error_message)
        except Exception as cleanup_error:
            logger.error(f"Failed to handle job failure for {job_id}: {str(cleanup_error)}")


# ============================================================================
# WEBHOOK ENDPOINTS (Phase 2)
# ============================================================================

@app.post("/webhook/garmin")
async def garmin_webhook(request: Request):
    """
    Garmin webhook endpoint for receiving activity notifications.
    
    Handles real-time notifications for activity uploads, updates, and deletions.
    Implements signature verification and async processing.
    """
    try:
        logger.info(f"🔔 WEBHOOK RECEIVED: Garmin webhook request incoming")
        
        # Apply rate limiting
        await webhook_rate_limit_middleware(request)
        
        logger.info("Received Garmin webhook request")
        
        # Verify webhook signature and get payload
        payload_bytes = await webhook_verifier.verify_request(request)
        
        # Parse JSON payload
        try:
            import json
            payload_dict = json.loads(payload_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Invalid webhook payload format: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Validate payload with Pydantic
        try:
            webhook_payload = GarminWebhookPayload(**payload_dict)
        except Exception as e:
            logger.error(f"Webhook payload validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid payload structure: {str(e)}")
        
        unified_activities = webhook_payload.get_unified_activities()
        activity_count = len(unified_activities)
        logger.info(f"Processing webhook with {activity_count} activities from {'activities' if webhook_payload.activities else 'activityDetails'} format")
        
        # Create webhook events for all activities and queue for processing
        webhook_event_ids = await webhook_service.create_webhook_event(webhook_payload)
        
        # Events have been queued for background processing
        logger.info(f"Created {len(webhook_event_ids)} webhook events: {webhook_event_ids}")
        
        # Return immediate success (required within 2 seconds per Garmin docs)
        return {
            "success": True,
            "message": f"Webhook received with {activity_count} activities, created {len(webhook_event_ids)} events",
            "event_ids": webhook_event_ids,
            "activities_processed": activity_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal webhook processing error")


@app.get("/health/webhook", response_model=WebhookHealthStatus)
async def webhook_health():
    """Get webhook infrastructure health status."""
    try:
        health_status = await webhook_service.get_health_status()
        
        # Add queue status (simplified for testing)
        try:
            queue_status = await queue_service.get_queue_status()
            # Create a new health status with updated values
            health_status = WebhookHealthStatus(
                webhook_endpoint=health_status.webhook_endpoint,
                queue_system=queue_status["running"],
                workers=queue_status["workers_active"] > 0,
                database=health_status.database,
                last_event_received=health_status.last_event_received,
                pending_events=health_status.pending_events,
                failed_events=health_status.failed_events
            )
        except Exception as e:
            logger.warning(f"Failed to get queue status: {str(e)}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to get webhook health status: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/health/queue")
async def queue_health():
    """Get queue system health and metrics."""
    try:
        queue_status = await queue_service.get_queue_status()
        webhook_metrics = await webhook_service.get_webhook_metrics()
        
        return {
            "queue_status": queue_status,
            "webhook_metrics": webhook_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get queue health: {str(e)}")
        raise HTTPException(status_code=500, detail="Queue health check failed")


@app.get("/webhook/metrics", response_model=WebhookMetrics)
async def webhook_metrics():
    """Get webhook processing metrics."""
    try:
        return await webhook_service.get_webhook_metrics()
    except Exception as e:
        logger.error(f"Failed to get webhook metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Metrics retrieval failed")


# Debug and Testing Endpoints
@app.post("/api/debug/trigger-threshold-estimation/{user_id}")
async def trigger_threshold_estimation_debug(user_id: str):
    """Debug endpoint to manually trigger threshold estimation with detailed diagnostics."""
    try:
        from app.services.threshold_estimator import ThresholdEstimator
        from sqlalchemy import text
        from datetime import datetime, timedelta
        
        logger.info(f"Starting detailed threshold estimation debug for user {user_id}")
        
        # Step 1: Check activity count
        cutoff_date = datetime.now() - timedelta(days=180)
        
        async with mysql_service.async_session() as session:
            # Count activities
            count_query = text("""
                SELECT COUNT(*) as activity_count
                FROM activities 
                WHERE user_id = :user_id 
                AND data_source = 'garmin'
                AND processed_at >= :cutoff_date
            """)
            
            result = await session.execute(count_query, {
                "user_id": user_id,
                "cutoff_date": cutoff_date
            })
            
            activity_count = result.fetchone().activity_count
            logger.info(f"Found {activity_count} activities for threshold estimation")
            
            if activity_count < 10:
                return {
                    "success": False,
                    "error": f"Not enough activities for threshold estimation. Found: {activity_count}, Need: 10+",
                    "user_id": user_id,
                    "activity_count": activity_count
                }
            
            # Step 2: Get activities
            activities_query = text("""
                SELECT activity_data, processed_at
                FROM activities 
                WHERE user_id = :user_id 
                AND data_source = 'garmin'
                AND processed_at >= :cutoff_date
                ORDER BY processed_at DESC
                LIMIT 200
            """)
            
            result = await session.execute(activities_query, {
                "user_id": user_id,
                "cutoff_date": cutoff_date
            })
            
            activities = []
            sample_activity_types = []
            for row in result.fetchall():
                if row.activity_data:
                    # Handle JSON data properly
                    if isinstance(row.activity_data, str):
                        import json
                        activity_dict = json.loads(row.activity_data)
                    else:
                        activity_dict = row.activity_data
                    activities.append(activity_dict)
                    sample_activity_types.append(activity_dict.get('activityType', 'Unknown'))
            
            logger.info(f"Loaded {len(activities)} activities for estimation")
            logger.info(f"Activity types: {set(sample_activity_types)}")
            
            # Step 3: Test threshold estimator
            try:
                estimator = ThresholdEstimator()
                estimated_thresholds = estimator.estimate_all_thresholds(activities, data_source="garmin")
                
                logger.info(f"Threshold estimation successful")
                logger.info(f"FTP: {estimated_thresholds.ftp}W")
                logger.info(f"Running pace: {estimated_thresholds.running_threshold_pace} min/km")
                logger.info(f"Swimming pace: {estimated_thresholds.swimming_threshold_pace} min/100m")
                logger.info(f"LTHR: {estimated_thresholds.lthr} bpm")
                logger.info(f"Max HR: {estimated_thresholds.max_hr} bpm")
                
                # Step 4: Save thresholds
                thresholds_dict = {
                    'ftp': estimated_thresholds.ftp,
                    'running_threshold_pace': estimated_thresholds.running_threshold_pace,
                    'swimming_threshold_pace': estimated_thresholds.swimming_threshold_pace,
                    'lthr': estimated_thresholds.lthr,
                    'max_hr': estimated_thresholds.max_hr
                }
                
                # Calculate confidence (simplified for debug)
                confidence = {
                    'ftp': 'medium' if estimated_thresholds.ftp else 'estimated',
                    'running_threshold_pace': 'medium' if estimated_thresholds.running_threshold_pace else 'estimated',
                    'swimming_threshold_pace': 'medium' if estimated_thresholds.swimming_threshold_pace else 'estimated',
                    'lthr': 'medium' if estimated_thresholds.lthr else 'estimated',
                    'max_hr': 'medium' if estimated_thresholds.max_hr else 'estimated'
                }
                
                await mysql_service.save_user_thresholds(user_id, thresholds_dict, confidence)
                logger.info(f"Thresholds saved successfully")
                
                return {
                    "success": True,
                    "message": f"Threshold estimation completed successfully",
                    "user_id": user_id,
                    "activity_count": len(activities),
                    "activity_types": list(set(sample_activity_types)),
                    "estimated_thresholds": {
                        "ftp": estimated_thresholds.ftp,
                        "running_threshold_pace": estimated_thresholds.running_threshold_pace,
                        "swimming_threshold_pace": estimated_thresholds.swimming_threshold_pace,
                        "lthr": estimated_thresholds.lthr,
                        "max_hr": estimated_thresholds.max_hr
                    },
                    "confidence_levels": confidence
                }
                
            except Exception as threshold_error:
                logger.error(f"Threshold estimation failed: {str(threshold_error)}")
                import traceback
                traceback.print_exc()
                
                return {
                    "success": False,
                    "error": f"Threshold estimation failed: {str(threshold_error)}",
                    "user_id": user_id,
                    "activity_count": len(activities),
                    "activity_types": list(set(sample_activity_types))
                }
        
    except Exception as e:
        logger.error(f"Debug threshold estimation failed for user {user_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False, 
            "error": str(e),
            "user_id": user_id
        }

# Garmin Activity Backfill Endpoints
@app.post("/api/garmin/backfill/historical/{user_id}")
async def start_configurable_backfill(user_id: str, days: int = 60):
    """
    Start historical activity backfill for a user with configurable days.
    
    Uses Garmin's Summary Backfill API to request specified number of days of activities.
    Data is delivered asynchronously via webhook notifications.
    
    Args:
        user_id: Internal user ID
        days: Number of days to backfill (1-60, default: 60)
        
    Returns:
        Job details with status and tracking information
    """
    try:
        # Validate days parameter
        if days < 1 or days > 60:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 60")
            
        logger.info(f"Starting {days}-day historical backfill for user {user_id}")
        
        from app.services.garmin_backfill_service import garmin_backfill_service
        
        result = await garmin_backfill_service.initiate_backfill(user_id, days_back=days)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start {days}-day backfill for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start backfill: {str(e)}")


@app.get("/api/garmin/backfill/{user_id}/status")
async def get_backfill_status(user_id: str, job_id: Optional[int] = None):
    """
    Get backfill job status for a user.
    
    Args:
        user_id: Internal user ID
        job_id: Optional specific job ID (returns latest if not provided)
        
    Returns:
        Job status and progress information
    """
    try:
        from app.services.garmin_backfill_service import garmin_backfill_service
        
        result = await garmin_backfill_service.get_backfill_status(user_id, job_id)
        return result
        
    except Exception as e:
        logger.error(f"Failed to get backfill status for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.post("/api/garmin/backfill/{user_id}")
async def start_activity_backfill_deprecated(
    user_id: str,
    days_back: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    DEPRECATED: Legacy backfill endpoint.
    Use POST /api/garmin/backfill/historical/{user_id} for 60-day sync instead.
    """
    return {
        "success": False,
        "message": "This endpoint is deprecated. Use the new 60-day backfill endpoint.",
        "user_id": user_id,
        "recommended_endpoint": f"/api/garmin/backfill/historical/{user_id}",
        "recommended_method": "POST",
        "note": "New endpoint uses Garmin's official Summary Backfill API and delivers data via webhooks"
    }


@app.post("/api/garmin/sync/{user_id}")
async def sync_single_activity(user_id: str, request: Dict[str, Any] = Body(...)):
    """
    Directly sync a single Garmin activity (bypass Celery for testing).
    """
    try:
        activity_ids = request.get('activity_ids', [])
        
        if not activity_ids:
            raise HTTPException(status_code=400, detail="activity_ids required")
            
        from app.services.garmin_api import GarminAPIClient
        
        # Get user tokens
        query = "SELECT garmin_access_token FROM users WHERE id = %s"
        result = await mysql_service.fetch_all(query, (user_id,))
        
        if not result or not result[0].get('garmin_access_token'):
            raise HTTPException(status_code=404, detail="User not found or not authenticated with Garmin")
            
        access_token = result[0]['garmin_access_token']
        
        # Initialize Garmin client
        garmin_client = GarminAPIClient()
        
        processed_activities = []
        
        for activity_id in activity_ids:
            try:
                logger.info(f"Fetching Garmin activity {activity_id}")
                
                # Fetch activity from Garmin
                activity_data = await garmin_client.get_activity_details(activity_id, user_id)
                
                if activity_data:
                    logger.info(f"Storing activity: {activity_data.get('activityName', 'Unknown')}")
                    
                    # Store in database
                    query = """
                    INSERT INTO activities (
                        user_id, activity_id, name, activity_type, start_time,
                        distance, duration, data_source, raw_data, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                        name = VALUES(name),
                        distance = VALUES(distance),
                        duration = VALUES(duration),
                        raw_data = VALUES(raw_data),
                        updated_at = NOW()
                    """
                    
                    values = (
                        user_id,
                        activity_id,
                        activity_data.get('activityName', 'Unknown'),
                        activity_data.get('activityType', 'UNKNOWN'),
                        activity_data.get('startTimeLocal', '2025-01-01T00:00:00'),
                        activity_data.get('distance', 0),
                        activity_data.get('duration', 0),
                        'garmin',
                        json.dumps(activity_data)
                    )
                    
                    await mysql_service.execute(query, values)
                    processed_activities.append(activity_id)
                    logger.info(f"✅ Successfully stored activity {activity_id}")
                else:
                    logger.warning(f"No data returned for activity {activity_id}")
                    
            except Exception as e:
                logger.error(f"Failed to process activity {activity_id}: {e}")
                
        return {
            "success": True,
            "message": f"Processed {len(processed_activities)} activities",
            "user_id": user_id,
            "processed_activity_ids": processed_activities,
            "total_requested": len(activity_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync activities for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)