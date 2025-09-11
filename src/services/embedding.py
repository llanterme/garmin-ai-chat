"""Embedding service for generating multi-vector embeddings from Garmin activity data."""

import os
from datetime import datetime
from typing import Any, Dict, List

import openai
from pydantic import BaseModel

from ..core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536


class EmbeddingResult(BaseModel):
    """Result from embedding generation."""
    vector_type: str
    text: str
    embedding: List[float]


class EmbeddingService:
    """Service for generating multi-vector embeddings using OpenAI API."""
    
    def __init__(self):
        from ..core.config import settings
        
        self.api_key = settings.openai_api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.config = EmbeddingConfig()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        try:
            response = self.client.embeddings.create(
                model=self.config.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch for efficiency."""
        try:
            response = self.client.embeddings.create(
                model=self.config.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    def create_multi_vector_embeddings(self, activity_data: Dict[str, Any]) -> Dict[str, str]:
        """Creates 4 different summaries for multi-vector approach."""
        return {
            "main": self._create_activity_summary(activity_data),
            "metrics": self._create_metrics_summary(activity_data),
            "temporal": self._create_temporal_summary(activity_data),
            "performance": self._create_performance_summary(activity_data)
        }
    
    def _create_activity_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create comprehensive human-readable activity summary with derived metrics."""
        # Extract basic info
        activity_type = activity_data.get("activity_type", "Activity")
        activity_name = activity_data.get("activity_name", "Untitled Activity")
        
        # Date and time
        start_time = activity_data.get("start_time")
        if start_time:
            if isinstance(start_time, str):
                date_str = start_time.split(" ")[0]
            else:
                date_str = start_time.strftime("%Y-%m-%d")
        else:
            date_str = "Unknown date"
        
        # Basic metrics
        distance_km = None
        if activity_data.get("distance"):
            distance_km = round(activity_data["distance"] / 1000, 2)
        
        duration_min = None
        if activity_data.get("duration"):
            duration_min = round(activity_data["duration"] / 60, 1)
        
        # Performance metrics
        avg_speed_kmh = None
        if activity_data.get("average_speed"):
            avg_speed_kmh = round(activity_data["average_speed"] * 3.6, 1)
        
        elevation_gain = activity_data.get("elevation_gain")
        avg_hr = activity_data.get("average_heart_rate")
        max_hr = activity_data.get("max_heart_rate")
        avg_power = activity_data.get("average_power")
        max_power = activity_data.get("max_power")
        
        # Build comprehensive summary
        summary_parts = [
            f"{activity_type}: {activity_name}",
            f"Date: {date_str}"
        ]
        
        if distance_km:
            summary_parts.append(f"Distance: {distance_km} km")
        
        if duration_min:
            summary_parts.append(f"Duration: {duration_min} minutes")
        
        if avg_speed_kmh:
            summary_parts.append(f"Average speed: {avg_speed_kmh} km/h")
        
        if elevation_gain:
            summary_parts.append(f"Elevation gain: {elevation_gain}m")
        
        if avg_hr:
            summary_parts.append(f"Average heart rate: {avg_hr} bpm")
            
        if max_hr:
            summary_parts.append(f"Max heart rate: {max_hr} bpm")
        
        if avg_power:
            summary_parts.append(f"Average power: {avg_power}W")
            
        if max_power:
            summary_parts.append(f"Max power: {max_power}W")
        
        # Add derived metrics to summary
        pace_formatted = activity_data.get('pace_per_km_formatted')
        if pace_formatted:
            summary_parts.append(f"Pace: {pace_formatted} /km")
        
        watts_per_kg = activity_data.get('watts_per_kg')
        if watts_per_kg:
            summary_parts.append(f"Power-to-weight: {watts_per_kg} W/kg")
        
        training_load = activity_data.get('training_load')
        if training_load:
            summary_parts.append(f"Training load: {training_load}")
        
        effort_level = activity_data.get('effort_level')
        if effort_level:
            summary_parts.append(f"Effort: {effort_level}")
        
        if activity_data.get('is_personal_record'):
            summary_parts.append("Personal Record!")
        
        return ". ".join(summary_parts) + "."
    
    def _create_metrics_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create metrics-focused summary for performance analysis."""
        metrics_parts = ["Activity metrics:"]
        
        # Distance metrics
        distance = activity_data.get("distance")
        if distance:
            distance_km = round(distance / 1000, 2)
            metrics_parts.append(f"distance {distance_km} kilometers")
        
        # Time metrics
        duration = activity_data.get("duration")
        if duration:
            duration_min = round(duration / 60, 1)
            metrics_parts.append(f"duration {duration_min} minutes")
        
        # Speed metrics
        avg_speed = activity_data.get("average_speed")
        if avg_speed:
            avg_speed_kmh = round(avg_speed * 3.6, 1)
            metrics_parts.append(f"average speed {avg_speed_kmh} kilometers per hour")
        
        max_speed = activity_data.get("max_speed")
        if max_speed:
            max_speed_kmh = round(max_speed * 3.6, 1)
            metrics_parts.append(f"maximum speed {max_speed_kmh} kilometers per hour")
        
        # Heart rate metrics
        avg_hr = activity_data.get("average_heart_rate")
        if avg_hr:
            metrics_parts.append(f"average heart rate {avg_hr} beats per minute")
        
        max_hr = activity_data.get("max_heart_rate")
        if max_hr:
            metrics_parts.append(f"maximum heart rate {max_hr} beats per minute")
        
        # Power metrics
        avg_power = activity_data.get("average_power")
        if avg_power:
            metrics_parts.append(f"average power {avg_power} watts")
        
        max_power = activity_data.get("max_power")
        if max_power:
            metrics_parts.append(f"maximum power {max_power} watts")
        
        # Elevation metrics
        elevation_gain = activity_data.get("elevation_gain")
        if elevation_gain:
            metrics_parts.append(f"elevation gain {elevation_gain} meters")
        
        # Calories
        calories = activity_data.get("calories")
        if calories:
            metrics_parts.append(f"calories burned {calories}")
        
        # Add derived metrics
        pace_per_km = activity_data.get('pace_per_km')
        if pace_per_km:
            metrics_parts.append(f"pace {pace_per_km} minutes per kilometer")
        
        watts_per_kg = activity_data.get('watts_per_kg')
        if watts_per_kg:
            metrics_parts.append(f"power to weight ratio {watts_per_kg} watts per kilogram")
        
        calculated_tss = activity_data.get('calculated_tss')
        if calculated_tss:
            metrics_parts.append(f"training stress score {calculated_tss}")
        
        primary_hr_zone = activity_data.get('primary_hr_zone')
        hr_zone_name = activity_data.get('hr_zone_name')
        if primary_hr_zone and hr_zone_name:
            metrics_parts.append(f"primary heart rate zone {primary_hr_zone} {hr_zone_name}")
        
        stride_length = activity_data.get('stride_length_m')
        if stride_length:
            metrics_parts.append(f"stride length {stride_length} meters")
        
        efficiency_factor = activity_data.get('efficiency_factor')
        if efficiency_factor:
            metrics_parts.append(f"efficiency factor {efficiency_factor}")
        
        return " ".join(metrics_parts) + "."
    
    def _create_temporal_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create temporal-focused summary with multiple date formats."""
        start_time = activity_data.get("start_time")
        activity_type = activity_data.get("activity_type", "activity")
        
        if not start_time:
            return f"{activity_type} activity with unknown date"
        
        # Parse datetime if it's a string
        if isinstance(start_time, str):
            try:
                dt = datetime.fromisoformat(start_time.replace('T', ' ').replace('Z', ''))
            except:
                return f"{activity_type} activity on {start_time}"
        else:
            dt = start_time
        
        # Create multiple temporal references
        date_str = dt.strftime("%Y-%m-%d")
        day_name = dt.strftime("%A")
        month_name = dt.strftime("%B")
        year = dt.year
        
        # Calculate relative time
        now = datetime.now()
        days_ago = (now - dt).days
        
        temporal_parts = [
            f"Activity on {date_str}",
            f"performed on {day_name}",
            f"in {month_name} {year}"
        ]
        
        if days_ago == 0:
            temporal_parts.append("activity today")
        elif days_ago == 1:
            temporal_parts.append("activity yesterday")
        elif days_ago <= 7:
            temporal_parts.append(f"activity {days_ago} days ago")
        elif days_ago <= 30:
            weeks_ago = days_ago // 7
            if weeks_ago == 1:
                temporal_parts.append("activity last week")
            else:
                temporal_parts.append(f"activity {weeks_ago} weeks ago")
        elif days_ago <= 365:
            months_ago = days_ago // 30
            if months_ago == 1:
                temporal_parts.append("activity last month")
            else:
                temporal_parts.append(f"activity {months_ago} months ago")
        
        return ", ".join(temporal_parts) + "."
    
    def _create_performance_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create performance analysis summary with derived metrics."""
        activity_type = activity_data.get("activity_type", "activity")
        performance_parts = [f"Performance analysis for {activity_type}"]
        
        # Use derived pace if available, otherwise calculate
        pace_formatted = activity_data.get('pace_per_km_formatted')
        if pace_formatted and activity_type in ["running", "treadmill_running"]:
            performance_parts.append(f"pace {pace_formatted} per kilometer")
        elif activity_data.get("average_speed"):
            avg_speed = activity_data.get("average_speed")
            speed_kmh = round(avg_speed * 3.6, 1)
            if activity_type == "running":
                pace_per_km = round(60 / speed_kmh, 2) if speed_kmh > 0 else None
                if pace_per_km:
                    performance_parts.append(f"pace {pace_per_km} minutes per kilometer")
            else:
                performance_parts.append(f"speed {speed_kmh} kilometers per hour")
        
        # Power analysis
        avg_power = activity_data.get("average_power")
        if avg_power:
            if avg_power > 300:
                power_level = "high power output"
            elif avg_power > 200:
                power_level = "moderate power output"
            elif avg_power > 100:
                power_level = "low power output"
            else:
                power_level = "minimal power output"
            performance_parts.append(f"{power_level} performance")
        
        # Heart rate zones analysis
        avg_hr = activity_data.get("average_heart_rate")
        max_hr = activity_data.get("max_heart_rate")
        if avg_hr:
            if avg_hr > 160:
                hr_intensity = "high intensity heart rate"
            elif avg_hr > 140:
                hr_intensity = "moderate intensity heart rate"
            elif avg_hr > 120:
                hr_intensity = "low intensity heart rate"
            else:
                hr_intensity = "recovery level heart rate"
            performance_parts.append(hr_intensity)
        
        # Duration analysis
        duration = activity_data.get("duration")
        if duration:
            duration_min = duration / 60
            if duration_min > 120:
                duration_category = "long duration endurance"
            elif duration_min > 60:
                duration_category = "medium duration"
            elif duration_min > 30:
                duration_category = "short duration"
            else:
                duration_category = "brief session"
            performance_parts.append(duration_category)
        
        # Add derived performance indicators
        if activity_data.get('is_personal_record'):
            performance_parts.append("personal record achievement")
        
        if activity_data.get('is_interval_workout'):
            performance_parts.append("interval training session")
            
        effort_level = activity_data.get('effort_level')
        if effort_level:
            performance_parts.append(f"{effort_level} effort intensity")
        
        relative_effort = activity_data.get('relative_effort')
        if relative_effort:
            performance_parts.append(f"relative effort {relative_effort} out of 10")
        
        training_load = activity_data.get('training_load')
        if training_load:
            performance_parts.append(f"training load {training_load}")
        
        return ", ".join(performance_parts) + "."
    
    async def process_activity_embeddings(self, activity_data: Dict[str, Any]) -> Dict[str, EmbeddingResult]:
        """Process an activity to generate all 4 vector embeddings."""
        try:
            # Generate summaries
            summaries = self.create_multi_vector_embeddings(activity_data)
            
            # Generate embeddings in batch for efficiency
            texts = list(summaries.values())
            embeddings = await self.generate_embeddings_batch(texts)
            
            # Create results
            results = {}
            vector_types = ["main", "metrics", "temporal", "performance"]
            
            for i, vector_type in enumerate(vector_types):
                results[vector_type] = EmbeddingResult(
                    vector_type=vector_type,
                    text=summaries[vector_type],
                    embedding=embeddings[i]
                )
            
            logger.info(f"Generated embeddings for activity {activity_data.get('garmin_activity_id', 'unknown')}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process activity embeddings: {str(e)}")
            raise