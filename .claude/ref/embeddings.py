"""
Embedding service for generating vector embeddings using OpenAI API.
"""

import os
from typing import List, Dict, Any
import openai


class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"  # 1536 dimensions, cost-effective
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings corresponding to input texts
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    def create_activity_summary(self, activity_data: Dict[str, Any], data_source: str = "strava") -> str:
        """
        Create a comprehensive human-readable summary of an activity.
        
        Args:
            activity_data: Raw activity data from Strava or Garmin API
            data_source: Data source ("strava" or "garmin")
            
        Returns:
            Human-readable summary suitable for embedding
        """
        if data_source.lower() == "garmin":
            return self._create_garmin_activity_summary(activity_data)
        else:
            return self._create_strava_activity_summary(activity_data)
    
    def _create_strava_activity_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create summary for Strava activity data (existing logic)."""
        activity_type = activity_data.get("type", "Activity")
        name = activity_data.get("name", "Untitled Activity")
        
        # Date and time
        start_date = activity_data.get("start_date_local", "").split("T")[0]
        
        # Basic metrics
        distance_km = round(activity_data.get("distance", 0) / 1000, 2) if activity_data.get("distance") else None
        moving_time_min = round(activity_data.get("moving_time", 0) / 60, 1) if activity_data.get("moving_time") else None
        elapsed_time_min = round(activity_data.get("elapsed_time", 0) / 60, 1) if activity_data.get("elapsed_time") else None
        
        # Elevation
        elevation_gain = activity_data.get("total_elevation_gain", 0)
        
        # Performance metrics
        avg_speed_kmh = round(activity_data.get("average_speed", 0) * 3.6, 1) if activity_data.get("average_speed") else None
        max_speed_kmh = round(activity_data.get("max_speed", 0) * 3.6, 1) if activity_data.get("max_speed") else None
        
        # Heart rate
        avg_heartrate = activity_data.get("average_heartrate")
        max_heartrate = activity_data.get("max_heartrate")
        
        # Power (for cycling)
        avg_watts = activity_data.get("average_watts")
        max_watts = activity_data.get("max_watts")
        
        # Perceived effort
        perceived_exertion = activity_data.get("perceived_exertion")
        
        # Build summary
        summary_parts = [
            f"{activity_type}: {name}",
            f"Date: {start_date}"
        ]
        
        if distance_km:
            summary_parts.append(f"Distance: {distance_km} km")
        
        if moving_time_min:
            summary_parts.append(f"Moving time: {moving_time_min} minutes")
        
        if elapsed_time_min and elapsed_time_min != moving_time_min:
            summary_parts.append(f"Total time: {elapsed_time_min} minutes")
        
        if elevation_gain:
            summary_parts.append(f"Elevation gain: {elevation_gain} meters")
        
        if avg_speed_kmh:
            summary_parts.append(f"Average speed: {avg_speed_kmh} km/h")
        
        if max_speed_kmh:
            summary_parts.append(f"Max speed: {max_speed_kmh} km/h")
        
        # Calculate pace for running activities
        if activity_type.lower() in ["run", "running"] and distance_km and moving_time_min:
            pace_min_per_km = moving_time_min / distance_km
            pace_minutes = int(pace_min_per_km)
            pace_seconds = int((pace_min_per_km - pace_minutes) * 60)
            summary_parts.append(f"Average pace: {pace_minutes}:{pace_seconds:02d} min/km")
        
        if avg_heartrate:
            summary_parts.append(f"Average heart rate: {avg_heartrate} bpm")
        
        if max_heartrate:
            summary_parts.append(f"Max heart rate: {max_heartrate} bpm")
        
        if avg_watts:
            summary_parts.append(f"Average power: {avg_watts} watts")
        
        if max_watts:
            summary_parts.append(f"Max power: {max_watts} watts")
        
        if perceived_exertion:
            summary_parts.append(f"Perceived exertion: {perceived_exertion}/10")
        
        # Add location if available
        if activity_data.get("location_city") and activity_data.get("location_country"):
            summary_parts.append(f"Location: {activity_data['location_city']}, {activity_data['location_country']}")
        
        # Add description if available
        description = activity_data.get("description")
        if description:
            summary_parts.append(f"Notes: {description}")
        
        return ". ".join(summary_parts)
    
    def _create_garmin_activity_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create summary for Garmin activity data."""
        activity_type = activity_data.get("activityType", "Activity")
        activity_name = activity_data.get("activityName", "Untitled Activity")
        
        # Date and time
        start_date = activity_data.get("startTimeLocal", "").split("T")[0]
        
        # Basic metrics
        distance_m = activity_data.get("distance", 0)
        distance_km = round(distance_m / 1000, 2) if distance_m else None
        duration_seconds = activity_data.get("duration", 0)
        duration_min = round(duration_seconds / 60, 1) if duration_seconds else None
        
        # Elevation
        elevation_gain = activity_data.get("elevationGain", 0)
        
        # Performance metrics
        avg_speed_ms = activity_data.get("averageSpeed", 0)
        avg_speed_kmh = round(avg_speed_ms * 3.6, 1) if avg_speed_ms else None
        max_speed_ms = activity_data.get("maxSpeed", 0)
        max_speed_kmh = round(max_speed_ms * 3.6, 1) if max_speed_ms else None
        
        # Heart rate
        avg_heartrate = activity_data.get("averageHeartRate")
        max_heartrate = activity_data.get("maxHeartRate")
        
        # Power (for cycling)
        avg_watts = activity_data.get("averagePower")
        max_watts = activity_data.get("maxPower")
        normalized_power = activity_data.get("normalizedPower")
        
        # Garmin-specific metrics
        intensity_factor = activity_data.get("intensityFactor")
        training_stress_score = activity_data.get("trainingStressScore")
        
        # Build summary
        summary_parts = [
            f"{activity_type}: {activity_name}",
            f"Date: {start_date}"
        ]
        
        if distance_km:
            summary_parts.append(f"Distance: {distance_km} km")
        
        if duration_min:
            summary_parts.append(f"Duration: {duration_min} minutes")
        
        if elevation_gain:
            summary_parts.append(f"Elevation gain: {elevation_gain} meters")
        
        if avg_speed_kmh:
            summary_parts.append(f"Average speed: {avg_speed_kmh} km/h")
        
        if max_speed_kmh:
            summary_parts.append(f"Max speed: {max_speed_kmh} km/h")
        
        # Calculate pace for running activities
        if activity_type.upper() in ["RUNNING", "TRAIL_RUNNING"] and distance_km and duration_min:
            pace_min_per_km = duration_min / distance_km
            pace_minutes = int(pace_min_per_km)
            pace_seconds = int((pace_min_per_km - pace_minutes) * 60)
            summary_parts.append(f"Average pace: {pace_minutes}:{pace_seconds:02d} min/km")
        
        if avg_heartrate:
            summary_parts.append(f"Average heart rate: {avg_heartrate} bpm")
        
        if max_heartrate:
            summary_parts.append(f"Max heart rate: {max_heartrate} bpm")
        
        if avg_watts:
            summary_parts.append(f"Average power: {avg_watts} watts")
        
        if max_watts:
            summary_parts.append(f"Max power: {max_watts} watts")
        
        if normalized_power:
            summary_parts.append(f"Normalized power: {normalized_power} watts")
        
        if intensity_factor:
            summary_parts.append(f"Intensity factor: {intensity_factor:.2f}")
        
        if training_stress_score:
            summary_parts.append(f"Training stress score: {training_stress_score}")
        
        # Add description if available
        description = activity_data.get("description")
        if description:
            summary_parts.append(f"Notes: {description}")
        
        return ". ".join(summary_parts)
    
    def extract_activity_metadata(self, activity_data: Dict[str, Any], data_source: str = "strava") -> Dict[str, Any]:
        """
        Extract comprehensive metadata from activity data for vector storage and filtering.
        
        Args:
            activity_data: Raw activity data from Strava or Garmin API
            data_source: Data source ("strava" or "garmin")
            
        Returns:
            Dictionary of rich metadata for vector storage
        """
        if data_source.lower() == "garmin":
            return self._extract_garmin_activity_metadata(activity_data)
        else:
            return self._extract_strava_activity_metadata(activity_data)
    
    def _extract_strava_activity_metadata(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for Strava activity data (existing logic)."""
        from datetime import datetime
        
        # Parse start date
        start_date_str = activity_data.get("start_date_local", "")
        start_date = None
        timestamp = None
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                timestamp = int(start_date.timestamp())
            except ValueError:
                start_date = datetime.now()
                timestamp = int(start_date.timestamp())
        
        # Calculate derived metrics
        distance = activity_data.get("distance", 0)
        moving_time = activity_data.get("moving_time", 0)
        elapsed_time = activity_data.get("elapsed_time", 0)
        average_speed = activity_data.get("average_speed", 0)
        
        # Convert to standard units
        distance_km = distance / 1000 if distance else 0
        duration_minutes = moving_time / 60 if moving_time else 0
        duration_hours = duration_minutes / 60 if duration_minutes else 0
        avg_speed_kmh = average_speed * 3.6 if average_speed else 0
        max_speed_kmh = activity_data.get("max_speed", 0) * 3.6 if activity_data.get("max_speed") else 0
        
        # Calculate pace (min/km) for running activities
        avg_pace_min_per_km = None
        if distance_km > 0 and duration_minutes > 0:
            avg_pace_min_per_km = duration_minutes / distance_km
        
        # Power metrics
        avg_power = activity_data.get("average_watts", 0) or 0
        max_power = activity_data.get("max_watts", 0) or 0
        
        # Heart rate metrics
        avg_heartrate = activity_data.get("average_heartrate", 0) or 0
        max_heartrate = activity_data.get("max_heartrate", 0) or 0
        
        # Elevation metrics
        elevation_gain = activity_data.get("total_elevation_gain", 0) or 0
        
        # Training metrics
        suffer_score = activity_data.get("suffer_score", 0) or 0
        perceived_exertion = activity_data.get("perceived_exertion", 0) or 0
        
        # Calculate training zones (basic estimation)
        # These would be more accurate with user's specific zones
        time_in_zones = self._estimate_training_zones(avg_heartrate, max_heartrate, duration_minutes)
        
        return {
            # Basic info
            "activity_type": activity_data.get("type", "").lower(),
            "name": activity_data.get("name", ""),
            "description": activity_data.get("description", ""),
            
            # Date and time
            "date": start_date.strftime("%Y-%m-%d") if start_date else "",
            "start_date": start_date_str,
            "timestamp": timestamp,
            "day_of_week": start_date.strftime("%A").lower() if start_date else "",
            "month": start_date.strftime("%B").lower() if start_date else "",
            "year": start_date.year if start_date else 0,
            
            # Distance and duration
            "distance": distance,
            "distance_km": distance_km,
            "moving_time": moving_time,
            "elapsed_time": elapsed_time,
            "duration_minutes": duration_minutes,
            "duration_hours": duration_hours,
            
            # Speed and pace
            "average_speed": average_speed,
            "max_speed": activity_data.get("max_speed", 0) or 0,
            "avg_speed_kmh": avg_speed_kmh,
            "max_speed_kmh": max_speed_kmh,
            "avg_pace_min_per_km": avg_pace_min_per_km,
            
            # Power metrics
            "average_watts": avg_power,
            "max_watts": max_power,
            "avg_power_watts": avg_power,
            "max_power_watts": max_power,
            "normalized_power": activity_data.get("normalized_power"),
            "intensity_factor": activity_data.get("intensity_factor"),
            "variability_index": activity_data.get("variability_index"),
            "has_power_data": avg_power > 0,
            
            # Heart rate metrics
            "average_heartrate": avg_heartrate,
            "max_heartrate": max_heartrate,
            "avg_heartrate": avg_heartrate,
            "max_heartrate_bpm": max_heartrate,
            "has_heartrate": activity_data.get("has_heartrate", False) or avg_heartrate > 0,
            
            # Elevation
            "total_elevation_gain": elevation_gain,
            "elevation_gain_m": elevation_gain,
            
            # Training metrics
            "suffer_score": suffer_score,
            "perceived_exertion": perceived_exertion,
            "workout_type": activity_data.get("workout_type"),
            
            # Training zones (estimated)
            **time_in_zones,
            
            # Other metrics
            "average_cadence": activity_data.get("average_cadence", 0) or 0,
            "max_cadence": activity_data.get("max_cadence", 0) or 0,
            "kilojoules": activity_data.get("kilojoules", 0) or 0,
            "calories": activity_data.get("calories", 0) or 0,
            "efficiency_factor": activity_data.get("efficiency_factor"),
            "grade_adjusted_pace": activity_data.get("grade_adjusted_pace"),
            
            # Location
            "location_city": activity_data.get("location_city", ""),
            "location_country": activity_data.get("location_country", ""),
            "start_latitude": activity_data.get("start_latitude"),
            "start_longitude": activity_data.get("start_longitude"),
            
            # Activity flags
            "trainer": activity_data.get("trainer", False),
            "commute": activity_data.get("commute", False),
            "manual": activity_data.get("manual", False),
            "private": activity_data.get("private", False),
            "indoor": activity_data.get("trainer", False),
            "outdoor": not activity_data.get("trainer", False),
            
            # Equipment
            "gear_id": activity_data.get("gear_id", ""),
            
            # Social metrics
            "kudos_count": activity_data.get("kudos_count", 0),
            "comment_count": activity_data.get("comment_count", 0),
            "athlete_count": activity_data.get("athlete_count", 1),
            "photo_count": activity_data.get("photo_count", 0),
            "achievement_count": activity_data.get("achievement_count", 0),
            "pr_count": activity_data.get("pr_count", 0),
            
            # Data quality flags
            "has_gps": bool(activity_data.get("start_latitude") and activity_data.get("start_longitude")),
            "manual_entry": activity_data.get("manual", False),
            "device_name": activity_data.get("device_name", ""),
            
            # Performance indicators
            "intensity_score": self._calculate_intensity_score(avg_power, avg_heartrate, duration_minutes),
            "efficiency_score": self._calculate_efficiency_score(avg_speed_kmh, avg_heartrate, elevation_gain),
        }
    
    def _estimate_training_zones(self, avg_hr: float, max_hr: float, duration_min: float) -> Dict[str, float]:
        """
        Estimate time spent in training zones based on heart rate.
        This is a simplified estimation - real zone analysis would need user's specific zones.
        """
        if not avg_hr or not duration_min:
            return {
                "time_in_z1_min": 0,
                "time_in_z2_min": 0,
                "time_in_z3_min": 0,
                "time_in_z4_min": 0,
                "time_in_z5_min": 0,
            }
        
        # Rough zone estimation based on typical percentages of max HR
        # Zone 1: 50-60% max HR
        # Zone 2: 60-70% max HR  
        # Zone 3: 70-80% max HR
        # Zone 4: 80-90% max HR
        # Zone 5: 90-100% max HR
        
        # This is a simplified model - in reality you'd analyze the full HR stream
        estimated_max_hr = max_hr if max_hr > avg_hr else avg_hr * 1.1
        hr_percentage = (avg_hr / estimated_max_hr) * 100 if estimated_max_hr > 0 else 0
        
        # Distribute time based on average HR (simplified)
        if hr_percentage < 60:
            return {"time_in_z1_min": duration_min, "time_in_z2_min": 0, "time_in_z3_min": 0, "time_in_z4_min": 0, "time_in_z5_min": 0}
        elif hr_percentage < 70:
            return {"time_in_z1_min": 0, "time_in_z2_min": duration_min, "time_in_z3_min": 0, "time_in_z4_min": 0, "time_in_z5_min": 0}
        elif hr_percentage < 80:
            return {"time_in_z1_min": 0, "time_in_z2_min": 0, "time_in_z3_min": duration_min, "time_in_z4_min": 0, "time_in_z5_min": 0}
        elif hr_percentage < 90:
            return {"time_in_z1_min": 0, "time_in_z2_min": 0, "time_in_z3_min": 0, "time_in_z4_min": duration_min, "time_in_z5_min": 0}
        else:
            return {"time_in_z1_min": 0, "time_in_z2_min": 0, "time_in_z3_min": 0, "time_in_z4_min": 0, "time_in_z5_min": duration_min}
    
    def _calculate_intensity_score(self, avg_power: float, avg_hr: float, duration_min: float) -> float:
        """Calculate a basic intensity score for the activity."""
        if not duration_min:
            return 0.0
        
        # Simple intensity calculation based on available metrics
        intensity = 0.0
        
        if avg_power > 0:
            # Power-based intensity (simplified TSS-like calculation)
            intensity = (avg_power * duration_min) / 1000  # Normalized
        elif avg_hr > 0:
            # HR-based intensity
            intensity = (avg_hr * duration_min) / 10000  # Normalized
        
        return round(intensity, 2)
    
    def _calculate_efficiency_score(self, avg_speed: float, avg_hr: float, elevation: float) -> float:
        """Calculate efficiency score based on speed, HR, and elevation."""
        if not avg_speed or not avg_hr:
            return 0.0
        
        # Basic efficiency: speed per heart rate beat, adjusted for elevation
        base_efficiency = avg_speed / avg_hr if avg_hr > 0 else 0
        elevation_factor = 1 + (elevation / 1000) * 0.1  # Slight bonus for elevation
        
        return round(base_efficiency * elevation_factor, 4)
    
    def create_multi_vector_embeddings(self, activity_data: Dict[str, Any], data_source: str = "strava") -> Dict[str, str]:
        """
        Create multiple embeddings for different aspects of the activity.
        
        Args:
            activity_data: Raw activity data from Strava or Garmin API
            data_source: Data source ("strava" or "garmin")
            
        Returns:
            Dictionary of different summary texts for embedding
        """
        if data_source.lower() == "garmin":
            return self._create_garmin_multi_vector_embeddings(activity_data)
        else:
            return self._create_strava_multi_vector_embeddings(activity_data)
    
    def _create_strava_multi_vector_embeddings(self, activity_data: Dict[str, Any]) -> Dict[str, str]:
        """Create multi-vector embeddings for Strava activities (existing logic)."""
        # Main activity summary (existing)
        main_summary = self._create_strava_activity_summary(activity_data)
        
        # Metrics-focused summary
        metrics_summary = self._create_metrics_summary(activity_data)
        
        # Temporal-focused summary
        temporal_summary = self._create_temporal_summary(activity_data)
        
        # Performance-focused summary
        performance_summary = self._create_performance_summary(activity_data)
        
        return {
            "main": main_summary,
            "metrics": metrics_summary,
            "temporal": temporal_summary,
            "performance": performance_summary
        }
    
    def _create_metrics_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create a metrics-focused summary for better metric-based queries."""
        parts = []
        
        # Power metrics
        if activity_data.get("average_watts"):
            parts.append(f"average power {activity_data['average_watts']} watts")
        if activity_data.get("max_watts"):
            parts.append(f"maximum power {activity_data['max_watts']} watts")
        
        # Heart rate metrics
        if activity_data.get("average_heartrate"):
            parts.append(f"average heart rate {activity_data['average_heartrate']} bpm")
        if activity_data.get("max_heartrate"):
            parts.append(f"maximum heart rate {activity_data['max_heartrate']} bpm")
        
        # Speed metrics
        if activity_data.get("average_speed"):
            speed_kmh = activity_data["average_speed"] * 3.6
            parts.append(f"average speed {speed_kmh:.1f} km/h")
        
        # Distance and duration
        if activity_data.get("distance"):
            distance_km = activity_data["distance"] / 1000
            parts.append(f"distance {distance_km:.1f} kilometers")
        
        if activity_data.get("moving_time"):
            duration_min = activity_data["moving_time"] / 60
            parts.append(f"duration {duration_min:.0f} minutes")
        
        # Elevation
        if activity_data.get("total_elevation_gain"):
            parts.append(f"elevation gain {activity_data['total_elevation_gain']} meters")
        
        # Training metrics
        if activity_data.get("suffer_score"):
            parts.append(f"suffer score {activity_data['suffer_score']}")
        
        if activity_data.get("perceived_exertion"):
            parts.append(f"perceived exertion {activity_data['perceived_exertion']} out of 10")
        
        return f"Activity metrics: {', '.join(parts)}"
    
    def _create_temporal_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create a temporal-focused summary for better date-based queries."""
        from datetime import datetime
        
        start_date_str = activity_data.get("start_date_local", "")
        if not start_date_str:
            return "Activity with unknown date"
        
        try:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            
            parts = [
                f"Activity on {start_date.strftime('%Y-%m-%d')}",
                f"performed on {start_date.strftime('%A')}",
                f"in {start_date.strftime('%B %Y')}",
                f"day {start_date.day} of {start_date.strftime('%B')}",
                f"week {start_date.isocalendar()[1]} of {start_date.year}",
            ]
            
            # Add relative time context (this would be calculated relative to current date)
            # Convert both to timezone-naive for relative time calculation
            current_date = datetime.now()
            start_date_naive = start_date.replace(tzinfo=None)
            days_ago = (current_date - start_date_naive).days
            
            if days_ago == 0:
                parts.append("activity today")
            elif days_ago == 1:
                parts.append("activity yesterday")
            elif days_ago < 7:
                parts.append(f"activity {days_ago} days ago")
            elif days_ago < 30:
                weeks_ago = days_ago // 7
                parts.append(f"activity {weeks_ago} weeks ago")
            elif days_ago < 365:
                months_ago = days_ago // 30
                parts.append(f"activity {months_ago} months ago")
            
            return ". ".join(parts)
            
        except ValueError:
            return f"Activity on {start_date_str}"
    
    def _create_performance_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create a performance-focused summary for training analysis."""
        activity_type = activity_data.get("type", "activity").lower()
        
        parts = [f"Performance analysis for {activity_type}"]
        
        # Calculate performance indicators
        distance = activity_data.get("distance", 0)
        moving_time = activity_data.get("moving_time", 0)
        avg_power = activity_data.get("average_watts", 0)
        avg_hr = activity_data.get("average_heartrate", 0)
        elevation = activity_data.get("total_elevation_gain", 0)
        
        if distance and moving_time:
            # Calculate pace/speed performance
            if activity_type in ["run", "running"]:
                pace_min_per_km = (moving_time / 60) / (distance / 1000)
                pace_min = int(pace_min_per_km)
                pace_sec = int((pace_min_per_km - pace_min) * 60)
                parts.append(f"pace {pace_min}:{pace_sec:02d} per kilometer")
            else:
                speed_kmh = (distance / 1000) / (moving_time / 3600)
                parts.append(f"speed {speed_kmh:.1f} km/h")
        
        # Power performance
        if avg_power > 0:
            # Estimate performance level based on power
            if avg_power > 300:
                parts.append("high power output performance")
            elif avg_power > 200:
                parts.append("moderate power output performance")
            else:
                parts.append("endurance power output performance")
        
        # Heart rate performance
        if avg_hr > 0:
            if avg_hr > 160:
                parts.append("high intensity heart rate")
            elif avg_hr > 140:
                parts.append("moderate intensity heart rate")
            else:
                parts.append("low intensity heart rate")
        
        # Elevation performance
        if elevation > 1000:
            parts.append("high elevation gain challenging terrain")
        elif elevation > 500:
            parts.append("moderate elevation gain hilly terrain")
        elif elevation > 0:
            parts.append("low elevation gain mostly flat terrain")
        
        # Training stress
        suffer_score = activity_data.get("suffer_score", 0)
        if suffer_score > 100:
            parts.append("high training stress")
        elif suffer_score > 50:
            parts.append("moderate training stress")
        elif suffer_score > 0:
            parts.append("low training stress")
        
        return ". ".join(parts)
    
    def _extract_garmin_activity_metadata(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for Garmin activity data."""
        from datetime import datetime
        
        # Parse start date
        start_date_str = activity_data.get("startTimeLocal", "")
        start_date = None
        timestamp = None
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                timestamp = int(start_date.timestamp())
            except ValueError:
                start_date = datetime.now()
                timestamp = int(start_date.timestamp())
        
        # Calculate derived metrics
        distance = activity_data.get("distance", 0)
        duration_seconds = activity_data.get("duration", 0)
        average_speed = activity_data.get("averageSpeed", 0)
        
        # Convert to standard units
        distance_km = distance / 1000 if distance else 0
        duration_minutes = duration_seconds / 60 if duration_seconds else 0
        duration_hours = duration_minutes / 60 if duration_minutes else 0
        avg_speed_kmh = average_speed * 3.6 if average_speed else 0
        max_speed_kmh = activity_data.get("maxSpeed", 0) * 3.6 if activity_data.get("maxSpeed") else 0
        
        # Calculate pace (min/km) for running activities
        avg_pace_min_per_km = None
        if distance_km > 0 and duration_minutes > 0:
            avg_pace_min_per_km = duration_minutes / distance_km
        
        # Power metrics
        avg_power = activity_data.get("averagePower", 0) or 0
        max_power = activity_data.get("maxPower", 0) or 0
        normalized_power = activity_data.get("normalizedPower", 0) or 0
        
        # Heart rate metrics
        avg_heartrate = activity_data.get("averageHeartRate", 0) or 0
        max_heartrate = activity_data.get("maxHeartRate", 0) or 0
        
        # Elevation metrics
        elevation_gain = activity_data.get("elevationGain", 0) or 0
        
        # Garmin-specific training metrics
        training_stress_score = activity_data.get("trainingStressScore", 0) or 0
        intensity_factor = activity_data.get("intensityFactor", 0) or 0
        
        # Calculate training zones (basic estimation)
        time_in_zones = self._estimate_training_zones(avg_heartrate, max_heartrate, duration_minutes)
        
        return {
            # Basic info
            "activity_type": activity_data.get("activityType", "").lower(),
            "name": activity_data.get("activityName", ""),
            "description": activity_data.get("description", ""),
            "data_source": "garmin",
            
            # Date and time
            "date": start_date.strftime("%Y-%m-%d") if start_date else "",
            "start_date": start_date_str,
            "timestamp": timestamp,
            "day_of_week": start_date.strftime("%A").lower() if start_date else "",
            "month": start_date.strftime("%B").lower() if start_date else "",
            "year": start_date.year if start_date else 0,
            
            # Distance and duration
            "distance": distance,
            "distance_km": distance_km,
            "duration": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_hours": duration_hours,
            
            # Speed and pace
            "average_speed": average_speed,
            "max_speed": activity_data.get("maxSpeed", 0) or 0,
            "avg_speed_kmh": avg_speed_kmh,
            "max_speed_kmh": max_speed_kmh,
            "avg_pace_min_per_km": avg_pace_min_per_km,
            
            # Power metrics
            "average_watts": avg_power,
            "max_watts": max_power,
            "avg_power_watts": avg_power,
            "max_power_watts": max_power,
            "normalized_power": normalized_power,
            "intensity_factor": intensity_factor,
            "has_power_data": avg_power > 0,
            
            # Heart rate metrics
            "average_heartrate": avg_heartrate,
            "max_heartrate": max_heartrate,
            "avg_heartrate": avg_heartrate,
            "max_heartrate_bpm": max_heartrate,
            "has_heartrate": avg_heartrate > 0,
            
            # Elevation
            "total_elevation_gain": elevation_gain,
            "elevation_gain_m": elevation_gain,
            
            # Training metrics
            "training_stress_score": training_stress_score,
            "tss": training_stress_score,
            
            # Training zones (estimated)
            **time_in_zones,
            
            # Other metrics
            "calories": activity_data.get("calories", 0) or 0,
            
            # Activity flags
            "indoor": activity_data.get("indoor", False),
            "outdoor": not activity_data.get("indoor", False),
            
            # Data quality flags
            "has_gps": distance > 0 and avg_speed > 0,
            "device_name": activity_data.get("deviceName", ""),
            
            # Performance indicators
            "intensity_score": self._calculate_garmin_intensity_score(avg_power, avg_heartrate, duration_minutes),
            "efficiency_score": self._calculate_garmin_efficiency_score(avg_speed_kmh, avg_heartrate, elevation_gain),
        }
    
    def _calculate_garmin_intensity_score(self, avg_power: float, avg_hr: float, duration_min: float) -> float:
        """Calculate a basic intensity score for Garmin activities."""
        if not duration_min:
            return 0.0
        
        # Simple intensity calculation based on available metrics
        intensity = 0.0
        
        if avg_power > 0:
            # Power-based intensity (simplified TSS-like calculation)
            intensity = (avg_power * duration_min) / 1000  # Normalized
        elif avg_hr > 0:
            # HR-based intensity
            intensity = (avg_hr * duration_min) / 10000  # Normalized
        
        return round(intensity, 2)
    
    def _calculate_garmin_efficiency_score(self, avg_speed: float, avg_hr: float, elevation: float) -> float:
        """Calculate efficiency score for Garmin activities."""
        if not avg_speed or not avg_hr:
            return 0.0
        
        # Basic efficiency: speed per heart rate beat, adjusted for elevation
        base_efficiency = avg_speed / avg_hr if avg_hr > 0 else 0
        elevation_factor = 1 + (elevation / 1000) * 0.1  # Slight bonus for elevation
        
        return round(base_efficiency * elevation_factor, 4)
    
    def _create_garmin_multi_vector_embeddings(self, activity_data: Dict[str, Any]) -> Dict[str, str]:
        """Create multi-vector embeddings for Garmin activities."""
        # Main activity summary
        main_summary = self._create_garmin_activity_summary(activity_data)
        
        # Metrics-focused summary
        metrics_summary = self._create_garmin_metrics_summary(activity_data)
        
        # Temporal-focused summary
        temporal_summary = self._create_garmin_temporal_summary(activity_data)
        
        # Performance-focused summary
        performance_summary = self._create_garmin_performance_summary(activity_data)
        
        return {
            "main": main_summary,
            "metrics": metrics_summary,
            "temporal": temporal_summary,
            "performance": performance_summary
        }
    
    def _create_garmin_metrics_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create a metrics-focused summary for Garmin activities."""
        parts = []
        
        # Power metrics
        if activity_data.get("averagePower"):
            parts.append(f"average power {activity_data['averagePower']} watts")
        if activity_data.get("maxPower"):
            parts.append(f"maximum power {activity_data['maxPower']} watts")
        if activity_data.get("normalizedPower"):
            parts.append(f"normalized power {activity_data['normalizedPower']} watts")
        
        # Heart rate metrics
        if activity_data.get("averageHeartRate"):
            parts.append(f"average heart rate {activity_data['averageHeartRate']} bpm")
        if activity_data.get("maxHeartRate"):
            parts.append(f"maximum heart rate {activity_data['maxHeartRate']} bpm")
        
        # Speed metrics
        if activity_data.get("averageSpeed"):
            speed_kmh = activity_data["averageSpeed"] * 3.6
            parts.append(f"average speed {speed_kmh:.1f} km/h")
        
        # Distance and duration
        if activity_data.get("distance"):
            distance_km = activity_data["distance"] / 1000
            parts.append(f"distance {distance_km:.1f} kilometers")
        
        if activity_data.get("duration"):
            duration_min = activity_data["duration"] / 60
            parts.append(f"duration {duration_min:.0f} minutes")
        
        # Elevation
        if activity_data.get("elevationGain"):
            parts.append(f"elevation gain {activity_data['elevationGain']} meters")
        
        # Garmin-specific training metrics
        if activity_data.get("trainingStressScore"):
            parts.append(f"training stress score {activity_data['trainingStressScore']}")
        
        if activity_data.get("intensityFactor"):
            parts.append(f"intensity factor {activity_data['intensityFactor']:.2f}")
        
        return f"Activity metrics: {', '.join(parts)}"
    
    def _create_garmin_temporal_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create a temporal-focused summary for Garmin activities."""
        from datetime import datetime
        
        start_date_str = activity_data.get("startTimeLocal", "")
        if not start_date_str:
            return "Activity with unknown date"
        
        try:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            
            parts = [
                f"Activity on {start_date.strftime('%Y-%m-%d')}",
                f"performed on {start_date.strftime('%A')}",
                f"in {start_date.strftime('%B %Y')}",
                f"day {start_date.day} of {start_date.strftime('%B')}",
                f"week {start_date.isocalendar()[1]} of {start_date.year}",
            ]
            
            # Add relative time context
            current_date = datetime.now()
            start_date_naive = start_date.replace(tzinfo=None)
            days_ago = (current_date - start_date_naive).days
            
            if days_ago == 0:
                parts.append("activity today")
            elif days_ago == 1:
                parts.append("activity yesterday")
            elif days_ago < 7:
                parts.append(f"activity {days_ago} days ago")
            elif days_ago < 30:
                weeks_ago = days_ago // 7
                parts.append(f"activity {weeks_ago} weeks ago")
            elif days_ago < 365:
                months_ago = days_ago // 30
                parts.append(f"activity {months_ago} months ago")
            
            return ". ".join(parts)
            
        except ValueError:
            return f"Activity on {start_date_str}"
    
    def _create_garmin_performance_summary(self, activity_data: Dict[str, Any]) -> str:
        """Create a performance-focused summary for Garmin activities."""
        activity_type = activity_data.get("activityType", "activity").upper()
        
        parts = [f"Performance analysis for {activity_type}"]
        
        # Calculate performance indicators
        distance = activity_data.get("distance", 0)
        duration_seconds = activity_data.get("duration", 0)
        avg_power = activity_data.get("averagePower", 0)
        normalized_power = activity_data.get("normalizedPower", 0)
        avg_hr = activity_data.get("averageHeartRate", 0)
        elevation = activity_data.get("elevationGain", 0)
        avg_speed = activity_data.get("averageSpeed", 0)
        
        if distance and duration_seconds:
            # Calculate pace/speed performance
            if activity_type in ["RUNNING", "TRAIL_RUNNING"]:
                pace_min_per_km = (duration_seconds / 60) / (distance / 1000)
                pace_min = int(pace_min_per_km)
                pace_sec = int((pace_min_per_km - pace_min) * 60)
                parts.append(f"pace {pace_min}:{pace_sec:02d} per kilometer")
            else:
                speed_kmh = (distance / 1000) / (duration_seconds / 3600)
                parts.append(f"speed {speed_kmh:.1f} km/h")
        
        # Power performance
        power_for_analysis = normalized_power if normalized_power else avg_power
        if power_for_analysis > 0:
            # Estimate performance level based on power
            if power_for_analysis > 300:
                parts.append("high power output performance")
            elif power_for_analysis > 200:
                parts.append("moderate power output performance")
            else:
                parts.append("endurance power output performance")
        
        # Heart rate performance
        if avg_hr > 0:
            if avg_hr > 160:
                parts.append("high intensity heart rate")
            elif avg_hr > 140:
                parts.append("moderate intensity heart rate")
            else:
                parts.append("low intensity heart rate")
        
        # Elevation performance
        if elevation > 1000:
            parts.append("high elevation gain challenging terrain")
        elif elevation > 500:
            parts.append("moderate elevation gain hilly terrain")
        elif elevation > 0:
            parts.append("low elevation gain mostly flat terrain")
        
        # Training stress
        training_stress_score = activity_data.get("trainingStressScore", 0)
        if training_stress_score > 100:
            parts.append("high training stress")
        elif training_stress_score > 50:
            parts.append("moderate training stress")
        elif training_stress_score > 0:
            parts.append("low training stress")
        
        return ". ".join(parts)