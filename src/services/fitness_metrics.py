"""Fitness metrics calculator for deriving advanced metrics from raw activity data."""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core.logging import get_logger

logger = get_logger(__name__)


class FitnessMetricsCalculator:
    """Calculate derived fitness metrics from raw activity data."""
    
    def __init__(self, user_weight_kg: Optional[float] = None, user_ftp: Optional[float] = None, user_max_hr: Optional[int] = None):
        """
        Initialize calculator with user-specific parameters.
        
        Args:
            user_weight_kg: User's weight in kilograms (for W/kg calculations)
            user_ftp: User's Functional Threshold Power in watts (cycling)
            user_max_hr: User's maximum heart rate (for zone calculations)
        """
        self.user_weight_kg = user_weight_kg or 70  # Default 70kg if not provided
        self.user_ftp = user_ftp or 250  # Default 250W if not provided
        self.user_max_hr = user_max_hr or 185  # Default max HR if not provided
        
    def calculate_all_metrics(self, activity_data: Dict[str, Any], activity_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate all derived metrics for an activity.
        
        Args:
            activity_data: Raw activity data from Garmin
            activity_history: Recent activity history for comparative metrics
            
        Returns:
            Dictionary of derived metrics
        """
        metrics = {}
        
        # Calculate base metrics based on activity type
        activity_type = activity_data.get('activity_type', '').lower()
        
        # Running metrics
        if 'running' in activity_type or 'treadmill' in activity_type:
            metrics.update(self.calculate_running_metrics(activity_data))
            
        # Cycling metrics
        if 'cycling' in activity_type or 'bike' in activity_type or 'ride' in activity_type:
            metrics.update(self.calculate_cycling_metrics(activity_data))
            
        # Swimming metrics
        if 'swimming' in activity_type or 'swim' in activity_type:
            metrics.update(self.calculate_swimming_metrics(activity_data))
            
        # Universal metrics (apply to all activities)
        metrics.update(self.calculate_heart_rate_zones(activity_data))
        metrics.update(self.calculate_training_load_metrics(activity_data, activity_history))
        metrics.update(self.calculate_performance_indicators(activity_data, activity_history))
        metrics.update(self.calculate_temporal_context(activity_data))
        metrics.update(self.calculate_effort_metrics(activity_data))
        
        return metrics
    
    def calculate_running_metrics(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate running-specific metrics."""
        metrics = {}
        
        # Calculate pace from speed
        avg_speed = activity_data.get('average_speed')  # m/s
        if avg_speed and avg_speed > 0:
            speed_kmh = avg_speed * 3.6
            pace_per_km = 60 / speed_kmh if speed_kmh > 0 else None
            
            if pace_per_km:
                metrics['pace_per_km'] = round(pace_per_km, 2)
                metrics['pace_per_km_formatted'] = self._format_pace(pace_per_km)
                metrics['pace_per_mile'] = round(pace_per_km * 1.60934, 2)
                metrics['pace_per_mile_formatted'] = self._format_pace(pace_per_km * 1.60934)
        
        # Calculate stride length if cadence is available
        cadence = activity_data.get('average_cadence')
        distance = activity_data.get('distance')  # meters
        duration = activity_data.get('duration')  # seconds
        
        if cadence and distance and duration and cadence > 0 and duration > 0:
            steps = (cadence * 2) * (duration / 60)  # cadence is per minute, doubled for both feet
            if steps > 0:
                metrics['stride_length_m'] = round(distance / steps, 2)
        
        # Calculate running efficiency (meters per heartbeat)
        avg_hr = activity_data.get('average_heart_rate')
        if distance and avg_hr and duration and avg_hr > 0 and duration > 0:
            heartbeats = avg_hr * (duration / 60)
            if heartbeats > 0:
                metrics['running_efficiency'] = round(distance / heartbeats, 3)
        
        # Grade adjusted pace (GAP) - simplified calculation
        elevation_gain = activity_data.get('elevation_gain', 0)
        if distance and distance > 0 and elevation_gain is not None and metrics.get('pace_per_km'):
            grade_percent = (elevation_gain / distance) * 100
            # Rough approximation: 1% grade = 6-8 seconds per km adjustment
            gap_adjustment = grade_percent * 7  # seconds per km
            metrics['grade_adjusted_pace'] = round(metrics['pace_per_km'] - (gap_adjustment / 60), 2)
            metrics['grade_adjusted_pace_formatted'] = self._format_pace(metrics['grade_adjusted_pace'])
        
        return metrics
    
    def calculate_cycling_metrics(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cycling-specific metrics."""
        metrics = {}
        
        avg_power = activity_data.get('average_power')
        normalized_power = activity_data.get('normalized_power', avg_power)
        
        # Power to weight ratio
        if avg_power and self.user_weight_kg and self.user_weight_kg > 0:
            metrics['watts_per_kg'] = round(avg_power / self.user_weight_kg, 2)
            
        if normalized_power and self.user_weight_kg and self.user_weight_kg > 0:
            metrics['normalized_watts_per_kg'] = round(normalized_power / self.user_weight_kg, 2)
        
        # FTP-related metrics
        if normalized_power and self.user_ftp and self.user_ftp > 0:
            # Intensity Factor (IF)
            metrics['intensity_factor'] = round(normalized_power / self.user_ftp, 2)
            
            # FTP percentage
            metrics['ftp_percentage'] = round((normalized_power / self.user_ftp) * 100, 1)
            
            # Training Stress Score (TSS)
            duration = activity_data.get('duration')  # seconds
            if duration and duration > 0:
                intensity_factor = normalized_power / self.user_ftp
                tss = (duration * normalized_power * intensity_factor) / (self.user_ftp * 36)
                metrics['calculated_tss'] = round(tss, 1)
        
        # Variability Index (VI) - shows how smooth the power output was
        if normalized_power and avg_power and avg_power > 0:
            metrics['variability_index'] = round(normalized_power / avg_power, 2)
        
        # Efficiency Factor (EF) - normalized power / average heart rate
        avg_hr = activity_data.get('average_heart_rate')
        if normalized_power and avg_hr and avg_hr > 0:
            metrics['efficiency_factor'] = round(normalized_power / avg_hr, 2)
        
        # Cycling dynamics
        avg_cadence = activity_data.get('average_cadence')
        if avg_cadence:
            # Categorize cadence
            if avg_cadence < 70:
                metrics['cadence_category'] = 'low'
            elif avg_cadence < 90:
                metrics['cadence_category'] = 'moderate'
            else:
                metrics['cadence_category'] = 'high'
        
        return metrics
    
    def calculate_swimming_metrics(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate swimming-specific metrics."""
        metrics = {}
        
        distance = activity_data.get('distance')  # meters
        duration = activity_data.get('duration')  # seconds
        strokes = activity_data.get('strokes')
        pool_length = activity_data.get('pool_length', 25)  # default 25m pool
        
        # SWOLF score (stroke count + time in seconds for one pool length)
        if strokes and duration and distance and distance > 0:
            laps = distance / pool_length
            if laps > 0:
                avg_strokes_per_lap = strokes / laps
                avg_time_per_lap = duration / laps
                metrics['swolf_score'] = round(avg_strokes_per_lap + avg_time_per_lap, 1)
        
        # Stroke efficiency (distance per stroke)
        if distance and strokes and strokes > 0:
            metrics['distance_per_stroke'] = round(distance / strokes, 2)
        
        # Pace per 100m
        if distance and duration and distance > 0:
            pace_per_100m = (duration / distance) * 100
            minutes = int(pace_per_100m // 60)
            seconds = int(pace_per_100m % 60)
            metrics['pace_per_100m'] = round(pace_per_100m / 60, 2)  # in minutes
            metrics['pace_per_100m_formatted'] = f"{minutes}:{seconds:02d}"
        
        return metrics
    
    def calculate_heart_rate_zones(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate heart rate zone distribution."""
        metrics = {}
        
        avg_hr = activity_data.get('average_heart_rate')
        max_hr = activity_data.get('max_heart_rate')
        
        if not avg_hr:
            return metrics
        
        # Use activity max HR if higher than user's set max HR
        effective_max_hr = max(self.user_max_hr, max_hr) if max_hr else self.user_max_hr
        
        # Calculate HR as percentage of max
        if effective_max_hr and effective_max_hr > 0:
            metrics['hr_percent_of_max'] = round((avg_hr / effective_max_hr) * 100, 1)
        
        # Determine primary HR zone (1-5)
        if effective_max_hr and effective_max_hr > 0:
            hr_percent = (avg_hr / effective_max_hr) * 100
            if hr_percent < 60:
                metrics['primary_hr_zone'] = 1
                metrics['hr_zone_name'] = 'recovery'
            elif hr_percent < 70:
                metrics['primary_hr_zone'] = 2
                metrics['hr_zone_name'] = 'aerobic'
            elif hr_percent < 80:
                metrics['primary_hr_zone'] = 3
                metrics['hr_zone_name'] = 'tempo'
            elif hr_percent < 90:
                metrics['primary_hr_zone'] = 4
                metrics['hr_zone_name'] = 'threshold'
            else:
                metrics['primary_hr_zone'] = 5
                metrics['hr_zone_name'] = 'vo2max'
        
        # Calculate TRIMP (Training Impulse) - simplified
        duration = activity_data.get('duration')  # seconds
        if duration and effective_max_hr and effective_max_hr > 60:
            duration_min = duration / 60
            # Exponential factor based on HR intensity
            if avg_hr > 0:
                hr_reserve_fraction = (avg_hr - 60) / (effective_max_hr - 60)  # assuming resting HR of 60
                if hr_reserve_fraction > 0:
                    gender_factor = 1.67  # default for males, would be 1.92 for females
                    trimp = duration_min * hr_reserve_fraction * 0.64 * math.exp(gender_factor * hr_reserve_fraction)
                    metrics['trimp_score'] = round(trimp, 1)
        
        return metrics
    
    def calculate_training_load_metrics(self, activity_data: Dict[str, Any], activity_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Calculate training load and stress metrics."""
        metrics = {}
        
        # Simple training load based on duration and intensity
        duration = activity_data.get('duration')  # seconds
        avg_hr = activity_data.get('average_heart_rate')
        
        if duration and duration > 0:
            duration_min = duration / 60
            
            # Basic training load (duration * intensity)
            if avg_hr and self.user_max_hr and self.user_max_hr > 0:
                intensity = avg_hr / self.user_max_hr
                metrics['training_load'] = round(duration_min * intensity, 1)
            else:
                # Fallback to just duration-based
                metrics['training_load'] = round(duration_min * 0.7, 1)  # assume moderate intensity
        
        # Calculate relative effort (0-10 scale)
        if duration and avg_hr and duration > 0:
            # Combination of duration and intensity
            duration_factor = min(duration / 7200, 1)  # max out at 2 hours
            intensity_factor = (avg_hr / self.user_max_hr) if (self.user_max_hr and self.user_max_hr > 0) else 0.7
            relative_effort = (duration_factor * 0.4 + intensity_factor * 0.6) * 10
            metrics['relative_effort'] = round(relative_effort, 1)
        
        # If we have history, calculate comparative metrics
        if activity_history and len(activity_history) > 0:
            # Get similar activities
            activity_type = activity_data.get('activity_type', '').lower()
            similar_activities = [a for a in activity_history if a.get('activity_type', '').lower() == activity_type]
            
            if similar_activities:
                # Calculate percentile ranking
                if avg_hr:
                    hr_values = [a.get('average_heart_rate', 0) for a in similar_activities if a.get('average_heart_rate')]
                    if hr_values:
                        percentile = sum(1 for hr in hr_values if hr < avg_hr) / len(hr_values) * 100
                        metrics['hr_percentile_in_history'] = round(percentile, 1)
                
                # Training monotony (variation in training load)
                loads = [a.get('training_load', 0) for a in similar_activities[-7:] if a.get('training_load')]
                if len(loads) > 2:
                    avg_load = sum(loads) / len(loads)
                    std_load = math.sqrt(sum((l - avg_load) ** 2 for l in loads) / len(loads))
                    if std_load > 0:
                        metrics['training_monotony'] = round(avg_load / std_load, 2)
        
        return metrics
    
    def calculate_performance_indicators(self, activity_data: Dict[str, Any], activity_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Calculate performance indicators and flags."""
        metrics = {}
        
        distance = activity_data.get('distance')  # meters
        duration = activity_data.get('duration')  # seconds
        activity_type = activity_data.get('activity_type', '').lower()
        
        # Detect if this is likely a race or hard effort
        avg_hr = activity_data.get('average_heart_rate')
        if avg_hr and self.user_max_hr:
            hr_percent = (avg_hr / self.user_max_hr) * 100
            if hr_percent > 85:
                metrics['is_hard_effort'] = True
                metrics['effort_level'] = 'hard'
            elif hr_percent > 75:
                metrics['effort_level'] = 'moderate'
            else:
                metrics['effort_level'] = 'easy'
        
        # Check for interval workout patterns (would need lap data for accurate detection)
        max_speed = activity_data.get('max_speed')
        avg_speed = activity_data.get('average_speed')
        if max_speed and avg_speed and avg_speed > 0:
            speed_variability = (max_speed - avg_speed) / avg_speed
            if speed_variability > 0.3:  # 30% variation suggests intervals
                metrics['is_interval_workout'] = True
        
        # Detect long runs/rides
        if duration:
            duration_hours = duration / 3600
            if 'running' in activity_type and duration_hours > 1.5:
                metrics['is_long_run'] = True
            elif 'cycling' in activity_type and duration_hours > 2:
                metrics['is_long_ride'] = True
        
        # Check if it's a recovery activity
        if metrics.get('effort_level') == 'easy' and duration and duration < 3600:
            metrics['is_recovery'] = True
        
        # Calculate fatigue index (performance decline over duration)
        # This would be more accurate with lap/split data
        if avg_speed and max_speed and max_speed > 0:
            # Simplified: compare average to max as proxy for consistency
            metrics['fatigue_index'] = round((1 - (avg_speed / max_speed)) * 100, 1)
        
        # If we have history, check for personal records
        if activity_history and distance and duration:
            similar_distances = []
            for hist_activity in activity_history:
                hist_distance = hist_activity.get('distance')
                hist_duration = hist_activity.get('duration')
                hist_type = hist_activity.get('activity_type', '').lower()
                
                if hist_type == activity_type and hist_distance and hist_duration:
                    # Check common race distances
                    if self._is_similar_distance(distance, hist_distance):
                        similar_distances.append(hist_duration)
            
            if similar_distances and duration < min(similar_distances):
                metrics['is_personal_record'] = True
                metrics['pr_improvement_seconds'] = round(min(similar_distances) - duration, 1)
        
        return metrics
    
    def calculate_temporal_context(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate temporal and contextual metadata."""
        metrics = {}
        
        start_time = activity_data.get('start_time')
        if start_time:
            if isinstance(start_time, str):
                try:
                    dt = datetime.fromisoformat(start_time.replace('T', ' ').replace('Z', ''))
                except:
                    return metrics
            else:
                dt = start_time
            
            # Time of day classification
            hour = dt.hour
            if 5 <= hour < 9:
                metrics['time_of_day'] = 'early_morning'
            elif 9 <= hour < 12:
                metrics['time_of_day'] = 'morning'
            elif 12 <= hour < 15:
                metrics['time_of_day'] = 'afternoon'
            elif 15 <= hour < 18:
                metrics['time_of_day'] = 'late_afternoon'
            elif 18 <= hour < 21:
                metrics['time_of_day'] = 'evening'
            else:
                metrics['time_of_day'] = 'night'
            
            # Day of week
            metrics['day_of_week'] = dt.strftime('%A').lower()
            metrics['is_weekend'] = dt.weekday() >= 5
            
            # Week of year
            metrics['week_of_year'] = dt.isocalendar()[1]
            
            # Season (Northern Hemisphere)
            month = dt.month
            if month in [12, 1, 2]:
                metrics['season'] = 'winter'
            elif month in [3, 4, 5]:
                metrics['season'] = 'spring'
            elif month in [6, 7, 8]:
                metrics['season'] = 'summer'
            else:
                metrics['season'] = 'fall'
        
        return metrics
    
    def calculate_effort_metrics(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate effort and intensity metrics."""
        metrics = {}
        
        # Suffer Score approximation (based on time in hard zones)
        avg_hr = activity_data.get('average_heart_rate')
        duration = activity_data.get('duration')
        
        if avg_hr and duration and self.user_max_hr:
            hr_percent = (avg_hr / self.user_max_hr) * 100
            duration_min = duration / 60
            
            # Higher multiplier for higher zones
            if hr_percent >= 90:
                suffer_score = duration_min * 4
            elif hr_percent >= 80:
                suffer_score = duration_min * 2
            elif hr_percent >= 70:
                suffer_score = duration_min * 1
            else:
                suffer_score = duration_min * 0.5
            
            metrics['suffer_score'] = round(suffer_score, 1)
        
        # Energy expenditure enhancement
        calories = activity_data.get('calories')
        if calories and duration and duration > 0:
            metrics['calories_per_minute'] = round(calories / (duration / 60), 1)
            
            # METs estimation from calories
            if self.user_weight_kg and self.user_weight_kg > 0 and duration > 0:
                # 1 MET = 1 kcal/kg/hour
                mets = (calories / self.user_weight_kg) / (duration / 3600)
                metrics['metabolic_equivalent'] = round(mets, 1)
        
        return metrics
    
    def _format_pace(self, pace_decimal_minutes: float) -> str:
        """Format pace from decimal minutes to mm:ss format."""
        if not pace_decimal_minutes or pace_decimal_minutes < 0:
            return "N/A"
        
        minutes = int(pace_decimal_minutes)
        seconds = int((pace_decimal_minutes - minutes) * 60)
        return f"{minutes}:{seconds:02d}"
    
    def _is_similar_distance(self, distance1: float, distance2: float, tolerance: float = 0.05) -> bool:
        """Check if two distances are similar (within tolerance)."""
        if distance1 == 0 or distance2 == 0:
            return False
        
        ratio = distance1 / distance2
        return (1 - tolerance) <= ratio <= (1 + tolerance)