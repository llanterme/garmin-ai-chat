"""Garmin Connect service for interfacing with Garmin API."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from garminconnect import Garmin

from ..core.exceptions import GarminConnectError
from ..core.logging import get_logger
from ..core.security import encryption_handler
from .fitness_metrics import FitnessMetricsCalculator

logger = get_logger(__name__)


class GarminService:
    """Service for interacting with Garmin Connect."""

    def __init__(self, user_weight_kg: Optional[float] = None, user_ftp: Optional[float] = None, user_max_hr: Optional[int] = None) -> None:
        self._client: Optional[Garmin] = None
        self._session_data: Optional[Dict[str, Any]] = None
        self.metrics_calculator = FitnessMetricsCalculator(user_weight_kg, user_ftp, user_max_hr)

    async def authenticate(
        self, username: str, password: str, session_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Authenticate with Garmin Connect.
        
        Returns:
            Tuple of (success, session_data)
        """
        try:
            # Decrypt credentials
            decrypted_username = encryption_handler.decrypt(username)
            decrypted_password = encryption_handler.decrypt(password)
            
            # Create Garmin client
            self._client = Garmin(decrypted_username, decrypted_password)
            
            # Try to use existing session data if available
            if session_data:
                try:
                    self._client.session_data = session_data
                    # Test if session is still valid
                    await self._run_sync_method(self._client.get_user_summary, datetime.now())
                    logger.info("Successfully reused existing Garmin session")
                    return True, session_data
                except Exception:
                    logger.info("Existing session expired, creating new one")
            
            # Login and get new session data
            await self._run_sync_method(self._client.login)
            
            # Get session data for future use
            new_session_data = getattr(self._client, 'session_data', None)
            
            logger.info("Successfully authenticated with Garmin Connect")
            return True, new_session_data
            
        except Exception as e:
            logger.error(f"Failed to authenticate with Garmin Connect: {str(e)}")
            raise GarminConnectError(f"Authentication failed: {str(e)}")

    async def get_activities(
        self, start_date: datetime, end_date: Optional[datetime] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get activities for a date range.
        
        Args:
            start_date: Start date for activities
            end_date: End date for activities (defaults to now)
            limit: Optional maximum number of activities to return (defaults to no limit)
        """
        if not self._client:
            raise GarminConnectError("Not authenticated with Garmin Connect")

        try:
            if end_date is None:
                end_date = datetime.now()

            # Use the get_activities_by_date with both start and end dates
            # This should return only activities within the specified range
            activities = await self._run_sync_method(
                self._client.get_activities_by_date, 
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            if activities is None:
                activities = []
            
            # Apply limit if specified
            if limit is not None and len(activities) > limit:
                activities = activities[:limit]

            logger.info(f"Retrieved {len(activities)} activities from Garmin Connect for date range {start_date.date()} to {end_date.date()}")
            
            # Double-check that activities are actually within our date range
            filtered_activities = []
            for activity in activities:
                activity_date_str = activity.get('startTimeLocal')
                if activity_date_str:
                    try:
                        # Parse the activity date
                        activity_date = datetime.fromisoformat(activity_date_str.replace('Z', '+00:00'))
                        # Only include if within our requested range
                        if start_date <= activity_date <= end_date + timedelta(days=1):  # Add 1 day for end of day
                            filtered_activities.append(activity)
                        else:
                            logger.debug(f"Filtering out activity from {activity_date} (outside range {start_date} to {end_date})")
                    except Exception as e:
                        logger.warning(f"Could not parse date {activity_date_str}: {e}")
                        # Include if we can't parse the date
                        filtered_activities.append(activity)
                else:
                    # Include if no date found
                    filtered_activities.append(activity)
            
            if len(filtered_activities) != len(activities):
                logger.info(f"Filtered {len(activities)} activities down to {len(filtered_activities)} within date range")
            
            return filtered_activities

        except Exception as e:
            logger.error(f"Failed to get activities: {str(e)}")
            raise GarminConnectError(f"Failed to retrieve activities: {str(e)}")

    async def get_activity_details(self, activity_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific activity."""
        if not self._client:
            raise GarminConnectError("Not authenticated with Garmin Connect")

        try:
            # Get basic activity details
            activity_details = await self._run_sync_method(
                self._client.get_activity, activity_id
            )
            
            # Try to get additional details (splits, heart rate zones, etc.)
            additional_data = {}
            
            try:
                splits = await self._run_sync_method(
                    self._client.get_activity_splits, activity_id
                )
                additional_data['splits'] = splits
            except Exception:
                logger.debug(f"Could not retrieve splits for activity {activity_id}")
            
            try:
                hr_zones = await self._run_sync_method(
                    self._client.get_activity_hr_in_timezones, activity_id
                )
                additional_data['hr_zones'] = hr_zones
            except Exception:
                logger.debug(f"Could not retrieve HR zones for activity {activity_id}")

            # Combine all data
            if additional_data:
                activity_details['additional_data'] = additional_data

            logger.debug(f"Retrieved details for activity {activity_id}")
            return activity_details

        except Exception as e:
            logger.error(f"Failed to get activity details for {activity_id}: {str(e)}")
            raise GarminConnectError(f"Failed to retrieve activity details: {str(e)}")

    async def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile information."""
        if not self._client:
            raise GarminConnectError("Not authenticated with Garmin Connect")

        try:
            profile = await self._run_sync_method(self._client.get_user_summary, datetime.now())
            logger.debug("Retrieved user profile from Garmin Connect")
            return profile

        except Exception as e:
            logger.error(f"Failed to get user profile: {str(e)}")
            raise GarminConnectError(f"Failed to retrieve user profile: {str(e)}")

    async def test_connection(self) -> bool:
        """Test if the connection to Garmin Connect is working."""
        if not self._client:
            return False

        try:
            await self._run_sync_method(self._client.get_user_summary, datetime.now())
            return True
        except Exception:
            return False

    @staticmethod
    async def _run_sync_method(method, *args, **kwargs) -> Any:
        """Run a synchronous method in an async context."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: method(*args, **kwargs))

    def parse_activity_data(self, raw_activity: Dict[str, Any], activity_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Parse raw Garmin activity data into standardized format with derived metrics."""
        try:
            # Extract basic information
            parsed = {
                'garmin_activity_id': str(raw_activity.get('activityId', '')),
                'activity_name': raw_activity.get('activityName'),
                'activity_type': raw_activity.get('activityType', {}).get('typeKey', 'unknown'),
                'sport_type': raw_activity.get('eventType', {}).get('typeKey'),
                'start_time': None,
                'duration': raw_activity.get('duration'),  # seconds
                'distance': raw_activity.get('distance'),  # meters
                'calories': raw_activity.get('calories'),
            }

            # Parse start time
            start_time_str = raw_activity.get('startTimeLocal')
            if start_time_str:
                try:
                    parsed['start_time'] = datetime.fromisoformat(
                        start_time_str.replace('Z', '+00:00')
                    )
                except ValueError:
                    logger.warning(f"Could not parse start time: {start_time_str}")

            # Parse performance metrics
            parsed.update({
                'average_speed': raw_activity.get('averageSpeed'),  # m/s
                'max_speed': raw_activity.get('maxSpeed'),  # m/s
                'average_heart_rate': raw_activity.get('averageHR'),
                'max_heart_rate': raw_activity.get('maxHR'),
            })

            # Parse elevation data
            parsed.update({
                'elevation_gain': raw_activity.get('elevationGain'),
                'elevation_loss': raw_activity.get('elevationLoss'),
                'min_elevation': raw_activity.get('minElevation'),
                'max_elevation': raw_activity.get('maxElevation'),
            })

            # Parse power data
            parsed.update({
                'average_power': raw_activity.get('avgPower'),
                'max_power': raw_activity.get('maxPower'),
                'normalized_power': raw_activity.get('normalizedPower'),
            })

            # Parse cycling-specific data
            parsed.update({
                'average_cadence': raw_activity.get('avgRunCadence') or raw_activity.get('avgBikingCadence'),
                'max_cadence': raw_activity.get('maxRunCadence') or raw_activity.get('maxBikingCadence'),
            })

            # Parse swimming-specific data
            parsed.update({
                'pool_length': raw_activity.get('poolLength'),
                'strokes': raw_activity.get('strokes'),
                'swim_stroke_type': raw_activity.get('strokeType', {}).get('typeKey') if raw_activity.get('strokeType') else None,
            })

            # Parse training metrics
            parsed.update({
                'training_stress_score': raw_activity.get('trainingStressScore'),
                'intensity_factor': raw_activity.get('intensityFactor'),
                'vo2_max': raw_activity.get('vO2MaxValue'),
            })

            # Parse location data
            start_latitude = raw_activity.get('startLatitude')
            start_longitude = raw_activity.get('startLongitude')
            
            # Convert from semicircles to degrees if needed
            if start_latitude is not None and abs(start_latitude) > 180:
                start_latitude = start_latitude * (180.0 / 2**31)
            if start_longitude is not None and abs(start_longitude) > 180:
                start_longitude = start_longitude * (180.0 / 2**31)
                
            parsed.update({
                'start_latitude': start_latitude,
                'start_longitude': start_longitude,
            })

            # Parse weather data
            parsed.update({
                'temperature': raw_activity.get('averageTemperature'),
                'weather_condition': raw_activity.get('weatherCondition'),
            })

            # Store raw data for reference
            parsed['raw_data'] = raw_activity
            parsed['summary_data'] = {
                'activity_type': parsed['activity_type'],
                'duration': parsed['duration'],
                'distance': parsed['distance'],
                'calories': parsed['calories'],
                'average_speed': parsed['average_speed'],
                'average_heart_rate': parsed['average_heart_rate'],
            }
            
            # Calculate all derived metrics
            derived_metrics = self.metrics_calculator.calculate_all_metrics(parsed, activity_history)
            
            # Merge derived metrics into parsed data
            parsed.update(derived_metrics)
            
            # Add derived metrics to summary data for quick access
            if derived_metrics.get('pace_per_km_formatted'):
                parsed['summary_data']['pace'] = derived_metrics['pace_per_km_formatted']
            if derived_metrics.get('watts_per_kg'):
                parsed['summary_data']['watts_per_kg'] = derived_metrics['watts_per_kg']
            if derived_metrics.get('training_load'):
                parsed['summary_data']['training_load'] = derived_metrics['training_load']

            return parsed

        except Exception as e:
            logger.error(f"Error parsing activity data: {str(e)}")
            # Return minimal data with raw data for debugging
            return {
                'garmin_activity_id': str(raw_activity.get('activityId', '')),
                'activity_type': 'unknown',
                'raw_data': raw_activity,
                'parsing_error': str(e),
            }