"""Temporal query processor for parsing and handling time-based queries in fitness conversations."""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..core.logging import get_logger

logger = get_logger(__name__)


class TemporalFilter(BaseModel):
    """Temporal filter for database queries."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    date_range_days: Optional[int] = None
    specific_date: Optional[datetime] = None
    is_exact_date: bool = False  # Flag for exact date matching (like specific weekdays)


class MetricFilter(BaseModel):
    """Metric-based filter for performance data."""
    field: str
    operator: str  # "gt", "lt", "gte", "lte", "eq", "between"
    value: float
    value_max: Optional[float] = None  # For "between" operator


class QueryContext(BaseModel):
    """Parsed query context with filters and enhanced query."""
    original_query: str
    enhanced_query: str
    temporal_filter: Optional[TemporalFilter] = None
    metric_filters: List[MetricFilter] = []
    activity_type_filter: Optional[str] = None
    has_temporal_context: bool = False
    has_metric_context: bool = False
    query_type: str = "general"  # "temporal", "aggregation", "comparison", "analysis"


class TemporalQueryProcessor:
    """Processes natural language queries to extract temporal and metric filters."""
    
    def __init__(self):
        self.logger = logger
        
        # Temporal patterns with lambda functions that capture match groups
        self.temporal_patterns = {
            # Time of day patterns
            r'\byesterday\s+(morning|afternoon|evening)\b': lambda m: self._get_yesterday_time_of_day(m.group(1)),
            r'\bthis\s+(morning|afternoon|evening)\b': lambda m: self._get_today_time_of_day(m.group(1)),
            r'\btonight\b': lambda m=None: datetime.now().replace(hour=18, minute=0, second=0),
            
            # Relative days
            r'\byesterday\b': lambda m=None: datetime.now() - timedelta(days=1),
            r'\btoday\b': lambda m=None: datetime.now(),
            r'\b(\d+)\s+days?\s+ago\b': lambda m: datetime.now() - timedelta(days=int(m.group(1))),
            
            # Weekday patterns (MUST come before general week patterns to take priority)
            r'\b(?:on\s+)?(?:this\s+week\s*(?:\'s)?\s+)?(monday)\b': lambda m: self._get_weekday_in_week(0, "this"),
            r'\b(?:on\s+)?(?:this\s+week\s*(?:\'s)?\s+)?(tuesday)\b': lambda m: self._get_weekday_in_week(1, "this"),
            r'\b(?:on\s+)?(?:this\s+week\s*(?:\'s)?\s+)?(wednesday)\b': lambda m: self._get_weekday_in_week(2, "this"),
            r'\b(?:on\s+)?(?:this\s+week\s*(?:\'s)?\s+)?(thursday)\b': lambda m: self._get_weekday_in_week(3, "this"),
            r'\b(?:on\s+)?(?:this\s+week\s*(?:\'s)?\s+)?(friday)\b': lambda m: self._get_weekday_in_week(4, "this"),
            r'\b(?:on\s+)?(?:this\s+week\s*(?:\'s)?\s+)?(saturday)\b': lambda m: self._get_weekday_in_week(5, "this"),
            r'\b(?:on\s+)?(?:this\s+week\s*(?:\'s)?\s+)?(sunday)\b': lambda m: self._get_weekday_in_week(6, "this"),
            
            r'\b(?:on\s+)?last\s+week\s*(?:\'s)?\s+(monday)\b': lambda m: self._get_weekday_in_week(0, "last"),
            r'\b(?:on\s+)?last\s+week\s*(?:\'s)?\s+(tuesday)\b': lambda m: self._get_weekday_in_week(1, "last"),
            r'\b(?:on\s+)?last\s+week\s*(?:\'s)?\s+(wednesday)\b': lambda m: self._get_weekday_in_week(2, "last"),
            r'\b(?:on\s+)?last\s+week\s*(?:\'s)?\s+(thursday)\b': lambda m: self._get_weekday_in_week(3, "last"),
            r'\b(?:on\s+)?last\s+week\s*(?:\'s)?\s+(friday)\b': lambda m: self._get_weekday_in_week(4, "last"),
            r'\b(?:on\s+)?last\s+week\s*(?:\'s)?\s+(saturday)\b': lambda m: self._get_weekday_in_week(5, "last"),
            r'\b(?:on\s+)?last\s+week\s*(?:\'s)?\s+(sunday)\b': lambda m: self._get_weekday_in_week(6, "last"),

            # Relative weeks (after weekday patterns)
            r'\blast\s+week\b': lambda m=None: self._start_of_week() - timedelta(weeks=1),
            r'\bthis\s+week\b': lambda m=None: self._start_of_week(),
            r'\b(\d+)\s+weeks?\s+ago\b': lambda m: datetime.now() - timedelta(weeks=int(m.group(1))),
            
            # Relative months
            r'\blast\s+month\b': lambda m=None: self._start_of_month() - timedelta(days=30),
            r'\bthis\s+month\b': lambda m=None: self._start_of_month(),
            r'\b(\d+)\s+months?\s+ago\b': lambda m: datetime.now() - timedelta(days=int(m.group(1)) * 30),
            
            # Relative years
            r'\blast\s+year\b': lambda m=None: datetime.now() - timedelta(days=365),
            r'\bthis\s+year\b': lambda m=None: self._start_of_year(),
            
            # Specific dates
            r'\bon\s+(\d{4}-\d{2}-\d{2})\b': lambda m: datetime.strptime(m.group(1), '%Y-%m-%d'),
            r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?\b': 
                lambda m: self._parse_month(m.group(1), m.group(2)),
            
            # Date ranges
            r'\bover\s+the\s+past\s+(\d+)\s+days?\b': lambda m: datetime.now() - timedelta(days=int(m.group(1))),
            r'\bin\s+the\s+last\s+(\d+)\s+weeks?\b': lambda m: datetime.now() - timedelta(weeks=int(m.group(1))),
        }
        
        # Metric patterns for performance queries
        self.metric_patterns = {
            r'\bfaster\s+than\s+([\d.]+)\s*(?:km/h|kmh|kph)\b': lambda m: MetricFilter(field="average_speed_kmh", operator="gt", value=float(m.group(1))),
            r'\bslower\s+than\s+([\d.]+)\s*(?:km/h|kmh|kph)\b': lambda m: MetricFilter(field="average_speed_kmh", operator="lt", value=float(m.group(1))),
            r'\bmore\s+than\s+([\d.]+)\s*(?:km|kilometers?)\b': lambda m: MetricFilter(field="distance_km", operator="gt", value=float(m.group(1))),
            r'\bless\s+than\s+([\d.]+)\s*(?:km|kilometers?)\b': lambda m: MetricFilter(field="distance_km", operator="lt", value=float(m.group(1))),
            r'\bover\s+([\d.]+)\s*(?:km|kilometers?)\b': lambda m: MetricFilter(field="distance_km", operator="gt", value=float(m.group(1))),
            r'\bunder\s+([\d.]+)\s*(?:km|kilometers?)\b': lambda m: MetricFilter(field="distance_km", operator="lt", value=float(m.group(1))),
            r'\blonger\s+than\s+([\d.]+)\s*(?:hours?|hrs?)\b': lambda m: MetricFilter(field="duration_minutes", operator="gt", value=float(m.group(1)) * 60),
            r'\bshorter\s+than\s+([\d.]+)\s*(?:hours?|hrs?)\b': lambda m: MetricFilter(field="duration_minutes", operator="lt", value=float(m.group(1)) * 60),
            r'\blonger\s+than\s+([\d.]+)\s*(?:minutes?|mins?)\b': lambda m: MetricFilter(field="duration_minutes", operator="gt", value=float(m.group(1))),
            r'\bshorter\s+than\s+([\d.]+)\s*(?:minutes?|mins?)\b': lambda m: MetricFilter(field="duration_minutes", operator="lt", value=float(m.group(1))),
            r'\babove\s+([\d.]+)\s*(?:watts?|w)\b': lambda m: MetricFilter(field="average_power", operator="gt", value=float(m.group(1))),
            r'\bbelow\s+([\d.]+)\s*(?:watts?|w)\b': lambda m: MetricFilter(field="average_power", operator="lt", value=float(m.group(1))),
            r'\bhr\s+above\s+([\d.]+)\s*(?:bpm)?\b': lambda m: MetricFilter(field="average_heart_rate", operator="gt", value=float(m.group(1))),
            r'\bhr\s+below\s+([\d.]+)\s*(?:bpm)?\b': lambda m: MetricFilter(field="average_heart_rate", operator="lt", value=float(m.group(1))),
        }
        
        # Activity type patterns with multiple synonyms and variations
        self.activity_type_patterns = {
            # Running and related activities
            r'\b(running?|runs?|ran|jogging?|jogs?|marathon|5k|10k)\b': "running",
            r'\btreadmill\s*(running?)?\b': "treadmill_running",
            
            # Cycling and related activities  
            r'\b(cycling?|cycles?|cycled|rides?|rode|ridden|bike|biking|biked|pedal|pedaling)\b': "cycling",
            r'\b(virtual\s*rides?|indoor\s*(cycling?|rides?|bike)|zwift|trainer|turbo)\b': "virtual_ride",
            r'\b(spin|spinning)\b': "virtual_ride",
            
            # Swimming
            r'\b(swimming?|swims?|swam|swum|pool|laps?)\b': "swimming",
            
            # Walking and hiking
            r'\b(walking?|walks?|walked|hiking?|hikes?|hiked|trekking?)\b': "walking",
            
            # Strength training
            r'\b(strength|weights?|lifting|gym|workout|resistance|crossfit)\b': "strength_training",
            
            # Other activities
            r'\b(yoga|pilates|stretching?)\b': "yoga",
            r'\b(rowing?|rows?|rowed|erg)\b': "rowing",
            r'\b(skiing?|skis?|skied|snowboarding?)\b': "winter_sports",
        }
        
        # Activity type aliases - map common terms to stored activity types
        self.activity_type_aliases = {
            "ride": ["cycling", "virtual_ride"],
            "rides": ["cycling", "virtual_ride"],
            "cycle": ["cycling", "virtual_ride"],
            "cycles": ["cycling", "virtual_ride"],
            "run": ["running", "treadmill_running"],
            "runs": ["running", "treadmill_running"],
            "bike": ["cycling", "virtual_ride"],
            "bikes": ["cycling", "virtual_ride"],
            "swim": ["swimming"],
            "swims": ["swimming"],
            "walk": ["walking"],
            "walks": ["walking"],
            "workout": ["strength_training", "running", "cycling"],
            "workouts": ["strength_training", "running", "cycling"],
            "exercise": ["running", "cycling", "strength_training", "walking"],
            "exercises": ["running", "cycling", "strength_training", "walking"],
            "training": ["running", "cycling", "strength_training"],
            "cardio": ["running", "cycling", "virtual_ride", "treadmill_running"],
            "indoor": ["virtual_ride", "treadmill_running", "strength_training"],
            "outdoor": ["running", "cycling", "walking"],
            "session": ["strength_training", "running", "cycling"],
            "sessions": ["strength_training", "running", "cycling"],
        }
        
        # Query type patterns
        self.query_type_patterns = {
            r'\b(average|mean|total|sum|count|how\s+many)\b': "aggregation",
            r'\b(best|worst|fastest|slowest|longest|shortest|highest|lowest)\b': "comparison", 
            r'\b(yesterday|today|last\s+week|this\s+month)\b': "temporal",
            r'\b(trend|progress|improvement|compare|analysis)\b': "analysis",
        }
    
    def process_query(self, query: str) -> QueryContext:
        """Process a natural language query to extract context and filters."""
        query_lower = query.lower()
        
        # Initialize context
        context = QueryContext(
            original_query=query,
            enhanced_query=query
        )
        
        # Extract temporal filters
        temporal_filter = self._extract_temporal_filter(query_lower)
        if temporal_filter:
            context.temporal_filter = temporal_filter
            context.has_temporal_context = True
        
        # Extract metric filters
        metric_filters = self._extract_metric_filters(query_lower)
        if metric_filters:
            context.metric_filters = metric_filters
            context.has_metric_context = True
        
        # Extract activity type filter
        activity_type = self._extract_activity_type(query_lower)
        if activity_type:
            context.activity_type_filter = activity_type
        
        # Determine query type
        context.query_type = self._determine_query_type(query_lower)
        
        # Enhance query with temporal context
        context.enhanced_query = self._enhance_query(context)
        
        logger.debug(f"Processed query: {query} -> {context.dict()}")
        return context
    
    def _extract_temporal_filter(self, query: str) -> Optional[TemporalFilter]:
        """Extract temporal filter from query."""
        for pattern, handler in self.temporal_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    if "days? ago" in pattern or "weeks? ago" in pattern or "months? ago" in pattern:
                        # Range patterns
                        start_date = handler(match)
                        end_date = datetime.now()
                        return TemporalFilter(start_date=start_date, end_date=end_date)
                    elif "over the past" in pattern or "in the last" in pattern:
                        # Range patterns
                        start_date = handler(match)
                        end_date = datetime.now()
                        return TemporalFilter(start_date=start_date, end_date=end_date)
                    else:
                        # Specific date patterns
                        specific_date = handler(match)
                        is_exact = any(day in pattern.lower() for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
                        return TemporalFilter(specific_date=specific_date, is_exact_date=is_exact)
                except Exception as e:
                    logger.warning(f"Failed to parse temporal pattern {pattern}: {str(e)}")
                    continue
        
        return None
    
    def _extract_metric_filters(self, query: str) -> List[MetricFilter]:
        """Extract metric filters from query."""
        filters = []
        for pattern, handler in self.metric_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    metric_filter = handler(match)
                    filters.append(metric_filter)
                except Exception as e:
                    logger.warning(f"Failed to parse metric pattern {pattern}: {str(e)}")
                    continue
        return filters
    
    def _extract_activity_type(self, query: str) -> Optional[str]:
        """Extract activity type from query with intelligent matching."""
        # First try exact pattern matching
        for pattern, activity_type in self.activity_type_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return activity_type
        
        # Then check for alias terms
        words = query.lower().split()
        for word in words:
            if word in self.activity_type_aliases:
                # Return the first matching activity type from aliases
                # This allows flexibility for terms that could match multiple activities
                return self.activity_type_aliases[word][0]
        
        return None
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query for optimization."""
        for pattern, query_type in self.query_type_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return query_type
        return "general"
    
    def _enhance_query(self, context: QueryContext) -> str:
        """Enhance query with temporal and contextual information."""
        enhanced_parts = [f"Current date: {datetime.now().strftime('%Y-%m-%d')}."]
        
        # Add temporal context
        if context.temporal_filter:
            if context.temporal_filter.specific_date:
                date_str = context.temporal_filter.specific_date.strftime('%Y-%m-%d')
                enhanced_parts.append(f"Query refers to activities on {date_str}.")
            elif context.temporal_filter.start_date and context.temporal_filter.end_date:
                start_str = context.temporal_filter.start_date.strftime('%Y-%m-%d')
                end_str = context.temporal_filter.end_date.strftime('%Y-%m-%d')
                enhanced_parts.append(f"Query refers to activities between {start_str} and {end_str}.")
        
        # Add activity type context with expanded descriptions
        if context.activity_type_filter:
            activity_descriptions = {
                "cycling": "cycling, bike rides, or virtual rides",
                "virtual_ride": "indoor cycling, virtual rides, trainer sessions, or Zwift rides",
                "running": "running, jogging, or outdoor runs",
                "treadmill_running": "treadmill running or indoor running",
                "swimming": "swimming or pool activities",
                "walking": "walking, hiking, or trekking",
                "strength_training": "strength training, weight lifting, gym workouts, or resistance training"
            }
            description = activity_descriptions.get(
                context.activity_type_filter, 
                context.activity_type_filter
            )
            enhanced_parts.append(f"Focus on {description}.")
        
        # Add original query
        enhanced_parts.append(context.original_query)
        
        return " ".join(enhanced_parts)
    
    def create_pinecone_filter(self, context: QueryContext) -> Optional[Dict[str, Any]]:
        """Create Pinecone metadata filter from query context."""
        filter_dict = {}
        
        # Activity type filter with support for multiple possible types
        if context.activity_type_filter:
            # Map activity types to possible variations stored in the database
            activity_variations = {
                "cycling": ["cycling", "virtual_ride", "road_biking", "mountain_biking"],
                "virtual_ride": ["virtual_ride", "indoor_cycling", "cycling"],
                "running": ["running", "treadmill_running", "trail_running"],
                "treadmill_running": ["treadmill_running", "running"],
                "swimming": ["swimming", "lap_swimming", "open_water_swimming"],
                "walking": ["walking", "hiking"],
                "strength_training": ["strength_training", "weight_training"]
            }
            
            # Get variations or use the single type
            variations = activity_variations.get(context.activity_type_filter, [context.activity_type_filter])
            
            # Use $in operator if multiple variations exist
            if len(variations) > 1:
                filter_dict["activity_type"] = {"$in": variations}
            else:
                filter_dict["activity_type"] = context.activity_type_filter
        
        # Temporal filter
        if context.temporal_filter:
            if context.temporal_filter.specific_date:
                date_str = context.temporal_filter.specific_date.strftime('%Y-%m-%d')
                filter_dict["date"] = date_str
            elif context.temporal_filter.start_date and context.temporal_filter.end_date:
                filter_dict["timestamp"] = {
                    "$gte": int(context.temporal_filter.start_date.timestamp()),
                    "$lte": int(context.temporal_filter.end_date.timestamp())
                }
        
        # Metric filters
        for metric_filter in context.metric_filters:
            field_name = metric_filter.field
            if metric_filter.operator == "gt":
                filter_dict[field_name] = {"$gt": metric_filter.value}
            elif metric_filter.operator == "lt":
                filter_dict[field_name] = {"$lt": metric_filter.value}
            elif metric_filter.operator == "gte":
                filter_dict[field_name] = {"$gte": metric_filter.value}
            elif metric_filter.operator == "lte":
                filter_dict[field_name] = {"$lte": metric_filter.value}
            elif metric_filter.operator == "eq":
                filter_dict[field_name] = metric_filter.value
            elif metric_filter.operator == "between" and metric_filter.value_max:
                filter_dict[field_name] = {
                    "$gte": metric_filter.value,
                    "$lte": metric_filter.value_max
                }
        
        return filter_dict if filter_dict else None
    
    def get_search_limit(self, context: QueryContext) -> int:
        """Get appropriate search limit based on query type."""
        limits = {
            "temporal": 20,      # Recent activities need more context
            "aggregation": 100,  # Need many activities for calculations
            "comparison": 50,    # Need good sample for comparisons  
            "analysis": 50,      # Need substantial data for trends
            "general": 15        # Default limit
        }
        return limits.get(context.query_type, 15)
    
    # Helper methods for date calculations
    def _start_of_week(self) -> datetime:
        """Get start of current week (Monday)."""
        today = datetime.now()
        days_since_monday = today.weekday()
        return today - timedelta(days=days_since_monday)
    
    def _start_of_month(self) -> datetime:
        """Get start of current month."""
        today = datetime.now()
        return today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _start_of_year(self) -> datetime:
        """Get start of current year."""
        today = datetime.now()
        return today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _get_yesterday_time_of_day(self, time_of_day: str) -> datetime:
        """Get yesterday at specific time of day."""
        yesterday = datetime.now() - timedelta(days=1)
        if time_of_day == "morning":
            return yesterday.replace(hour=8, minute=0, second=0)
        elif time_of_day == "afternoon":
            return yesterday.replace(hour=14, minute=0, second=0)
        elif time_of_day == "evening":
            return yesterday.replace(hour=18, minute=0, second=0)
        return yesterday
    
    def _get_today_time_of_day(self, time_of_day: str) -> datetime:
        """Get today at specific time of day."""
        today = datetime.now()
        if time_of_day == "morning":
            return today.replace(hour=8, minute=0, second=0)
        elif time_of_day == "afternoon":
            return today.replace(hour=14, minute=0, second=0)
        elif time_of_day == "evening":
            return today.replace(hour=18, minute=0, second=0)
        return today
    
    def _parse_month(self, month_name: str, year_str: Optional[str] = None) -> datetime:
        """Parse month name with optional year."""
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        
        month = months.get(month_name.lower(), 1)
        year = int(year_str) if year_str else datetime.now().year
        
        return datetime(year=year, month=month, day=1)
    
    def _get_weekday_in_week(self, target_weekday: int, week: str) -> datetime:
        """Get specific weekday in current or last week.
        
        Args:
            target_weekday: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
            week: "this" or "last"
        """
        today = datetime.now()
        current_weekday = today.weekday()
        
        if week == "this":
            # Calculate days from today to target weekday
            days_diff = target_weekday - current_weekday
            if days_diff < 0:
                # Target day has already passed this week, so it was earlier in the week
                target_date = today + timedelta(days=days_diff)
            else:
                # Target day is today or later this week
                target_date = today + timedelta(days=days_diff)
        else:  # week == "last"
            # Go to last week first, then to target weekday
            last_week_start = today - timedelta(days=current_weekday + 7)  # Last Monday
            target_date = last_week_start + timedelta(days=target_weekday)
        
        # Set to start of day
        return target_date.replace(hour=0, minute=0, second=0, microsecond=0)