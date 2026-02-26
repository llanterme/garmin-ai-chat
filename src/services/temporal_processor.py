"""Temporal query processor for parsing and handling time-based queries in fitness conversations."""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

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
    activity_type_filter: Optional[Union[str, List[str]]] = None
    has_temporal_context: bool = False
    has_metric_context: bool = False
    query_type: str = "general"  # "temporal", "aggregation", "comparison", "analysis"


class TemporalQueryProcessor:
    """Processes natural language queries to extract temporal and metric filters."""

    def __init__(self):
        self.logger = logger

        # Temporal patterns with lambda functions that capture match groups.
        # ORDER MATTERS: patterns are checked in insertion order, first match wins.
        # Broader time spans (weeks/months/years) must come before narrow ones
        # (time-of-day) so that "my morning rides this week" matches "this week",
        # not "this morning".
        self.temporal_patterns = {
            # --- 1. Weekday-in-week patterns (most specific) ---
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

            # --- 2. "last <weekday>" without "week" (e.g., "last Monday") ---
            r'\blast\s+(monday)\b': lambda m: self._get_last_specific_weekday(0),
            r'\blast\s+(tuesday)\b': lambda m: self._get_last_specific_weekday(1),
            r'\blast\s+(wednesday)\b': lambda m: self._get_last_specific_weekday(2),
            r'\blast\s+(thursday)\b': lambda m: self._get_last_specific_weekday(3),
            r'\blast\s+(friday)\b': lambda m: self._get_last_specific_weekday(4),
            r'\blast\s+(saturday)\b': lambda m: self._get_last_specific_weekday(5),
            r'\blast\s+(sunday)\b': lambda m: self._get_last_specific_weekday(6),

            # --- 3. Relative weeks/months/years (return tuples for date ranges) ---
            r'\blast\s+week\b': lambda m=None: self._get_last_week_range(),
            r'\bpast\s+week\b': lambda m=None: (datetime.now() - timedelta(days=7), datetime.now()),
            r'\bthis\s+week\b': lambda m=None: self._get_this_week_range(),
            r'\b(\d+)\s+weeks?\s+ago\b': lambda m: datetime.now() - timedelta(weeks=int(m.group(1))),

            r'\blast\s+month\b': lambda m=None: self._get_last_month_range(),
            r'\bpast\s+month\b': lambda m=None: (datetime.now() - timedelta(days=30), datetime.now()),
            r'\bthis\s+month\b': lambda m=None: self._get_this_month_range(),
            r'\b(\d+)\s+months?\s+ago\b': lambda m: datetime.now() - timedelta(days=int(m.group(1)) * 30),

            r'\blast\s+year\b': lambda m=None: self._get_last_year_range(),
            r'\bthis\s+year\b': lambda m=None: self._get_this_year_range(),

            # --- 4. Date ranges ---
            r'\b(?:last|past)\s+(\d+)\s+days?\b': lambda m: (datetime.now() - timedelta(days=int(m.group(1))), datetime.now()),
            r'\bover\s+the\s+past\s+(\d+)\s+days?\b': lambda m: datetime.now() - timedelta(days=int(m.group(1))),
            r'\bin\s+the\s+last\s+(\d+)\s+(?:days?|weeks?)\b': lambda m: datetime.now() - timedelta(days=int(m.group(1)) * (7 if 'week' in m.group(0) else 1)),

            # --- 5. Specific dates ---
            r'\bon\s+(\d{4}-\d{2}-\d{2})\b': lambda m: datetime.strptime(m.group(1), '%Y-%m-%d'),
            r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?\b':
                lambda m: self._parse_month(m.group(1), m.group(2)),

            # --- 6. Relative days (yesterday time-of-day BEFORE plain yesterday) ---
            r'\byesterday\s+(morning|afternoon|evening)\b': lambda m: self._get_yesterday_time_of_day(m.group(1)),
            r'\byesterday\b': lambda m=None: datetime.now() - timedelta(days=1),
            r'\btoday\b': lambda m=None: datetime.now(),
            r'\b(\d+)\s+days?\s+ago\b': lambda m: datetime.now() - timedelta(days=int(m.group(1))),

            # --- 7. Time of day (lowest priority) ---
            r'\bthis\s+(morning|afternoon|evening)\b': lambda m: self._get_today_time_of_day(m.group(1)),
            r'\btonight\b': lambda m=None: datetime.now().replace(hour=18, minute=0, second=0),
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
            r'\b(trail\s*runs?(?:ning)?)\b': "trail_running",
            r'\b(running?|runs?|ran|jogging?|jogs?|jogged|marathon|half\s*marathon|5k|10k|sprint(?:ing)?|sprints?)\b': "running",
            r'\btreadmill\s*(running?)?\b': "treadmill_running",

            # Cycling and related activities
            r'\b(mountain\s*bik(?:e|ing)|mtb)\b': "mountain_biking",
            r'\b(virtual\s*rides?|indoor\s*(cycling?|rides?|bike)|zwift|trainer|turbo|peloton)\b': "virtual_ride",
            r'\b(spin|spinning)\b': "virtual_ride",
            r'\b(cycling?|cycles?|cycled|rides?|rode|ridden|bike|biking|biked|pedal|pedaling|road\s*bik(?:e|ing)|road\s*cycling?|gravel)\b': "cycling",

            # Swimming
            r'\b(swimming?|swims?|swam|swum|pool|laps?)\b': "swimming",

            # Walking and hiking
            r'\b(walking?|walks?|walked|hiking?|hikes?|hiked|trekking?)\b': "walking",

            # Strength training (note: "workout" deliberately excluded — handled by aliases)
            r'\b(strength|weights?|lifting|gym|resistance|crossfit|hiit|bootcamp|boot\s*camp)\b': "strength_training",

            # Other activities
            r'\b(yoga|pilates|stretching?)\b': "yoga",
            r'\b(rowing?|rows?|rowed|erg)\b': "rowing",
            r'\b(skiing?|skis?|skied|snowboarding?)\b': "winter_sports",
        }

        # Activity type aliases - map common terms to stored activity types.
        # When a word matches here, the FULL list is returned so that
        # create_pinecone_filter can build a proper $in query.
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
            "peloton": ["virtual_ride"],
            "intervals": ["running", "cycling"],
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
        # "last N <activity_type>" means "N most recent activities" — no date filter needed.
        # Without this check, "today" in "suggest a workout today" would create an
        # exact-date filter that excludes all historical activities.
        activity_words = (
            r'runs?|rides?|cycles?|cycling|swims?|walks?|workouts?|sessions?'
            r'|activities|exercises?|hikes?|rows?|jogs?|marathons?|sprints?'
            r'|laps?|lifts?|treks?|spins?'
        )
        if re.search(rf'\blast\s+\d+\s+(?:{activity_words})\b', query, re.IGNORECASE):
            return None

        for pattern, handler in self.temporal_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    result = handler(match)

                    # Tuple result = explicit date range (from range helper methods)
                    if isinstance(result, tuple) and len(result) == 2:
                        return TemporalFilter(start_date=result[0], end_date=result[1])

                    # Legacy range detection for "N days/weeks/months ago", "over the past", "in the last"
                    if "days? ago" in pattern or "weeks? ago" in pattern or "months? ago" in pattern:
                        start_date = result
                        end_date = datetime.now()
                        return TemporalFilter(start_date=start_date, end_date=end_date)
                    elif "over the past" in pattern or "in the last" in pattern:
                        start_date = result
                        end_date = datetime.now()
                        return TemporalFilter(start_date=start_date, end_date=end_date)
                    else:
                        # Specific date patterns
                        specific_date = result
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

    def _extract_activity_type(self, query: str) -> Optional[Union[str, List[str]]]:
        """Extract activity type from query with intelligent matching."""
        # First try exact pattern matching (returns single type)
        for pattern, activity_type in self.activity_type_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return activity_type

        # Then check for alias terms (may return multiple types)
        words = query.lower().split()
        for word in words:
            if word in self.activity_type_aliases:
                aliases = self.activity_type_aliases[word]
                if len(aliases) == 1:
                    return aliases[0]
                return aliases

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
                "trail_running": "trail running or off-road runs",
                "mountain_biking": "mountain biking or off-road cycling",
                "swimming": "swimming or pool activities",
                "walking": "walking, hiking, or trekking",
                "strength_training": "strength training, weight lifting, gym workouts, or resistance training",
                "yoga": "yoga, pilates, or stretching",
                "rowing": "rowing or ergometer sessions",
                "winter_sports": "skiing, snowboarding, or winter sports",
            }

            if isinstance(context.activity_type_filter, list):
                descriptions = []
                for at in context.activity_type_filter:
                    descriptions.append(activity_descriptions.get(at, at))
                description = " or ".join(descriptions)
            else:
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
                "virtual_ride": ["virtual_ride", "indoor_cycling"],
                "running": ["running", "treadmill_running", "trail_running"],
                "treadmill_running": ["treadmill_running", "running"],
                "trail_running": ["trail_running", "running"],
                "mountain_biking": ["mountain_biking", "cycling"],
                "swimming": ["swimming", "lap_swimming", "open_water_swimming"],
                "walking": ["walking", "hiking"],
                "strength_training": ["strength_training", "weight_training"],
                "yoga": ["yoga", "pilates"],
                "rowing": ["rowing", "indoor_rowing"],
                "winter_sports": ["winter_sports", "skiing", "snowboarding", "cross_country_skiing"],
            }

            # Collect all types to include
            if isinstance(context.activity_type_filter, list):
                # Multiple types from alias — gather all variations
                all_types = set()
                for activity_type in context.activity_type_filter:
                    variations = activity_variations.get(activity_type, [activity_type])
                    all_types.update(variations)
                all_types = sorted(all_types)
            else:
                # Single type — get its variations
                all_types = activity_variations.get(
                    context.activity_type_filter, [context.activity_type_filter]
                )

            if len(all_types) > 1:
                filter_dict["activity_type"] = {"$in": all_types}
            else:
                filter_dict["activity_type"] = all_types[0] if all_types else context.activity_type_filter

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

    def _get_this_week_range(self) -> tuple:
        """Get date range for current week (Monday 00:00 to now)."""
        start = self._start_of_week().replace(hour=0, minute=0, second=0, microsecond=0)
        return (start, datetime.now())

    def _get_last_week_range(self) -> tuple:
        """Get date range for last calendar week (Monday 00:00 to Sunday 23:59)."""
        this_monday = self._start_of_week().replace(hour=0, minute=0, second=0, microsecond=0)
        last_monday = this_monday - timedelta(weeks=1)
        last_sunday = this_monday - timedelta(seconds=1)
        return (last_monday, last_sunday)

    def _get_this_month_range(self) -> tuple:
        """Get date range for current month (1st 00:00 to now)."""
        start = self._start_of_month()
        return (start, datetime.now())

    def _get_last_month_range(self) -> tuple:
        """Get date range for last calendar month."""
        first_of_this_month = self._start_of_month()
        last_day_prev_month = first_of_this_month - timedelta(days=1)
        first_of_last_month = last_day_prev_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_of_last_month = last_day_prev_month.replace(hour=23, minute=59, second=59, microsecond=999999)
        return (first_of_last_month, end_of_last_month)

    def _get_this_year_range(self) -> tuple:
        """Get date range for current year (Jan 1 00:00 to now)."""
        start = self._start_of_year()
        return (start, datetime.now())

    def _get_last_year_range(self) -> tuple:
        """Get date range for last calendar year."""
        now = datetime.now()
        start = datetime(year=now.year - 1, month=1, day=1, hour=0, minute=0, second=0)
        end = datetime(year=now.year - 1, month=12, day=31, hour=23, minute=59, second=59)
        return (start, end)

    def _get_last_specific_weekday(self, target_weekday: int) -> datetime:
        """Get the most recent past occurrence of a specific weekday.

        Args:
            target_weekday: 0=Monday through 6=Sunday
        """
        today = datetime.now()
        current_weekday = today.weekday()
        days_back = (current_weekday - target_weekday) % 7
        if days_back == 0:
            days_back = 7  # "last Monday" on a Monday means the previous Monday
        target_date = today - timedelta(days=days_back)
        return target_date.replace(hour=0, minute=0, second=0, microsecond=0)

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
