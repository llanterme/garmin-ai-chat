"""
Temporal query processor for parsing and handling time-based queries.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging


class TemporalFilter(BaseModel):
    """Temporal filter for database queries."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    date_range_days: Optional[int] = None
    specific_date: Optional[datetime] = None


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


class TemporalQueryProcessor:
    """
    Processes natural language queries to extract temporal and metric filters.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Temporal patterns
        self.temporal_patterns = {
            # Relative days with time of day
            r'\byesterday\s+(morning|afternoon|evening)\b': lambda: datetime.now() - timedelta(days=1),
            r'\bthis\s+morning\b': lambda: datetime.now(),
            r'\bthis\s+afternoon\b': lambda: datetime.now(),
            r'\bthis\s+evening\b': lambda: datetime.now(),
            r'\btonight\b': lambda: datetime.now(),
            
            # Relative days
            r'\byesterday\b': lambda: datetime.now() - timedelta(days=1),
            r'\btoday\b': lambda: datetime.now(),
            r'\b(\d+)\s+days?\s+ago\b': lambda m: datetime.now() - timedelta(days=int(m.group(1))),
            
            # Relative weeks
            r'\blast\s+week\b': lambda: datetime.now() - timedelta(weeks=1),
            r'\bthis\s+week\b': lambda: self._start_of_week(),
            r'\b(\d+)\s+weeks?\s+ago\b': lambda m: datetime.now() - timedelta(weeks=int(m.group(1))),
            
            # Relative months
            r'\blast\s+month\b': lambda: datetime.now() - timedelta(days=30),
            r'\bthis\s+month\b': lambda: self._start_of_month(),
            r'\b(\d+)\s+months?\s+ago\b': lambda m: datetime.now() - timedelta(days=int(m.group(1)) * 30),
            
            # Relative years
            r'\blast\s+year\b': lambda: datetime.now() - timedelta(days=365),
            r'\bthis\s+year\b': lambda: self._start_of_year(),
            
            # Specific dates
            r'\bon\s+(\d{4}-\d{2}-\d{2})\b': lambda m: datetime.strptime(m.group(1), '%Y-%m-%d'),
            r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b': self._parse_month,
        }
        
        # Metric patterns
        self.metric_patterns = {
            # Power metrics
            r'\bpower\s+(above|over|greater\s+than)\s+(\d+)\s*w?\b': ('avg_power_watts', 'gt'),
            r'\bpower\s+(below|under|less\s+than)\s+(\d+)\s*w?\b': ('avg_power_watts', 'lt'),
            r'\baverage\s+power\s+(above|over|greater\s+than)\s+(\d+)\s*w?\b': ('avg_power_watts', 'gt'),
            r'\bavg\s+power\s+(above|over|greater\s+than)\s+(\d+)\s*w?\b': ('avg_power_watts', 'gt'),
            
            # Heart rate metrics
            r'\bheart\s+rate\s+(above|over|greater\s+than)\s+(\d+)\s*bpm?\b': ('avg_heartrate', 'gt'),
            r'\bheart\s+rate\s+(below|under|less\s+than)\s+(\d+)\s*bpm?\b': ('avg_heartrate', 'lt'),
            r'\bhr\s+(above|over|greater\s+than)\s+(\d+)\s*bpm?\b': ('avg_heartrate', 'gt'),
            r'\bhr\s+(below|under|less\s+than)\s+(\d+)\s*bpm?\b': ('avg_heartrate', 'lt'),
            
            # Speed metrics
            r'\bspeed\s+(above|over|greater\s+than)\s+(\d+(?:\.\d+)?)\s*km/?h?\b': ('avg_speed_kmh', 'gt'),
            r'\bspeed\s+(below|under|less\s+than)\s+(\d+(?:\.\d+)?)\s*km/?h?\b': ('avg_speed_kmh', 'lt'),
            r'\bpace\s+(?:faster\s+than|below)\s+(\d+):(\d+)\b': ('avg_pace_min_per_km', 'lt'),  # pace is inverse
            r'\bpace\s+(?:slower\s+than|above)\s+(\d+):(\d+)\b': ('avg_pace_min_per_km', 'gt'),
            
            # Distance metrics
            r'\bdistance\s+(above|over|greater\s+than)\s+(\d+(?:\.\d+)?)\s*km?\b': ('distance_km', 'gt'),
            r'\bdistance\s+(below|under|less\s+than)\s+(\d+(?:\.\d+)?)\s*km?\b': ('distance_km', 'lt'),
            r'\blonger\s+than\s+(\d+(?:\.\d+)?)\s*km?\b': ('distance_km', 'gt'),
            r'\bshorter\s+than\s+(\d+(?:\.\d+)?)\s*km?\b': ('distance_km', 'lt'),
            
            # Duration metrics
            r'\bduration\s+(above|over|greater\s+than)\s+(\d+)\s*(hours?|hrs?|h)\b': ('duration_minutes', 'gt'),
            r'\bduration\s+(below|under|less\s+than)\s+(\d+)\s*(hours?|hrs?|h)\b': ('duration_minutes', 'lt'),
            r'\blonger\s+than\s+(\d+)\s*(hours?|hrs?|h)\b': ('duration_minutes', 'gt'),
            r'\bshorter\s+than\s+(\d+)\s*(hours?|hrs?|h)\b': ('duration_minutes', 'lt'),
        }
        
        # Activity type patterns
        self.activity_patterns = {
            r'\b(run|running|runs)\b': 'run',
            r'\b(ride|cycling|bike|rides)\b': 'ride',
            r'\b(swim|swimming|swims)\b': 'swim',
            r'\b(walk|walking|walks)\b': 'walk',
            r'\b(hike|hiking|hikes)\b': 'hike',
            r'\b(workout|training|strength)\b': 'workout',
        }
        
        # Activity type groups - maps query types to Strava activity types
        self.activity_type_groups = {
            'run': ['run'],
            'ride': ['ride', 'virtualride'],  # Include both outdoor and indoor rides
            'swim': ['swim'],
            'walk': ['walk'],
            'hike': ['hike'],
            'workout': ['workout', 'weighttraining', 'crosstraining'],
        }
    
    def process_query(self, query: str, current_date: Optional[datetime] = None) -> QueryContext:
        """
        Process a natural language query to extract temporal and metric filters.
        
        Args:
            query: Natural language query
            current_date: Current date for relative calculations
            
        Returns:
            QueryContext with parsed filters and enhanced query
        """
        if current_date is None:
            current_date = datetime.now()
        
        query_lower = query.lower()
        context = QueryContext(
            original_query=query,
            enhanced_query=query
        )
        
        # Extract temporal filters
        temporal_filter = self._extract_temporal_filter(query_lower, current_date)
        if temporal_filter:
            context.temporal_filter = temporal_filter
            context.has_temporal_context = True
        
        # Extract metric filters
        metric_filters = self._extract_metric_filters(query_lower)
        if metric_filters:
            context.metric_filters = metric_filters
            context.has_metric_context = True
        
        # Extract activity type
        activity_type = self._extract_activity_type(query_lower)
        if activity_type:
            context.activity_type_filter = activity_type
        
        # Enhance query with temporal context
        context.enhanced_query = self._enhance_query_with_context(query, context, current_date)
        
        return context
    
    def _extract_temporal_filter(self, query: str, current_date: datetime) -> Optional[TemporalFilter]:
        """Extract temporal filter from query."""
        for pattern, date_func in self.temporal_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    if callable(date_func):
                        if pattern in [r'\b(\d+)\s+days?\s+ago\b', r'\b(\d+)\s+weeks?\s+ago\b', r'\b(\d+)\s+months?\s+ago\b']:
                            target_date = date_func(match)
                        elif pattern == r'\bon\s+(\d{4}-\d{2}-\d{2})\b':
                            target_date = date_func(match)
                        else:
                            target_date = date_func()
                    else:
                        target_date = date_func
                    
                    # Create appropriate date range
                    if any(word in pattern for word in ['yesterday', 'today', 'morning', 'afternoon', 'evening', 'tonight', 'on ']):
                        # Specific day (or time of day)
                        return TemporalFilter(
                            start_date=target_date.replace(hour=0, minute=0, second=0, microsecond=0),
                            end_date=target_date.replace(hour=23, minute=59, second=59, microsecond=999999),
                            specific_date=target_date
                        )
                    elif 'week' in pattern:
                        # Week range
                        start_date = target_date - timedelta(days=7)
                        return TemporalFilter(
                            start_date=start_date,
                            end_date=current_date,
                            date_range_days=7
                        )
                    elif 'month' in pattern:
                        # Month range
                        start_date = target_date - timedelta(days=30)
                        return TemporalFilter(
                            start_date=start_date,
                            end_date=current_date,
                            date_range_days=30
                        )
                    elif 'year' in pattern:
                        # Year range
                        start_date = target_date - timedelta(days=365)
                        return TemporalFilter(
                            start_date=start_date,
                            end_date=current_date,
                            date_range_days=365
                        )
                        
                except Exception as e:
                    self.logger.warning(f"Error parsing temporal pattern {pattern}: {e}")
                    continue
        
        return None
    
    def _extract_metric_filters(self, query: str) -> List[MetricFilter]:
        """Extract metric filters from query."""
        filters = []
        
        for pattern, (field, operator) in self.metric_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    if field == 'avg_pace_min_per_km':
                        # Handle pace format (MM:SS) - groups are 1 and 2 for non-capturing group
                        minutes = int(match.group(1))
                        seconds = int(match.group(2))
                        value = minutes + seconds / 60.0
                    elif field == 'duration_minutes' and len(match.groups()) >= 2:
                        # Handle duration in hours
                        hours = float(match.group(2))
                        value = hours * 60  # Convert to minutes
                    elif 'longer\s+than' in pattern or 'shorter\s+than' in pattern:
                        # Handle distance patterns - group 1 for non-capturing group
                        value = float(match.group(1))
                    else:
                        # Regular numeric value - group 2 for patterns with capturing group for operator
                        value = float(match.group(2))
                    
                    # Convert speed from km/h to m/s for storage consistency
                    if field == 'avg_speed_kmh':
                        field = 'avg_speed'  # Store as m/s internally
                        value = value / 3.6
                    
                    filters.append(MetricFilter(
                        field=field,
                        operator=operator,
                        value=value
                    ))
                    
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing metric from pattern {pattern}: {e}")
                    continue
        
        return filters
    
    def _extract_activity_type(self, query: str) -> Optional[str]:
        """Extract activity type from query."""
        for pattern, activity_type in self.activity_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return activity_type
        return None
    
    def _enhance_query_with_context(self, query: str, context: QueryContext, current_date: datetime) -> str:
        """Enhance query with temporal context for better embedding."""
        enhanced = query
        
        # Add current date context
        date_str = current_date.strftime("%Y-%m-%d")
        enhanced = f"Current date: {date_str}. {enhanced}"
        
        # Add temporal context
        if context.temporal_filter:
            if context.temporal_filter.specific_date:
                specific_date = context.temporal_filter.specific_date.strftime("%Y-%m-%d")
                enhanced = enhanced.replace("yesterday", f"on {specific_date}")
                enhanced = enhanced.replace("today", f"on {date_str}")
            elif context.temporal_filter.start_date:
                start_date = context.temporal_filter.start_date.strftime("%Y-%m-%d")
                end_date = context.temporal_filter.end_date.strftime("%Y-%m-%d")
                enhanced = f"{enhanced} (date range: {start_date} to {end_date})"
        
        # Add activity type context
        if context.activity_type_filter:
            enhanced = f"{enhanced} (activity type: {context.activity_type_filter})"
        
        return enhanced
    
    def create_pinecone_filter(self, context: QueryContext) -> Dict[str, Any]:
        """
        Create Pinecone metadata filter from query context.
        
        Args:
            context: Parsed query context
            
        Returns:
            Pinecone filter dictionary
        """
        filters = {}
        
        # Temporal filters
        if context.temporal_filter:
            if context.temporal_filter.specific_date:
                # Specific date - filter by date string
                date_str = context.temporal_filter.specific_date.strftime("%Y-%m-%d")
                filters["date"] = {"$eq": date_str}
            elif context.temporal_filter.start_date and context.temporal_filter.end_date:
                # Date range
                start_timestamp = int(context.temporal_filter.start_date.timestamp())
                end_timestamp = int(context.temporal_filter.end_date.timestamp())
                filters["timestamp"] = {
                    "$gte": start_timestamp,
                    "$lte": end_timestamp
                }
        
        # Activity type filter
        if context.activity_type_filter:
            # Use activity type groups to support multiple related types
            activity_types = self.activity_type_groups.get(context.activity_type_filter, [context.activity_type_filter])
            
            if len(activity_types) == 1:
                # Single type - use exact match
                filters["activity_type"] = {"$eq": activity_types[0]}
            else:
                # Multiple types - use $in operator
                filters["activity_type"] = {"$in": activity_types}
        
        # Metric filters
        for metric_filter in context.metric_filters:
            field = metric_filter.field
            operator = metric_filter.operator
            value = metric_filter.value
            
            if operator == "gt":
                filters[field] = {"$gt": value}
            elif operator == "lt":
                filters[field] = {"$lt": value}
            elif operator == "gte":
                filters[field] = {"$gte": value}
            elif operator == "lte":
                filters[field] = {"$lte": value}
            elif operator == "eq":
                filters[field] = {"$eq": value}
            elif operator == "between" and metric_filter.value_max:
                filters[field] = {
                    "$gte": value,
                    "$lte": metric_filter.value_max
                }
        
        return filters
    
    def _start_of_week(self) -> datetime:
        """Get start of current week (Monday)."""
        now = datetime.now()
        return now - timedelta(days=now.weekday())
    
    def _start_of_month(self) -> datetime:
        """Get start of current month."""
        now = datetime.now()
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _start_of_year(self) -> datetime:
        """Get start of current year."""
        now = datetime.now()
        return now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _parse_month(self, match) -> datetime:
        """Parse month name to datetime."""
        month_name = match.group(1).lower()
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        month_num = month_map.get(month_name)
        if month_num:
            current_year = datetime.now().year
            return datetime(current_year, month_num, 1)
        
        return datetime.now()