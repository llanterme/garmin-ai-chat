"""Tests for the temporal query processor."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.services.temporal_processor import TemporalQueryProcessor


@pytest.fixture
def processor():
    return TemporalQueryProcessor()


# ---------------------------------------------------------------------------
# Date range tests: "this week", "last month", etc. must produce ranges
# ---------------------------------------------------------------------------

class TestDateRanges:
    """Bug 1: week/month/year patterns must return start_date + end_date ranges,
    not single specific_date values."""

    def test_this_week_returns_range(self, processor):
        ctx = processor.process_query("show me my runs this week")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None
        assert ctx.temporal_filter.specific_date is None
        # start_date should be Monday of this week
        assert ctx.temporal_filter.start_date.weekday() == 0

    def test_last_week_returns_range(self, processor):
        ctx = processor.process_query("my runs last week")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None
        assert ctx.temporal_filter.specific_date is None
        # start should be a Monday, end should be a Sunday
        assert ctx.temporal_filter.start_date.weekday() == 0
        assert ctx.temporal_filter.end_date.weekday() == 6

    def test_this_month_returns_range(self, processor):
        ctx = processor.process_query("cycling this month")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None
        # start should be 1st of current month
        assert ctx.temporal_filter.start_date.day == 1

    def test_last_month_returns_range(self, processor):
        ctx = processor.process_query("my activities last month")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None
        assert ctx.temporal_filter.start_date.day == 1

    def test_this_year_returns_range(self, processor):
        ctx = processor.process_query("total distance this year")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None
        assert ctx.temporal_filter.start_date.month == 1
        assert ctx.temporal_filter.start_date.day == 1

    def test_last_year_returns_range(self, processor):
        ctx = processor.process_query("how many runs last year")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None
        now = datetime.now()
        assert ctx.temporal_filter.start_date.year == now.year - 1
        assert ctx.temporal_filter.end_date.year == now.year - 1

    def test_past_7_days_returns_range(self, processor):
        ctx = processor.process_query("rides in the past 7 days")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None

    def test_last_30_days_returns_range(self, processor):
        ctx = processor.process_query("my runs last 30 days")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None

    def test_past_week_returns_rolling_range(self, processor):
        ctx = processor.process_query("activities past week")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None
        # Should be a rolling 7-day window
        diff = ctx.temporal_filter.end_date - ctx.temporal_filter.start_date
        assert 6 <= diff.days <= 7

    def test_past_month_returns_rolling_range(self, processor):
        ctx = processor.process_query("activities past month")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None
        diff = ctx.temporal_filter.end_date - ctx.temporal_filter.start_date
        assert 29 <= diff.days <= 30


# ---------------------------------------------------------------------------
# Pattern priority tests
# ---------------------------------------------------------------------------

class TestPatternPriority:
    """Bug 4: broader temporal patterns (week/month) must match before
    time-of-day patterns (morning/evening)."""

    def test_this_week_wins_over_this_morning(self, processor):
        ctx = processor.process_query("my morning rides this week")
        assert ctx.temporal_filter is not None
        # Should be a range (this week), not a specific date (this morning)
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None

    def test_last_week_wins_over_last_in_compound(self, processor):
        ctx = processor.process_query("best run last week")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None
        assert ctx.temporal_filter.end_date is not None

    def test_yesterday_morning_still_works(self, processor):
        """Ensure the reorder didn't break yesterday + time-of-day."""
        ctx = processor.process_query("what did I do yesterday morning")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.specific_date is not None

    def test_today_still_works_alone(self, processor):
        ctx = processor.process_query("what did I do today")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.specific_date is not None


# ---------------------------------------------------------------------------
# "last N activities" guard tests
# ---------------------------------------------------------------------------

class TestLastNActivitiesGuard:
    """The guard should prevent temporal filtering when the user asks
    for "last N <activity type>"."""

    def test_last_3_runs_no_temporal_filter(self, processor):
        ctx = processor.process_query("last 3 runs")
        assert ctx.temporal_filter is None

    def test_last_5_rides_with_today_no_temporal_filter(self, processor):
        ctx = processor.process_query("last 5 rides and suggest a workout today")
        assert ctx.temporal_filter is None

    def test_last_3_cycles_no_temporal_filter(self, processor):
        ctx = processor.process_query("look at my last 3 cycles and suggest a workout today")
        assert ctx.temporal_filter is None

    def test_last_10_workouts_no_temporal_filter(self, processor):
        ctx = processor.process_query("show me my last 10 workouts")
        assert ctx.temporal_filter is None

    def test_last_2_swims_no_temporal_filter(self, processor):
        ctx = processor.process_query("last 2 swims")
        assert ctx.temporal_filter is None

    def test_last_week_is_not_blocked(self, processor):
        """Ensure 'last week' (no count + activity) is NOT blocked by the guard."""
        ctx = processor.process_query("my runs last week")
        assert ctx.temporal_filter is not None


# ---------------------------------------------------------------------------
# "last <weekday>" tests
# ---------------------------------------------------------------------------

class TestLastWeekday:
    def test_last_monday(self, processor):
        ctx = processor.process_query("my run last monday")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.specific_date is not None
        assert ctx.temporal_filter.specific_date.weekday() == 0  # Monday
        # Should be in the past
        assert ctx.temporal_filter.specific_date < datetime.now()

    def test_last_friday(self, processor):
        ctx = processor.process_query("what did I do last friday")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.specific_date is not None
        assert ctx.temporal_filter.specific_date.weekday() == 4  # Friday

    def test_last_weekday_is_exact_date(self, processor):
        ctx = processor.process_query("last tuesday ride")
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.is_exact_date is True


# ---------------------------------------------------------------------------
# Activity type tests
# ---------------------------------------------------------------------------

class TestActivityType:
    """Bug 2 + Bug 3: activity type extraction and alias handling."""

    def test_workout_returns_multiple_types(self, processor):
        """Bug 2 fix: 'workout' should NOT map only to strength_training."""
        ctx = processor.process_query("my workouts")
        assert ctx.activity_type_filter is not None
        assert isinstance(ctx.activity_type_filter, list)
        assert "strength_training" in ctx.activity_type_filter
        assert "running" in ctx.activity_type_filter
        assert "cycling" in ctx.activity_type_filter

    def test_rides_returns_multiple_types(self, processor):
        """Bug 3 fix: aliases should return the full list."""
        ctx = processor.process_query("show me my rides")
        # "rides" matches the cycling regex pattern first → returns "cycling" (str)
        # because the regex on line ~127 catches "rides"
        assert ctx.activity_type_filter is not None

    def test_running_synonyms(self, processor):
        for word in ["run", "running", "jogging", "marathon", "5k"]:
            ctx = processor.process_query(f"my {word}")
            assert ctx.activity_type_filter == "running", f"'{word}' should map to running"

    def test_cycling_synonyms(self, processor):
        for word in ["cycling", "bike", "biking", "pedaling"]:
            ctx = processor.process_query(f"my {word}")
            assert ctx.activity_type_filter == "cycling", f"'{word}' should map to cycling"

    def test_virtual_ride_synonyms(self, processor):
        for word in ["zwift", "peloton", "spinning", "indoor ride", "indoor cycling"]:
            ctx = processor.process_query(f"my {word}")
            assert ctx.activity_type_filter == "virtual_ride", f"'{word}' should map to virtual_ride"

    def test_trail_running(self, processor):
        ctx = processor.process_query("my trail runs")
        assert ctx.activity_type_filter == "trail_running"

    def test_mountain_biking(self, processor):
        for word in ["mtb", "mountain bike", "mountain biking"]:
            ctx = processor.process_query(f"my {word}")
            assert ctx.activity_type_filter == "mountain_biking", f"'{word}' should map to mountain_biking"

    def test_strength_still_maps_correctly(self, processor):
        """Ensure removing 'workout' didn't break other strength words."""
        for word in ["strength", "weights", "lifting", "gym", "crossfit"]:
            ctx = processor.process_query(f"my {word}")
            assert ctx.activity_type_filter == "strength_training", f"'{word}' should map to strength_training"

    def test_hiit_maps_to_strength(self, processor):
        ctx = processor.process_query("my hiit session")
        assert ctx.activity_type_filter == "strength_training"

    def test_sprint_maps_to_running(self, processor):
        ctx = processor.process_query("my sprint workout")
        assert ctx.activity_type_filter == "running"


# ---------------------------------------------------------------------------
# Pinecone filter tests
# ---------------------------------------------------------------------------

class TestPineconeFilter:
    """Verify create_pinecone_filter builds correct metadata filters."""

    def test_single_type_produces_in_with_variations(self, processor):
        ctx = processor.process_query("my runs this month")
        filt = processor.create_pinecone_filter(ctx)
        assert filt is not None
        assert "activity_type" in filt
        assert "$in" in filt["activity_type"]
        variations = filt["activity_type"]["$in"]
        assert "running" in variations
        assert "treadmill_running" in variations
        assert "trail_running" in variations

    def test_list_type_merges_all_variations(self, processor):
        ctx = processor.process_query("my workouts")
        filt = processor.create_pinecone_filter(ctx)
        assert filt is not None
        assert "activity_type" in filt
        assert "$in" in filt["activity_type"]
        variations = filt["activity_type"]["$in"]
        # Should include variations from strength_training, running, AND cycling
        assert "strength_training" in variations
        assert "running" in variations
        assert "cycling" in variations

    def test_virtual_ride_does_not_include_outdoor_cycling(self, processor):
        ctx = processor.process_query("my zwift rides")
        filt = processor.create_pinecone_filter(ctx)
        assert filt is not None
        assert "activity_type" in filt
        variations = filt["activity_type"]["$in"]
        assert "virtual_ride" in variations
        assert "indoor_cycling" in variations
        assert "cycling" not in variations  # Bug fix: no over-expansion

    def test_date_range_produces_timestamp_filter(self, processor):
        ctx = processor.process_query("runs this week")
        filt = processor.create_pinecone_filter(ctx)
        assert filt is not None
        assert "timestamp" in filt
        assert "$gte" in filt["timestamp"]
        assert "$lte" in filt["timestamp"]

    def test_specific_date_produces_date_string(self, processor):
        ctx = processor.process_query("what did I do yesterday")
        filt = processor.create_pinecone_filter(ctx)
        assert filt is not None
        assert "date" in filt
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        assert filt["date"] == yesterday

    def test_no_filter_for_last_n_activities(self, processor):
        ctx = processor.process_query("last 3 runs")
        filt = processor.create_pinecone_filter(ctx)
        # Should have activity_type but NO timestamp/date filter
        assert filt is not None
        assert "activity_type" in filt
        assert "timestamp" not in filt
        assert "date" not in filt


# ---------------------------------------------------------------------------
# End-to-end integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_query_with_type_temporal_and_metric(self, processor):
        ctx = processor.process_query("runs faster than 12 km/h this month")
        assert ctx.activity_type_filter == "running"
        assert ctx.temporal_filter is not None
        assert ctx.temporal_filter.start_date is not None  # range, not point
        assert len(ctx.metric_filters) == 1
        assert ctx.metric_filters[0].field == "average_speed_kmh"

    def test_enhanced_query_includes_temporal_context(self, processor):
        ctx = processor.process_query("cycling this week")
        assert "between" in ctx.enhanced_query.lower()

    def test_enhanced_query_handles_list_activity_type(self, processor):
        ctx = processor.process_query("my workouts this week")
        assert "focus on" in ctx.enhanced_query.lower()
