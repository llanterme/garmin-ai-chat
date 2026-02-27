"""Tests for ActivityViewService aggregation logic."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.schemas.activity import RAW_SCHEMA_VERSION, DateWindow, ZoneDistribution7Days
from src.services.activity_views import ActivityViewService


_UNSET = object()


def _make_activity(
    id: str = "test-id",
    garmin_activity_id: str = "12345",
    activity_type: str = "running",
    start_time: datetime | None | object = _UNSET,
    duration: float = 3600.0,
    distance: float = 10000.0,
    calories: int = 500,
    average_heart_rate: int = 150,
    raw_data: dict | None = None,
    **kwargs,
) -> MagicMock:
    """Create a mock Activity ORM object."""
    a = MagicMock()
    a.id = id
    a.garmin_activity_id = garmin_activity_id
    a.activity_name = "Test Activity"
    a.activity_type = activity_type
    a.sport_type = None
    a.start_time = datetime(2026, 2, 20, 8, 0) if start_time is _UNSET else start_time
    a.duration = duration
    a.distance = distance
    a.calories = calories
    a.average_speed = 2.78
    a.max_speed = 4.0
    a.average_heart_rate = average_heart_rate
    a.max_heart_rate = 180
    a.elevation_gain = 100.0
    a.elevation_loss = 95.0
    a.average_power = None
    a.max_power = None
    a.normalized_power = None
    a.vo2_max = None
    a.training_stress_score = None
    a.temperature = None
    a.raw_data = raw_data or {}
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def _window(days_back: int = 14) -> DateWindow:
    return DateWindow(
        daysBack=days_back,
        startDate=date(2026, 2, 12),
        endDate=date(2026, 2, 26),
    )


# --------------------------------------------------------------------------
# Hard Session Classification
# --------------------------------------------------------------------------


class TestHardSessionClassification:
    svc = ActivityViewService()

    def test_aerobic_at_threshold(self):
        assert self.svc._is_hard_session({"aerobicTrainingEffect": 3.5}) is True

    def test_aerobic_below_threshold(self):
        assert self.svc._is_hard_session({"aerobicTrainingEffect": 3.4}) is False

    def test_anaerobic_at_threshold(self):
        assert self.svc._is_hard_session({"anaerobicTrainingEffect": 1.0}) is True

    def test_anaerobic_below_threshold(self):
        assert self.svc._is_hard_session({"anaerobicTrainingEffect": 0.9}) is False

    def test_load_at_threshold(self):
        assert self.svc._is_hard_session({"activityTrainingLoad": 150}) is True

    def test_load_below_threshold(self):
        assert self.svc._is_hard_session({"activityTrainingLoad": 149}) is False

    def test_all_below_thresholds(self):
        raw = {
            "aerobicTrainingEffect": 2.0,
            "anaerobicTrainingEffect": 0.5,
            "activityTrainingLoad": 50,
        }
        assert self.svc._is_hard_session(raw) is False

    def test_empty_raw(self):
        assert self.svc._is_hard_session({}) is False

    def test_any_one_sufficient(self):
        # Only anaerobic meets threshold
        raw = {
            "aerobicTrainingEffect": 1.0,
            "anaerobicTrainingEffect": 2.0,
            "activityTrainingLoad": 50,
        }
        assert self.svc._is_hard_session(raw) is True


# --------------------------------------------------------------------------
# ACR (Acute:Chronic Ratio)
# --------------------------------------------------------------------------


class TestACR:
    svc = ActivityViewService()
    window = _window()

    def test_normal_ratio(self):
        activities = [
            _make_activity(
                start_time=datetime(2026, 2, 25, 8, 0),
                raw_data={"activityTrainingLoad": 100},
            ),
            _make_activity(
                start_time=datetime(2026, 2, 22, 8, 0),
                raw_data={"activityTrainingLoad": 80},
            ),
            _make_activity(
                start_time=datetime(2026, 2, 10, 8, 0),
                raw_data={"activityTrainingLoad": 60},
            ),
            _make_activity(
                start_time=datetime(2026, 2, 5, 8, 0),
                raw_data={"activityTrainingLoad": 40},
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        state = resp.trainingState

        # 7d load: 100 + 80 = 180 (Feb 19-26)
        assert state.totalLoad7Days == 180.0
        # 28d load: 100 + 80 + 60 + 40 = 280
        assert state.totalLoad28Days == 280.0
        # ACR: 180 / (280/4) = 180 / 70 = 2.57
        assert state.acuteChronicRatio == 2.57

    def test_zero_chronic_returns_zero(self):
        resp = self.svc.build_agent_response([], self.window)
        assert resp.trainingState.acuteChronicRatio == 0.0

    def test_only_acute_load(self):
        activities = [
            _make_activity(
                start_time=datetime(2026, 2, 25, 8, 0),
                raw_data={"activityTrainingLoad": 100},
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        state = resp.trainingState
        # acute = 100, chronic = 100/4 = 25, ACR = 100/25 = 4.0
        assert state.totalLoad7Days == 100.0
        assert state.acuteChronicRatio == 4.0


# --------------------------------------------------------------------------
# Hard Sessions 7 Days
# --------------------------------------------------------------------------


class TestHardSessions7Days:
    svc = ActivityViewService()
    window = _window()

    def test_counts_only_7d_hard_sessions(self):
        activities = [
            # Hard, within 7d
            _make_activity(
                start_time=datetime(2026, 2, 25, 8, 0),
                raw_data={"aerobicTrainingEffect": 4.0},
            ),
            # Not hard, within 7d
            _make_activity(
                start_time=datetime(2026, 2, 24, 8, 0),
                raw_data={"aerobicTrainingEffect": 2.0},
            ),
            # Hard, outside 7d
            _make_activity(
                start_time=datetime(2026, 2, 10, 8, 0),
                raw_data={"aerobicTrainingEffect": 5.0},
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        assert resp.trainingState.hardSessions7Days == 1


# --------------------------------------------------------------------------
# Days Since Hard Session
# --------------------------------------------------------------------------


class TestDaysSinceHardSession:
    svc = ActivityViewService()
    window = _window()

    def test_recent_hard_session(self):
        activities = [
            _make_activity(
                start_time=datetime(2026, 2, 24, 8, 0),
                raw_data={"aerobicTrainingEffect": 4.0},
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        # Feb 26 - Feb 24 = 2
        assert resp.trainingState.daysSinceHardSession == 2

    def test_no_hard_sessions(self):
        activities = [
            _make_activity(
                start_time=datetime(2026, 2, 24, 8, 0),
                raw_data={"aerobicTrainingEffect": 2.0},
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        assert resp.trainingState.daysSinceHardSession is None

    def test_no_activities(self):
        resp = self.svc.build_agent_response([], self.window)
        assert resp.trainingState.daysSinceHardSession is None


# --------------------------------------------------------------------------
# Days Since Rest Day
# --------------------------------------------------------------------------


class TestDaysSinceRestDay:
    svc = ActivityViewService()
    window = _window()

    def test_rest_day_today(self):
        # No activity on Feb 26 (endDate)
        activities = [
            _make_activity(start_time=datetime(2026, 2, 25, 8, 0)),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        # Feb 26 has no activity → daysSinceRestDay = 0
        assert resp.trainingState.daysSinceRestDay == 0

    def test_consecutive_days(self):
        # Activities on Feb 26, 25, 24 — rest on Feb 23
        activities = [
            _make_activity(start_time=datetime(2026, 2, 26, 8, 0)),
            _make_activity(start_time=datetime(2026, 2, 25, 8, 0)),
            _make_activity(start_time=datetime(2026, 2, 24, 8, 0)),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        # Feb 23 is the most recent rest day → 3 days ago
        assert resp.trainingState.daysSinceRestDay == 3

    def test_no_activities_means_today_is_rest(self):
        resp = self.svc.build_agent_response([], self.window)
        assert resp.trainingState.daysSinceRestDay == 0


# --------------------------------------------------------------------------
# Avg Aerobic Effect 7 Days
# --------------------------------------------------------------------------


class TestAvgAerobicEffect7Days:
    svc = ActivityViewService()
    window = _window()

    def test_average_computed(self):
        activities = [
            _make_activity(
                start_time=datetime(2026, 2, 25, 8, 0),
                raw_data={"aerobicTrainingEffect": 3.0},
            ),
            _make_activity(
                start_time=datetime(2026, 2, 24, 8, 0),
                raw_data={"aerobicTrainingEffect": 4.0},
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        assert resp.trainingState.avgAerobicEffect7Days == 3.5

    def test_excludes_older_activities(self):
        activities = [
            _make_activity(
                start_time=datetime(2026, 2, 25, 8, 0),
                raw_data={"aerobicTrainingEffect": 3.0},
            ),
            _make_activity(
                start_time=datetime(2026, 2, 10, 8, 0),
                raw_data={"aerobicTrainingEffect": 5.0},
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        # Only the 7d activity counts
        assert resp.trainingState.avgAerobicEffect7Days == 3.0

    def test_none_when_no_data(self):
        activities = [
            _make_activity(
                start_time=datetime(2026, 2, 25, 8, 0),
                raw_data={},
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        assert resp.trainingState.avgAerobicEffect7Days is None


# --------------------------------------------------------------------------
# Zone Distribution
# --------------------------------------------------------------------------


class TestZoneDistribution:
    svc = ActivityViewService()

    def test_hr_zones_individual_keys(self):
        activities = [
            _make_activity(raw_data={
                "hrTimeInZone_1": 100,
                "hrTimeInZone_2": 200,
                "hrTimeInZone_3": 150,
                "hrTimeInZone_4": 80,
                "hrTimeInZone_5": 20,
            }),
        ]
        dist = self.svc._compute_zone_distribution(activities)
        total = 100 + 200 + 150 + 80 + 20  # 550
        assert dist.lowIntensityPercent == round(300 / total * 100)
        assert dist.moderatePercent == round(150 / total * 100)
        assert dist.highPercent == round(100 / total * 100)

    def test_hr_zones_list_format(self):
        activities = [
            _make_activity(raw_data={
                "hrTimeInZone": [100, 200, 150, 80, 20],
            }),
        ]
        dist = self.svc._compute_zone_distribution(activities)
        total = 550
        assert dist.lowIntensityPercent == round(300 / total * 100)
        assert dist.moderatePercent == round(150 / total * 100)
        assert dist.highPercent == round(100 / total * 100)

    def test_power_zones_fallback(self):
        activities = [
            _make_activity(raw_data={
                "powerTimeInZone": [50, 100, 80, 40, 20, 10, 5],
            }),
        ]
        dist = self.svc._compute_zone_distribution(activities)
        # Low = 50+100=150, Mod=80, High=40+20+10+5=75
        total = 150 + 80 + 75  # 305
        assert dist.lowIntensityPercent == round(150 / total * 100)
        assert dist.moderatePercent == round(80 / total * 100)
        assert dist.highPercent == round(75 / total * 100)

    def test_no_zone_data_returns_zeros(self):
        activities = [_make_activity(raw_data={})]
        dist = self.svc._compute_zone_distribution(activities)
        assert dist == ZoneDistribution7Days()

    def test_multiple_activities_aggregated(self):
        activities = [
            _make_activity(raw_data={"hrTimeInZone": [100, 100, 100, 50, 50]}),
            _make_activity(raw_data={"hrTimeInZone": [100, 100, 100, 50, 50]}),
        ]
        dist = self.svc._compute_zone_distribution(activities)
        # Total: Low=400, Mod=200, High=200 → total=800
        assert dist.lowIntensityPercent == 50
        assert dist.moderatePercent == 25
        assert dist.highPercent == 25


# --------------------------------------------------------------------------
# Raw View Response
# --------------------------------------------------------------------------


def _raw_window(days_back: int = 7) -> DateWindow:
    return DateWindow(
        daysBack=days_back,
        startDate=date(2026, 2, 20),
        endDate=date(2026, 2, 27),
        endDateExclusive=True,
    )


class TestRawView:
    svc = ActivityViewService()

    def test_returns_raw_data_verbatim(self):
        raw_payload = {"activityId": 123, "distance": 5000}
        activities = [_make_activity(id="abc", raw_data=raw_payload)]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.total == 1
        assert resp.items[0].id == "abc"
        assert resp.items[0].raw == raw_payload

    def test_null_raw_data_returns_empty_dict(self):
        activities = [_make_activity(raw_data=None)]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].raw == {}

    # --- schemaVersion ---

    def test_schema_version_present(self):
        resp = self.svc.build_raw_response([], _raw_window())
        assert resp.schemaVersion == RAW_SCHEMA_VERSION
        assert resp.schemaVersion == "activities.raw.v1"

    # --- paging fields ---

    def test_paging_fields_present_empty(self):
        resp = self.svc.build_raw_response([], _raw_window())
        assert resp.page == 1
        assert resp.pageSize == 50
        assert resp.pages == 1
        assert resp.hasNext is False
        assert resp.hasPrev is False

    def test_paging_fields_single_page(self):
        activities = [_make_activity(id=f"id-{i}") for i in range(5)]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.page == 1
        assert resp.pages == 1
        assert resp.hasNext is False
        assert resp.hasPrev is False
        assert resp.total == 5

    def test_paging_indicates_multiple_pages(self):
        activities = [_make_activity(id=f"id-{i}") for i in range(75)]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.total == 75
        assert resp.pageSize == 50
        assert resp.pages == 2
        assert resp.hasNext is True
        assert resp.hasPrev is False

    # --- endDateExclusive in window ---

    def test_window_end_date_exclusive_flag(self):
        resp = self.svc.build_raw_response([], _raw_window())
        assert resp.window.endDateExclusive is True

    def test_window_without_exclusive_flag(self):
        """Structured/agent windows default to None."""
        w = _window()
        assert w.endDateExclusive is None

    # --- item-level metadata ---

    def test_item_garmin_activity_id_from_raw(self):
        raw = {"activityId": 21956930402}
        activities = [_make_activity(
            garmin_activity_id="fallback-id", raw_data=raw
        )]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].garminActivityId == "21956930402"

    def test_item_garmin_activity_id_fallback_to_db(self):
        activities = [_make_activity(
            garmin_activity_id="db-12345", raw_data={}
        )]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].garminActivityId == "db-12345"

    def test_item_type_from_raw(self):
        raw = {"activityType": {"typeKey": "virtual_ride"}}
        activities = [_make_activity(
            activity_type="cycling", raw_data=raw
        )]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].type == "virtual_ride"

    def test_item_type_fallback_to_db(self):
        activities = [_make_activity(
            activity_type="running", raw_data={}
        )]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].type == "running"

    def test_item_start_time_from_raw_local(self):
        raw = {"startTimeLocal": "2026-02-23 11:25:45"}
        activities = [_make_activity(raw_data=raw)]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].startTime == "2026-02-23T11:25:45"

    def test_item_start_time_from_raw_gmt_fallback(self):
        raw = {"startTimeGMT": "2026-02-23 10:25:45"}
        activities = [_make_activity(raw_data=raw)]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].startTime == "2026-02-23T10:25:45"

    def test_item_start_time_prefers_local_over_gmt(self):
        raw = {
            "startTimeLocal": "2026-02-23 11:25:45",
            "startTimeGMT": "2026-02-23 10:25:45",
        }
        activities = [_make_activity(raw_data=raw)]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].startTime == "2026-02-23T11:25:45"

    def test_item_start_time_fallback_to_db(self):
        activities = [_make_activity(
            start_time=datetime(2026, 2, 23, 11, 25, 45),
            raw_data={},
        )]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].startTime == "2026-02-23T11:25:45"

    def test_item_start_time_none_when_missing(self):
        activities = [_make_activity(start_time=None, raw_data={})]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].startTime is None

    # --- raw object unchanged ---

    def test_raw_object_not_modified(self):
        """Metadata fields must NOT alter the raw object."""
        raw = {
            "activityId": 999,
            "activityType": {"typeKey": "trail_running"},
            "startTimeLocal": "2026-02-23 08:00:00",
            "distance": 15000,
            "nested": {"key": "value"},
        }
        import copy
        expected_raw = copy.deepcopy(raw)
        activities = [_make_activity(raw_data=raw)]
        resp = self.svc.build_raw_response(activities, _raw_window())
        assert resp.items[0].raw == expected_raw


# --------------------------------------------------------------------------
# Structured View Response
# --------------------------------------------------------------------------


class TestStructuredView:
    svc = ActivityViewService()

    def test_extracts_training_metrics_from_raw(self):
        raw = {
            "activityTrainingLoad": 128.5,
            "aerobicTrainingEffect": 3.4,
            "anaerobicTrainingEffect": 0.5,
            "trainingEffectLabel": "TEMPO",
            "normalizedPower": 210.0,
            "activityType": {"typeKey": "virtual_ride"},
        }
        activities = [_make_activity(raw_data=raw)]
        resp = self.svc.build_structured_response(activities, _window())
        a = resp.activities[0]

        assert a.trainingLoad == 128.5
        assert a.aerobicEffect == 3.4
        assert a.anaerobicEffect == 0.5
        assert a.trainingEffectLabel == "TEMPO"
        assert a.normPower == 210.0
        assert a.type == "virtual_ride"

    def test_falls_back_to_orm_columns(self):
        activities = [_make_activity(
            raw_data={},
            average_heart_rate=145,
            normalized_power=200.0,
        )]
        resp = self.svc.build_structured_response(activities, _window())
        a = resp.activities[0]
        assert a.avgHr == 145
        assert a.normPower == 200.0

    def test_hr_zones_extracted(self):
        raw = {"hrTimeInZone_1": 600, "hrTimeInZone_2": 1200,
               "hrTimeInZone_3": 900, "hrTimeInZone_4": 300, "hrTimeInZone_5": 0}
        activities = [_make_activity(raw_data=raw)]
        resp = self.svc.build_structured_response(activities, _window())
        assert resp.activities[0].hrZones is not None
        assert resp.activities[0].hrZones.z1 == 600
        assert resp.activities[0].hrZones.z3 == 900


# --------------------------------------------------------------------------
# Agent View — Full Integration
# --------------------------------------------------------------------------


class TestAgentViewIntegration:
    svc = ActivityViewService()
    window = _window()

    def test_full_agent_response(self):
        activities = [
            _make_activity(
                id="1",
                start_time=datetime(2026, 2, 25, 8, 0),
                raw_data={
                    "activityTrainingLoad": 150,
                    "aerobicTrainingEffect": 4.0,
                    "anaerobicTrainingEffect": 0.5,
                    "hrTimeInZone": [600, 1200, 900, 300, 100],
                },
            ),
            _make_activity(
                id="2",
                start_time=datetime(2026, 2, 23, 8, 0),
                raw_data={
                    "activityTrainingLoad": 80,
                    "aerobicTrainingEffect": 2.5,
                    "anaerobicTrainingEffect": 0.2,
                    "hrTimeInZone": [800, 1000, 600, 100, 0],
                },
            ),
            _make_activity(
                id="3",
                start_time=datetime(2026, 2, 10, 8, 0),
                raw_data={
                    "activityTrainingLoad": 120,
                    "aerobicTrainingEffect": 3.8,
                    "anaerobicTrainingEffect": 0.8,
                },
            ),
        ]
        resp = self.svc.build_agent_response(activities, self.window)
        state = resp.trainingState

        assert state.activityCount == 3
        assert state.totalLoad7Days == 230.0  # 150 + 80
        assert state.totalLoad28Days == 350.0  # 150 + 80 + 120
        assert state.hardSessions7Days == 1  # only id=1 is hard in 7d
        assert state.daysSinceHardSession == 1  # Feb 26 - Feb 25
        assert state.avgAerobicEffect7Days == 3.2  # (4.0 + 2.5) / 2 = 3.25 → rounded 3.2

        # Zone distribution from 7d activities (id 1 and 2)
        # Low: (600+1200) + (800+1000) = 3600
        # Mod: 900 + 600 = 1500
        # High: (300+100) + (100+0) = 500
        # Total: 5600
        assert state.zoneDistribution7Days.lowIntensityPercent == round(3600 / 5600 * 100)
        assert state.zoneDistribution7Days.moderatePercent == round(1500 / 5600 * 100)
        assert state.zoneDistribution7Days.highPercent == round(500 / 5600 * 100)
