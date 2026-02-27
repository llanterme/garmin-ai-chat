"""Stateless service for building activity view responses (raw, structured, agent).

All aggregation is deterministic and reads from the raw_data JSON column.
No LLM logic. No side effects.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import math

from ..db.models import Activity
from ..schemas.activity import (
    AgentViewResponse,
    DateWindow,
    HrZones,
    PowerZones,
    RAW_DEFAULT_PAGE_SIZE,
    RawActivityItem,
    RawViewResponse,
    StructuredActivity,
    StructuredViewResponse,
    TrainingState,
    ZoneDistribution7Days,
)


class ActivityViewService:
    """Builds view responses from Activity ORM objects."""

    # ------------------------------------------------------------------ raw

    def build_raw_response(
        self,
        activities: List[Activity],
        window: DateWindow,
    ) -> RawViewResponse:
        items = [self._build_raw_item(a) for a in activities]
        total = len(items)
        page_size = RAW_DEFAULT_PAGE_SIZE
        pages = max(1, math.ceil(total / page_size))
        return RawViewResponse(
            window=window,
            total=total,
            items=items,
            page=1,
            pageSize=page_size,
            pages=pages,
            hasNext=pages > 1,
            hasPrev=False,
        )

    def _build_raw_item(self, activity: Activity) -> RawActivityItem:
        """Build a raw item with metadata envelope around the untouched raw JSON."""
        raw: Dict[str, Any] = activity.raw_data or {}

        # garminActivityId: prefer raw.activityId, fallback DB column
        raw_activity_id = raw.get("activityId")
        garmin_id = (
            str(raw_activity_id) if raw_activity_id is not None
            else activity.garmin_activity_id
        )

        # type: prefer raw.activityType.typeKey, fallback DB column
        activity_type_raw = raw.get("activityType")
        if isinstance(activity_type_raw, dict):
            atype = activity_type_raw.get("typeKey", activity.activity_type)
        else:
            atype = activity.activity_type

        # startTime: prefer raw.startTimeLocal, then raw.startTimeGMT, then DB
        start_time = self._extract_raw_start_time(raw, activity)

        return RawActivityItem(
            id=activity.id,
            garminActivityId=garmin_id,
            type=atype,
            startTime=start_time,
            raw=raw,
        )

    @staticmethod
    def _extract_raw_start_time(
        raw: Dict[str, Any], activity: Activity
    ) -> Optional[str]:
        """Extract start time as ISO-8601 string.

        Priority: raw.startTimeLocal → raw.startTimeGMT → DB start_time.
        Garmin stores local times as "YYYY-MM-DD HH:mm:ss"; convert to ISO-8601.
        """
        for key in ("startTimeLocal", "startTimeGMT"):
            val = raw.get(key)
            if val and isinstance(val, str):
                # "2026-02-23 11:25:45" → "2026-02-23T11:25:45"
                return val.replace(" ", "T")

        if activity.start_time:
            return activity.start_time.isoformat()

        return None

    # ------------------------------------------------------------- structured

    def build_structured_response(
        self,
        activities: List[Activity],
        window: DateWindow,
    ) -> StructuredViewResponse:
        structured = [self._extract_structured(a) for a in activities]
        return StructuredViewResponse(
            window=window, total=len(structured), activities=structured
        )

    def _extract_structured(self, activity: Activity) -> StructuredActivity:
        """Extract enriched fields from raw_data with ORM column fallback."""
        raw: Dict[str, Any] = activity.raw_data or {}

        activity_type_raw = raw.get("activityType")
        if isinstance(activity_type_raw, dict):
            atype = activity_type_raw.get("typeKey", activity.activity_type)
        else:
            atype = activity.activity_type

        return StructuredActivity(
            id=activity.id,
            garminActivityId=activity.garmin_activity_id,
            type=atype,
            startTime=activity.start_time,
            durationSeconds=raw.get("duration") or activity.duration,
            distanceMeters=raw.get("distance") or activity.distance,
            calories=raw.get("calories") or activity.calories,
            avgHr=raw.get("averageHR") or activity.average_heart_rate,
            maxHr=raw.get("maxHR") or activity.max_heart_rate,
            avgPower=raw.get("avgPower") or activity.average_power,
            normPower=(
                raw.get("normalizedPower")
                or raw.get("normPower")
                or activity.normalized_power
            ),
            trainingLoad=raw.get("activityTrainingLoad"),
            aerobicEffect=raw.get("aerobicTrainingEffect"),
            anaerobicEffect=raw.get("anaerobicTrainingEffect"),
            trainingEffectLabel=raw.get("trainingEffectLabel"),
            hrZones=self._extract_hr_zones(raw),
            powerZones=self._extract_power_zones(raw),
        )

    @staticmethod
    def _extract_hr_zones(raw: Dict[str, Any]) -> Optional[HrZones]:
        """Extract HR time-in-zone from raw_data.

        Garmin may store zones as individual keys (hrTimeInZone_1 .. _5)
        or as a list (hrTimeInZone).
        """
        # Try individual keys first
        z_vals = [raw.get(f"hrTimeInZone_{i}") for i in range(1, 6)]
        if any(v is not None for v in z_vals):
            return HrZones(
                z1=z_vals[0] or 0,
                z2=z_vals[1] or 0,
                z3=z_vals[2] or 0,
                z4=z_vals[3] or 0,
                z5=z_vals[4] or 0,
            )

        # Try list format
        zones = raw.get("hrTimeInZone")
        if isinstance(zones, list) and len(zones) >= 5:
            return HrZones(
                z1=zones[0] or 0,
                z2=zones[1] or 0,
                z3=zones[2] or 0,
                z4=zones[3] or 0,
                z5=zones[4] or 0,
            )

        return None

    @staticmethod
    def _extract_power_zones(raw: Dict[str, Any]) -> Optional[PowerZones]:
        """Extract power time-in-zone from raw_data.

        Garmin may store zones as individual keys (powerTimeInZone_1 .. _7)
        or as a list (powerTimeInZone).
        """
        z_vals = [raw.get(f"powerTimeInZone_{i}") for i in range(1, 8)]
        if any(v is not None for v in z_vals):
            return PowerZones(
                z1=z_vals[0] or 0,
                z2=z_vals[1] or 0,
                z3=z_vals[2] or 0,
                z4=z_vals[3] or 0,
                z5=z_vals[4] or 0,
                z6=z_vals[5] or 0,
                z7=z_vals[6] or 0,
            )

        zones = raw.get("powerTimeInZone")
        if isinstance(zones, list) and len(zones) >= 7:
            return PowerZones(
                z1=zones[0] or 0,
                z2=zones[1] or 0,
                z3=zones[2] or 0,
                z4=zones[3] or 0,
                z5=zones[4] or 0,
                z6=zones[5] or 0,
                z7=zones[6] or 0,
            )

        return None

    # ---------------------------------------------------------------- agent

    def build_agent_response(
        self,
        all_28d_activities: List[Activity],
        window: DateWindow,
    ) -> AgentViewResponse:
        """Build agent view from 28 days of activities ending at window.endDate.

        The 28-day set is used for all computations (ACR, 7-day subsets, etc.).
        """
        end_dt = datetime.combine(window.endDate, datetime.max.time())
        seven_days_ago = end_dt - timedelta(days=7)

        # Partition into 7-day and 28-day sets
        activities_7d: List[Activity] = []
        for a in all_28d_activities:
            if a.start_time and a.start_time >= seven_days_ago:
                activities_7d.append(a)

        state = TrainingState()
        state.activityCount = len(all_28d_activities)

        # --- Load totals ---
        load_7d = 0.0
        load_28d = 0.0
        for a in all_28d_activities:
            raw: Dict[str, Any] = a.raw_data or {}
            load = raw.get("activityTrainingLoad") or 0.0
            load_28d += load
            if a.start_time and a.start_time >= seven_days_ago:
                load_7d += load

        state.totalLoad7Days = round(load_7d, 2)
        state.totalLoad28Days = round(load_28d, 2)

        # --- ACR ---
        chronic_avg = load_28d / 4.0
        state.acuteChronicRatio = (
            round(load_7d / chronic_avg, 2) if chronic_avg > 0 else 0.0
        )

        # --- Hard sessions (7 days) ---
        hard_count = 0
        last_hard_date: Optional[date] = None
        for a in all_28d_activities:
            raw = a.raw_data or {}
            if self._is_hard_session(raw):
                if a.start_time and a.start_time >= seven_days_ago:
                    hard_count += 1
                if a.start_time:
                    a_date = a.start_time.date() if isinstance(a.start_time, datetime) else a.start_time
                    if last_hard_date is None or a_date > last_hard_date:
                        last_hard_date = a_date

        state.hardSessions7Days = hard_count

        # --- Days since hard session ---
        if last_hard_date is not None:
            state.daysSinceHardSession = (window.endDate - last_hard_date).days

        # --- Days since rest day ---
        state.daysSinceRestDay = self._compute_days_since_rest(
            all_28d_activities, window.endDate
        )

        # --- Avg aerobic effect (7 days) ---
        aerobic_vals: List[float] = []
        for a in activities_7d:
            raw = a.raw_data or {}
            ae = raw.get("aerobicTrainingEffect")
            if ae is not None:
                aerobic_vals.append(float(ae))
        if aerobic_vals:
            state.avgAerobicEffect7Days = round(
                sum(aerobic_vals) / len(aerobic_vals), 1
            )

        # --- Zone distribution (7 days) ---
        state.zoneDistribution7Days = self._compute_zone_distribution(activities_7d)

        return AgentViewResponse(window=window, trainingState=state)

    # --------------------------------------------------- helper: hard session

    @staticmethod
    def _is_hard_session(raw: Dict[str, Any]) -> bool:
        """Classify a session as hard.

        Criteria (any one sufficient):
        - aerobicTrainingEffect >= 3.5
        - anaerobicTrainingEffect >= 1.0
        - activityTrainingLoad >= 150
        """
        aerobic = raw.get("aerobicTrainingEffect") or 0.0
        anaerobic = raw.get("anaerobicTrainingEffect") or 0.0
        load = raw.get("activityTrainingLoad") or 0.0
        return float(aerobic) >= 3.5 or float(anaerobic) >= 1.0 or float(load) >= 150

    # ------------------------------------------- helper: days since rest day

    @staticmethod
    def _compute_days_since_rest(
        activities: List[Activity], end: date
    ) -> Optional[int]:
        """Scan backward from end date to find the most recent rest day.

        A rest day is a calendar day with zero activities.
        """
        active_dates: set[date] = set()
        for a in activities:
            if a.start_time:
                d = (
                    a.start_time.date()
                    if isinstance(a.start_time, datetime)
                    else a.start_time
                )
                active_dates.add(d)

        # Scan backward up to 28 days
        for i in range(0, 29):
            check_date = end - timedelta(days=i)
            if check_date not in active_dates:
                return i

        return None

    # --------------------------------------------------------- helper: zones

    @staticmethod
    def _compute_zone_distribution(
        activities: List[Activity],
    ) -> ZoneDistribution7Days:
        """Aggregate zone time across activities.

        Prefers hrTimeInZone; falls back to powerTimeInZone.
        Low = Z1+Z2, Moderate = Z3, High = Z4+Z5(+Z6+Z7 for power).
        Returns integer percentages.
        """
        total_low = 0.0
        total_mod = 0.0
        total_high = 0.0
        has_data = False

        for a in activities:
            raw: Dict[str, Any] = a.raw_data or {}

            # Try HR zones first (individual keys then list)
            zones: Optional[List[float]] = None
            hr_vals = [raw.get(f"hrTimeInZone_{i}") for i in range(1, 6)]
            if any(v is not None for v in hr_vals):
                zones = [float(v or 0) for v in hr_vals]
            else:
                hr_list = raw.get("hrTimeInZone")
                if isinstance(hr_list, list) and len(hr_list) >= 5:
                    zones = [float(v or 0) for v in hr_list]

            # Fallback to power zones
            if zones is None:
                pwr_vals = [raw.get(f"powerTimeInZone_{i}") for i in range(1, 8)]
                if any(v is not None for v in pwr_vals):
                    zones = [float(v or 0) for v in pwr_vals]
                else:
                    pwr_list = raw.get("powerTimeInZone")
                    if isinstance(pwr_list, list) and len(pwr_list) >= 5:
                        zones = [float(v or 0) for v in pwr_list]

            if zones is None or len(zones) < 5:
                continue

            has_data = True
            total_low += zones[0] + zones[1]
            total_mod += zones[2]
            high = zones[3] + zones[4]
            for z in zones[5:]:
                high += z
            total_high += high

        total = total_low + total_mod + total_high
        if total == 0 or not has_data:
            return ZoneDistribution7Days()

        return ZoneDistribution7Days(
            lowIntensityPercent=round(total_low / total * 100),
            moderatePercent=round(total_mod / total * 100),
            highPercent=round(total_high / total * 100),
        )
