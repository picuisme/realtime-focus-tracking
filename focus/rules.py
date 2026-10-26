"""Deterministic rules, measured in seconds.

Study mode: a per-frame 'looking away' check, calibrated to the person.
Driver mode: the safety alarms. These never go through a learned model.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StudyRules:
    yaw_max: float = 40.0        # degrees from the calibrated neutral pose
    pitch_down_max: float = 30.0
    pitch_up_max: float = 25.0
    closed_ratio: float = 0.7    # EAR below 70 % of open-eye EAR = closed
    note_taking_max_s: float = 30.0

    def reasons(self, s: dict, ref: dict, looking_down_s: float, phone: int) -> list[str]:
        """s: one frame (ear_score, head_pitch, head_yaw, person). Pose already relative or raw;
        ref holds the person's neutral pose and open-eye EAR."""
        if not s.get("person", 1):
            return ["absent"]
        if s["ear_score"] <= 0:
            return ["no_face"]
        out = []
        pitch = s["head_pitch"] - ref["pitch"]
        yaw = s["head_yaw"] - ref["yaw"]
        eyes_open = s["ear_score"] >= self.closed_ratio * ref["ear"]
        note_taking = pitch > 20 and abs(yaw) < 20 and phone == 0 and eyes_open and looking_down_s < self.note_taking_max_s
        if not eyes_open:
            out.append("eyes_closed")
        if abs(yaw) > self.yaw_max:
            out.append("head_turned")
        if pitch > self.pitch_down_max and not note_taking:
            out.append("head_down")
        if pitch < -self.pitch_up_max:
            out.append("looking_up")
        if phone:
            out.append("phone")
        return out


@dataclass
class DriverRules:
    microsleep_s: float = 1.5
    off_road_s: float = 2.0
    phone_s: float = 1.0
    perclos_warn: float = 0.15
    perclos_alarm: float = 0.30
    yawn_warn_count: int = 3
    state: dict = field(default_factory=lambda: {"closed": 0.0, "off": 0.0, "phone": 0.0})

    def step(self, dt: float, closed: bool, off_road: bool, phone: bool, perclos: float,
             perclos_ready: bool, yawns_5min: int) -> tuple[str, str]:
        st = self.state
        st["closed"] = st["closed"] + dt if closed else 0.0
        st["off"] = st["off"] + dt if (off_road and not closed) else 0.0
        st["phone"] = st["phone"] + dt if phone else 0.0
        level, why = "ok", "alert"
        if yawns_5min >= self.yawn_warn_count:
            level, why = "warn", "yawning"
        if perclos_ready and perclos > self.perclos_warn:
            level, why = "warn", "perclos"
        if perclos_ready and perclos > self.perclos_alarm:
            level, why = "alarm", "perclos"
        if st["off"] > self.off_road_s:
            level, why = "alarm", "off_road"
        if st["phone"] > self.phone_s:
            level, why = "alarm", "phone"
        if st["closed"] > self.microsleep_s:
            level, why = "alarm", "microsleep"
        return level, why
