"""Utilities for loading recorded session data for offline analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # Optional dependency, same as session_recorder
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

from config import settings as cfg
from utils import load_json


@dataclass
class SessionInfo:
    """Lightweight descriptor for a recorded session."""

    session_id: str
    path: Path
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-readable label composed from metadata."""
        if not self.metadata:
            return self.session_id
        date = self.metadata.get("date", "?")
        time_start = self.metadata.get("time_start", "")
        session_type = self.metadata.get("session_type", "")
        return f"{self.session_id} - {date} {time_start} [{session_type}]".strip()


@dataclass
class PatientInfo:
    """Descriptor for a patient and its stored sessions."""

    patient_id: str
    path: Path
    profile: Dict[str, object] = field(default_factory=dict)
    sessions: List[SessionInfo] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        name = str(self.profile.get("name", "")).strip()
        return name or self.patient_id


class SessionDataset:
    """Aggregate loaded session data with helper computations."""

    def __init__(self, patient_id: str, session_info: SessionInfo) -> None:
        self.patient_id = patient_id
        self.session_info = session_info
        self.session_dir = session_info.path

        self.metadata = session_info.metadata or {}
        self.calibration = self._safe_load_json("calibration.json")
        events_payload = self._safe_load_json("events.json")
        self.events = events_payload.get("events", []) if isinstance(events_payload, dict) else []
        notes_path = self.session_dir / "notes.txt"
        self.notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""

        self._data_loaded = False
        self._emg_data: Dict[str, np.ndarray] = {}
        self._imu_data: Dict[str, np.ndarray] = {}
        self._derived_data: Dict[str, np.ndarray] = {}
        self._time_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _safe_load_json(self, name: str) -> Dict[str, object]:
        file_path = self.session_dir / name
        if not file_path.exists():
            return {}
        payload = load_json(str(file_path))
        return payload if isinstance(payload, dict) else {}

    def _load_raw_data(self) -> None:
        if self._data_loaded:
            return

        h5_path = self.session_dir / "raw_data.h5"
        npz_path = self.session_dir / "raw_data.npz"
        if h5_path.exists() and h5py is not None:
            self._load_from_h5(h5_path)
        elif npz_path.exists():
            self._load_from_npz(npz_path)
        else:
            self._emg_data = {}
            self._imu_data = {}
            self._derived_data = {}
        self._data_loaded = True

    def _load_from_npz(self, path: Path) -> None:
        payload = np.load(path, allow_pickle=False)
        self._emg_data = {
            "timestamps_us": payload.get("emg_timestamps_us"),
            "sequence": payload.get("emg_sequence"),
            "raw_ch0": payload.get("emg_raw_ch0"),
            "raw_ch1": payload.get("emg_raw_ch1"),
            "filtered_ch0": payload.get("emg_filtered_ch0"),
            "filtered_ch1": payload.get("emg_filtered_ch1"),
            "rms_ch0": payload.get("emg_rms_ch0"),
            "rms_ch1": payload.get("emg_rms_ch1"),
        }
        self._imu_data = {
            "timestamps_us": payload.get("imu_timestamps_us"),
            "sequence": payload.get("imu_sequence"),
            "accel_x": payload.get("imu_accel_x"),
            "accel_y": payload.get("imu_accel_y"),
            "accel_z": payload.get("imu_accel_z"),
            "gyro_x": payload.get("imu_gyro_x"),
            "gyro_y": payload.get("imu_gyro_y"),
            "gyro_z": payload.get("imu_gyro_z"),
            "angle": payload.get("imu_angle"),
        }
        self._derived_data = {
            "timestamps_us": payload.get("derived_timestamps_us"),
            "rom": payload.get("derived_rom_instant"),
            "velocity": payload.get("derived_velocity_angular"),
        }

    def _load_from_h5(self, path: Path) -> None:
        self._emg_data = {}
        self._imu_data = {}
        self._derived_data = {}
        with h5py.File(path, "r") as handle:  # type: ignore[operator]
            if "emg" in handle:
                emg_grp = handle["emg"]
                self._emg_data = {
                    "timestamps_us": np.array(emg_grp.get("timestamps_us")),
                    "sequence": np.array(emg_grp.get("sequence")),
                    "raw_ch0": np.array(emg_grp.get("ch0/raw")),
                    "filtered_ch0": np.array(emg_grp.get("ch0/filtered")),
                    "rms_ch0": np.array(emg_grp.get("ch0/rms")),
                    "raw_ch1": np.array(emg_grp.get("ch1/raw")),
                    "filtered_ch1": np.array(emg_grp.get("ch1/filtered")),
                    "rms_ch1": np.array(emg_grp.get("ch1/rms")),
                }
            if "imu" in handle:
                imu_grp = handle["imu"]
                self._imu_data = {
                    "timestamps_us": np.array(imu_grp.get("timestamps_us")),
                    "sequence": np.array(imu_grp.get("sequence")),
                    "accel_x": np.array(imu_grp.get("accel/x")),
                    "accel_y": np.array(imu_grp.get("accel/y")),
                    "accel_z": np.array(imu_grp.get("accel/z")),
                    "gyro_x": np.array(imu_grp.get("gyro/x")),
                    "gyro_y": np.array(imu_grp.get("gyro/y")),
                    "gyro_z": np.array(imu_grp.get("gyro/z")),
                    "angle": np.array(imu_grp.get("angle")),
                }
            if "derived" in handle:
                drv_grp = handle["derived"]
                self._derived_data = {
                    "timestamps_us": np.array(drv_grp.get("timestamps_us")),
                    "rom": np.array(drv_grp.get("rom_instant")),
                    "velocity": np.array(drv_grp.get("velocity_angular")),
                }

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def has_emg(self) -> bool:
        self._load_raw_data()
        return bool(self._emg_data.get("timestamps_us") is not None)

    def has_imu(self) -> bool:
        self._load_raw_data()
        return bool(self._imu_data.get("timestamps_us") is not None)

    def time_axis(self, source: str) -> np.ndarray:
        self._load_raw_data()
        if source in self._time_cache:
            return self._time_cache[source]

        arr = None
        if source == "emg":
            arr = self._emg_data.get("timestamps_us")
        elif source == "imu":
            arr = self._imu_data.get("timestamps_us")
        elif source == "derived":
            arr = self._derived_data.get("timestamps_us")
        if arr is None:
            self._time_cache[source] = np.array([])
            return self._time_cache[source]
        arr = np.asarray(arr)
        if arr.size == 0:
            self._time_cache[source] = np.array([])
            return self._time_cache[source]
        arr = arr.astype(np.float64)
        arr = (arr - arr[0]) / 1e6
        self._time_cache[source] = arr
        return arr

    def emg_channel(self, channel: int, kind: str = "rms") -> np.ndarray:
        self._load_raw_data()
        suffix = {"raw": "raw", "filtered": "filtered", "rms": "rms"}.get(kind, "rms")
        key = f"{suffix}_ch{channel}"
        data = self._emg_data.get(key)
        if data is None:
            return np.array([])
        return np.asarray(data)

    def imu_series(self, axis: str) -> np.ndarray:
        self._load_raw_data()
        data = self._imu_data.get(axis)
        if data is None:
            return np.array([])
        return np.asarray(data)

    def angle_series(self) -> np.ndarray:
        self._load_raw_data()
        data = self._imu_data.get("angle")
        if data is None:
            return np.array([])
        return np.asarray(data)

    def derived_series(self, key: str) -> np.ndarray:
        self._load_raw_data()
        data = self._derived_data.get(key)
        if data is None:
            return np.array([])
        return np.asarray(data)

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------
    def compute_basic_metrics(self) -> Dict[str, float]:
        angle = self.angle_series()
        t_angle = self.time_axis("imu")
        t_emg = self.time_axis("emg")
        rms_ch0 = self.emg_channel(0, "rms")
        rms_ch1 = self.emg_channel(1, "rms")

        metrics: Dict[str, float] = {}
        if angle.size:
            metrics["angle_min"] = float(np.min(angle))
            metrics["angle_max"] = float(np.max(angle))
            metrics["rom_total"] = float(np.max(angle) - np.min(angle))
            metrics["angle_mean"] = float(np.mean(angle))
            metrics["angle_std"] = float(np.std(angle))
            if t_angle.size > 1:
                velocity = np.gradient(angle, t_angle)
                metrics["velocity_peak"] = float(np.max(np.abs(velocity)))
                metrics["time_span"] = float(t_angle[-1] - t_angle[0])
                zero_crossings = np.where(np.diff(np.sign(velocity)) != 0)[0]
                if zero_crossings.size:
                    metrics["angle_cycles"] = float(max(1, zero_crossings.size // 2))
        if "time_span" in metrics:
            metrics.setdefault("duration_sec", metrics["time_span"])
        if rms_ch0.size:
            metrics["rms_ch0_peak"] = float(np.max(np.abs(rms_ch0)))
            metrics["rms_ch0_mean"] = float(np.mean(np.abs(rms_ch0)))
        if rms_ch1.size:
            metrics["rms_ch1_peak"] = float(np.max(np.abs(rms_ch1)))
            metrics["rms_ch1_mean"] = float(np.mean(np.abs(rms_ch1)))
        if rms_ch0.size and rms_ch1.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(rms_ch1 > 1e-6, rms_ch0 / rms_ch1, np.nan)
            ratio = ratio[np.isfinite(ratio)]
            if ratio.size:
                metrics["ratio_qh"] = float(np.mean(ratio))
                coactivation = np.minimum(rms_ch0, rms_ch1) / np.maximum(rms_ch0, rms_ch1)
                coactivation = coactivation[np.isfinite(coactivation)]
                if coactivation.size:
                    metrics["co_contraction"] = float(np.mean(coactivation))
            peak0 = metrics.get("rms_ch0_peak")
            peak1 = metrics.get("rms_ch1_peak")
            if peak0 and peak1:
                balance = (peak0 - peak1) / max(peak0, peak1)
                metrics["emg_balance_pct"] = float(balance * 100.0)
        if t_emg.size:
            metrics.setdefault("emg_duration_sec", float(t_emg[-1] - t_emg[0]) if t_emg.size > 1 else 0.0)
        metrics["valid_emg_packets"] = self._sequence_quality(self._emg_data.get("sequence"))
        metrics["valid_imu_packets"] = self._sequence_quality(self._imu_data.get("sequence"))
        metrics["event_count"] = float(len(self.events))
        return metrics

    def _sequence_quality(self, sequence: Optional[np.ndarray]) -> float:
        if sequence is None or sequence.size < 2:
            return 1.0 if sequence is not None and sequence.size > 0 else 0.0
        seq = sequence.astype(np.int64)
        diffs = np.diff(seq)
        valid = np.count_nonzero(diffs == 1)
        return float(valid / diffs.size)

    def compute_fatigue_metrics(self, channel: int = 0) -> Dict[str, float]:
        rms = self.emg_channel(channel, "rms")
        t_emg = self.time_axis("emg")
        if rms.size == 0 or t_emg.size == 0:
            return {}
        limit = min(len(rms), len(t_emg))
        if limit < 32:
            return {"duration": float(t_emg[limit - 1] - t_emg[0]) if limit > 1 else 0.0}
        rms = np.abs(rms[:limit])
        times = t_emg[:limit]
        duration = float(times[-1] - times[0]) if limit > 1 else 0.0
        window = max(32, int(limit * 0.05))
        baseline = float(np.mean(rms[:window])) if window <= limit else float(np.mean(rms))
        tail = float(np.mean(rms[-window:])) if window <= limit else baseline
        decline = ((baseline - tail) / baseline) * 100.0 if baseline > 1e-6 else 0.0
        slope = 0.0
        if duration > 0.0 and limit > 1:
            coeffs = np.polyfit(times, rms, 1)
            slope = float(coeffs[0])
        median_frequency = None
        if limit >= 256 and duration > 0.0:
            window_fn = np.hanning(limit)
            demeaned = (rms - np.mean(rms)) * window_fn
            spectrum = np.abs(np.fft.rfft(demeaned)) ** 2
            dt = times[1] - times[0] if limit > 1 else 1.0
            freqs = np.fft.rfftfreq(limit, d=dt)
            cumulative = np.cumsum(spectrum)
            if cumulative[-1] > 0:
                median_idx = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
                median_frequency = float(freqs[median_idx])
        metrics = {
            "duration": duration,
            "rms_decline_pct": float(decline),
            "rms_slope": slope,
            "baseline_rms": baseline,
            "tail_rms": tail,
        }
        if median_frequency is not None:
            metrics["median_frequency"] = median_frequency
        return metrics


# ----------------------------------------------------------------------
# Repository helpers
# ----------------------------------------------------------------------
def discover_patients() -> List[PatientInfo]:
    """Traverse project data directory and gather patient/session info."""
    patients_dir = cfg.PROJECT_ROOT / "data" / "patients"
    if not patients_dir.exists():
        return []

    patients: List[PatientInfo] = []
    for child in sorted(patients_dir.iterdir()):
        if not child.is_dir():
            continue
        profile = load_json(str(child / "profile.json"))
        info = PatientInfo(patient_id=child.name, path=child, profile=profile or {})
        sessions_dir = child / "sessions"
        if sessions_dir.exists():
            for session_child in sorted(sessions_dir.iterdir()):
                if not session_child.is_dir():
                    continue
                metadata = load_json(str(session_child / "metadata.json"))
                session_info = SessionInfo(
                    session_id=session_child.name,
                    path=session_child,
                    metadata=metadata or {},
                )
                info.sessions.append(session_info)
        patients.append(info)
    return patients


def load_session(patient_id: str, session_info: SessionInfo) -> SessionDataset:
    """Factory helper used by the UI components."""
    return SessionDataset(patient_id, session_info)
