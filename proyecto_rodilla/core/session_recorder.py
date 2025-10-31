"""Session recording utilities for saving EMG/IMU sessions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

import numpy as np

try:  # Optional dependency for HDF5 storage
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

from config import settings as cfg
from utils import save_json


@dataclass
class EventMarker:
    """Represents a time marker captured during a session."""

    timestamp_us: int
    timestamp_relative_sec: float
    event_type: str
    description: str
    metadata: Dict[str, float | int | str | bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "timestamp_us": self.timestamp_us,
            "timestamp_relative_sec": self.timestamp_relative_sec,
            "type": self.event_type,
            "description": self.description,
        }
        if self.metadata:
            payload.update(self.metadata)
        return payload


class SessionRecorder:
    """Manages persistence for a single recording session."""

    def __init__(self, patient_id: str, session_number: int, session_id: str, base_dir: Optional[Path] = None) -> None:
        self.patient_id = patient_id
        self.session_number = session_number
        self.session_id = session_id
        data_root = base_dir or (cfg.PROJECT_ROOT / "data" / "patients")
        self.patient_dir = data_root / patient_id
        self.session_dir = self.patient_dir / "sessions" / session_id

        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._emg_records: List[Tuple[int, int, float, float, float, float, float, float]] = []
        self._imu_records: List[Tuple[int, int, float, float, float, float, float, float, float]] = []
        self._derived_records: List[Tuple[int, float, float]] = []
        self._events: List[EventMarker] = []

        self._start_monotonic: Optional[float] = None
        self._emg_start_us: Optional[int] = None
        self._imu_start_us: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, *, emg_start_us: Optional[int] = None, imu_start_us: Optional[int] = None) -> None:
        self._start_monotonic = time.monotonic()
        self._emg_start_us = emg_start_us
        self._imu_start_us = imu_start_us

    def reset(self) -> None:
        self._emg_records.clear()
        self._imu_records.clear()
        self._derived_records.clear()
        self._events.clear()
        self._start_monotonic = None
        self._emg_start_us = None
        self._imu_start_us = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def record_emg(self, timestamp_us: int, sequence: int, raw_ch0: float, raw_ch1: float,
                   filtered_ch0: float, filtered_ch1: float, rms_ch0: float, rms_ch1: float) -> None:
        self._emg_records.append((timestamp_us, sequence, raw_ch0, raw_ch1, filtered_ch0, filtered_ch1, rms_ch0, rms_ch1))

    def record_imu(self, timestamp_us: int, sequence: int, ax: float, ay: float, az: float,
                   gx: float, gy: float, gz: float, angle_deg: float) -> None:
        self._imu_records.append((timestamp_us, sequence, ax, ay, az, gx, gy, gz, angle_deg))

    def record_derived(self, timestamp_us: int, rom_instant: float, angular_velocity: float) -> None:
        self._derived_records.append((timestamp_us, rom_instant, angular_velocity))

    def add_event(self, event: EventMarker) -> None:
        self._events.append(event)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def _write_metadata(self, metadata: Dict[str, object]) -> None:
        save_json(metadata, str(self.session_dir / "metadata.json"))

    def _write_events(self) -> None:
        events_payload = {"events": [event.to_dict() for event in self._events]}
        save_json(events_payload, str(self.session_dir / "events.json"))

    def _write_calibration(self, calibration: Dict[str, object]) -> None:
        save_json(calibration, str(self.session_dir / "calibration.json"))

    def _write_notes(self, notes: str) -> None:
        notes_path = self.session_dir / "notes.txt"
        notes_path.write_text(notes, encoding="utf-8")

    def _write_raw_data_h5(self) -> Path:
        target = self.session_dir / "raw_data.h5"
        if not h5py:  # pragma: no cover - handled transparently
            return self._write_raw_data_npz()

        with h5py.File(target, "w") as h5:
            emg_grp = h5.create_group("emg")
            if self._emg_records:
                emg_array = np.array(self._emg_records, dtype=np.float64)
                timestamps = emg_array[:, 0].astype(np.uint64)
                sequence = emg_array[:, 1].astype(np.uint16)
                raw = emg_array[:, 2:4].astype(np.float32)
                filtered = emg_array[:, 4:6].astype(np.float32)
                rms = emg_array[:, 6:8].astype(np.float32)

                emg_grp.create_dataset("timestamps_us", data=timestamps)
                emg_grp.create_dataset("sequence", data=sequence)
                ch0_grp = emg_grp.create_group("ch0")
                ch1_grp = emg_grp.create_group("ch1")
                ch0_grp.create_dataset("raw", data=raw[:, 0])
                ch0_grp.create_dataset("filtered", data=filtered[:, 0])
                ch0_grp.create_dataset("rms", data=rms[:, 0])
                ch1_grp.create_dataset("raw", data=raw[:, 1])
                ch1_grp.create_dataset("filtered", data=filtered[:, 1])
                ch1_grp.create_dataset("rms", data=rms[:, 1])

            imu_grp = h5.create_group("imu")
            if self._imu_records:
                imu_array = np.array(self._imu_records, dtype=np.float64)
                imu_grp.create_dataset("timestamps_us", data=imu_array[:, 0].astype(np.uint64))
                imu_grp.create_dataset("sequence", data=imu_array[:, 1].astype(np.uint16))
                accel_grp = imu_grp.create_group("accel")
                gyro_grp = imu_grp.create_group("gyro")
                accel_grp.create_dataset("x", data=imu_array[:, 2].astype(np.float32))
                accel_grp.create_dataset("y", data=imu_array[:, 3].astype(np.float32))
                accel_grp.create_dataset("z", data=imu_array[:, 4].astype(np.float32))
                gyro_grp.create_dataset("x", data=imu_array[:, 5].astype(np.float32))
                gyro_grp.create_dataset("y", data=imu_array[:, 6].astype(np.float32))
                gyro_grp.create_dataset("z", data=imu_array[:, 7].astype(np.float32))
                imu_grp.create_dataset("angle", data=imu_array[:, 8].astype(np.float32))

            if self._derived_records:
                derived_grp = h5.create_group("derived")
                derived_array = np.array(self._derived_records, dtype=np.float64)
                derived_grp.create_dataset("timestamps_us", data=derived_array[:, 0].astype(np.uint64))
                derived_grp.create_dataset("rom_instant", data=derived_array[:, 1].astype(np.float32))
                derived_grp.create_dataset("velocity_angular", data=derived_array[:, 2].astype(np.float32))
        return target

    def _write_raw_data_npz(self) -> Path:
        target = self.session_dir / "raw_data.npz"
        payload: Dict[str, np.ndarray] = {}
        if self._emg_records:
            emg_array = np.array(self._emg_records, dtype=np.float64)
            payload["emg_timestamps_us"] = emg_array[:, 0].astype(np.uint64)
            payload["emg_sequence"] = emg_array[:, 1].astype(np.uint16)
            payload["emg_raw_ch0"] = emg_array[:, 2].astype(np.float32)
            payload["emg_raw_ch1"] = emg_array[:, 3].astype(np.float32)
            payload["emg_filtered_ch0"] = emg_array[:, 4].astype(np.float32)
            payload["emg_filtered_ch1"] = emg_array[:, 5].astype(np.float32)
            payload["emg_rms_ch0"] = emg_array[:, 6].astype(np.float32)
            payload["emg_rms_ch1"] = emg_array[:, 7].astype(np.float32)
        if self._imu_records:
            imu_array = np.array(self._imu_records, dtype=np.float64)
            payload["imu_timestamps_us"] = imu_array[:, 0].astype(np.uint64)
            payload["imu_sequence"] = imu_array[:, 1].astype(np.uint16)
            payload["imu_accel_x"] = imu_array[:, 2].astype(np.float32)
            payload["imu_accel_y"] = imu_array[:, 3].astype(np.float32)
            payload["imu_accel_z"] = imu_array[:, 4].astype(np.float32)
            payload["imu_gyro_x"] = imu_array[:, 5].astype(np.float32)
            payload["imu_gyro_y"] = imu_array[:, 6].astype(np.float32)
            payload["imu_gyro_z"] = imu_array[:, 7].astype(np.float32)
            payload["imu_angle"] = imu_array[:, 8].astype(np.float32)
        if self._derived_records:
            derived_array = np.array(self._derived_records, dtype=np.float64)
            payload["derived_timestamps_us"] = derived_array[:, 0].astype(np.uint64)
            payload["derived_rom_instant"] = derived_array[:, 1].astype(np.float32)
            payload["derived_velocity_angular"] = derived_array[:, 2].astype(np.float32)
        if payload:
            np.savez(target, **payload)
        else:  # pragma: no cover - empty session
            target.touch()
        return target

    # ------------------------------------------------------------------
    # Public finalization
    # ------------------------------------------------------------------
    def finalize(self, metadata: Dict[str, object], calibration: Dict[str, object], notes: str) -> Dict[str, Path]:
        self._write_metadata(metadata)
        self._write_events()
        self._write_calibration(calibration)
        self._write_notes(notes or "")
        data_path = self._write_raw_data_h5()
        return {
            "metadata": self.session_dir / "metadata.json",
            "events": self.session_dir / "events.json",
            "calibration": self.session_dir / "calibration.json",
            "notes": self.session_dir / "notes.txt",
            "data": data_path,
        }

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def emg_sample_count(self) -> int:
        return len(self._emg_records)

    @property
    def imu_sample_count(self) -> int:
        return len(self._imu_records)

    @property
    def has_started(self) -> bool:
        return self._start_monotonic is not None

    def elapsed_seconds(self) -> float:
        if self._start_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._start_monotonic)
