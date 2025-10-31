"""Ventana de grabación de sesión EMG/IMU."""
from __future__ import annotations

import shutil
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import settings as cfg
from core import AngleCalculator, EMGProcessor, SerialReaderThread, get_available_ports
from core.session_recorder import EventMarker, SessionRecorder
from utils import load_json, save_json
from .calibration_dialog import CalibrationDialog
from .emg_normalization_dialog import EMGNormalizationDialog


class RecordingState(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    RECORDING = auto()
    PAUSED = auto()


class SessionRecordingWindow(QMainWindow):
    """Módulo de grabación de sesión con captura en vivo y persistencia."""

    window_reload_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Grabación de Sesión EMG/IMU")
        self.setWindowState(QtCore.Qt.WindowState.WindowMaximized)
        self.setMinimumSize(1200, 820)

        self.serial_thread: Optional[SerialReaderThread] = None
        self.is_connected = False
        self.recording_state = RecordingState.IDLE
        self.current_session_id: Optional[str] = None

        self.emg_ch0_processor = EMGProcessor()
        self.emg_ch1_processor = EMGProcessor()
        self.angle_calculator = AngleCalculator()

        self.mvc_values: Dict[int, Optional[float]] = {0: None, 1: None}
        self.session_recorder: Optional[SessionRecorder] = None
        self.pending_countdown = 3

        buffer_len_emg = int(cfg.EMG_FS * cfg.WINDOW_TIME_SEC)
        buffer_len_imu = int(cfg.IMU_FS * cfg.WINDOW_TIME_SEC)
        self.time_emg_ch0 = np.zeros(buffer_len_emg, dtype=np.float64)
        self.data_rms_ch0 = np.zeros(buffer_len_emg, dtype=np.float64)
        self.data_emg_ch0 = np.zeros(buffer_len_emg, dtype=np.float64)
        self.time_emg_ch1 = np.zeros(buffer_len_emg, dtype=np.float64)
        self.data_rms_ch1 = np.zeros(buffer_len_emg, dtype=np.float64)
        self.data_emg_ch1 = np.zeros(buffer_len_emg, dtype=np.float64)
        self.time_imu = np.zeros(buffer_len_imu, dtype=np.float64)
        self.data_angle = np.zeros(buffer_len_imu, dtype=np.float64)

        self.idx_emg_ch0 = 0
        self.idx_emg_ch1 = 0
        self.idx_imu = 0
        self.current_time_emg = 0.0
        self.current_time_imu = 0.0
        self.t0_emg: Optional[int] = None
        self.t0_imu: Optional[int] = None

        self.event_counter = 0
        self.last_stats_update = QtCore.QTime.currentTime()
        self.emg_count = 0
        self.imu_count = 0
        self.last_angle = 0.0
        self.current_raw_angle = 0.0
        self.current_rms_values = {0: 0.0, 1: 0.0}
        self._record_wallclock_start = None  # type: Optional[datetime]
        self._loading_patient_ids = False

        self._build_ui()
        self._load_patient_profiles()

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(int(1000 / max(1, cfg.UPDATE_FPS)))
        self.preview_timer.timeout.connect(self._update_plots)
        self.preview_timer.start()

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(500)
        self.status_timer.timeout.connect(self._update_status_bar)
        self.status_timer.start()

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._advance_countdown)

    # ------------------------------------------------------------------
    # UI creation
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)
        self.setCentralWidget(central)

        splitter = QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(left_panel)

        patient_box = QGroupBox("Datos del paciente")
        patient_form = QFormLayout(patient_box)
        patient_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        patient_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.input_patient_id = QComboBox()
        self.input_patient_id.setEditable(True)
        self.input_patient_id.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.input_patient_id.lineEdit().editingFinished.connect(self._on_patient_id_committed)
        self.input_patient_id.currentIndexChanged.connect(self._on_patient_id_committed)
        patient_form.addRow("ID paciente", self.input_patient_id)

        self.input_patient_name = QLineEdit()
        patient_form.addRow("Nombre", self.input_patient_name)

        self.input_age = QSpinBox()
        self.input_age.setRange(0, 120)
        patient_form.addRow("Edad", self.input_age)

        self.input_sex = QComboBox()
        self.input_sex.addItems(["", "Femenino", "Masculino", "Otro"])
        patient_form.addRow("Sexo", self.input_sex)

        self.input_weight = QDoubleSpinBox()
        self.input_weight.setRange(0.0, 300.0)
        self.input_weight.setSuffix(" kg")
        self.input_weight.setDecimals(1)
        patient_form.addRow("Peso", self.input_weight)

        self.input_height = QDoubleSpinBox()
        self.input_height.setRange(0.0, 220.0)
        self.input_height.setSuffix(" cm")
        self.input_height.setDecimals(1)
        patient_form.addRow("Altura", self.input_height)

        self.input_pathology = QComboBox()
        self.input_pathology.setEditable(True)
        self.input_pathology.addItems([
            "",
            "Rehabilitación LCA",
            "Osteoartritis",
            "Lesión menisco",
            "Dolor patelofemoral",
        ])
        patient_form.addRow("Diagnóstico", self.input_pathology)

        self.input_side = QComboBox()
        self.input_side.addItems(["", "Izquierdo", "Derecho", "Bilateral"])
        patient_form.addRow("Lado afectado", self.input_side)

        self.input_session_number = QSpinBox()
        self.input_session_number.setRange(1, 999)
        patient_form.addRow("Sesión", self.input_session_number)

        left_layout.addWidget(patient_box)

        notes_box = QGroupBox("Notas clínicas")
        notes_layout = QVBoxLayout(notes_box)
        self.notes_field = QTextEdit()
        self.notes_field.setPlaceholderText("Detalles relevantes de la sesión...")
        notes_layout.addWidget(self.notes_field)
        left_layout.addWidget(notes_box)

        config_box = QGroupBox("Configuración de grabación")
        config_form = QFormLayout(config_box)

        self.session_type_combo = QComboBox()
        self.session_type_combo.addItems([
            "Evaluación inicial",
            "Seguimiento",
            "Evaluación final",
        ])
        config_form.addRow("Tipo de sesión", self.session_type_combo)

        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems([
            "Libre (grabación continua)",
        ])
        config_form.addRow("Protocolo", self.protocol_combo)

        self.duration_label = QLabel("Duración estimada: -- min")
        config_form.addRow("Duración", self.duration_label)

        fs_label = QLabel(f"EMG: {cfg.EMG_FS:.0f} Hz | IMU: {cfg.IMU_FS:.0f} Hz")
        config_form.addRow("Frecuencias", fs_label)

        left_layout.addWidget(config_box)

        events_box = QGroupBox("Marcadores de evento")
        events_layout = QVBoxLayout(events_box)
        events_layout.setSpacing(8)

        combo_row = QHBoxLayout()
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItems([
            "Inicio repetición",
            "Máximo flexión",
            "Máximo extensión",
            "Dolor reportado",
            "Pérdida balance",
            "Cambio ejercicio",
            "Personalizado",
        ])
        combo_row.addWidget(self.event_type_combo)

        self.custom_event_input = QLineEdit()
        self.custom_event_input.setPlaceholderText("Descripción personalizada")
        combo_row.addWidget(self.custom_event_input)

        events_layout.addLayout(combo_row)

        button_row = QHBoxLayout()
        self.btn_mark_event = QPushButton("Marcar evento")
        self.btn_mark_event.clicked.connect(self._on_mark_event)
        button_row.addWidget(self.btn_mark_event)

        self.btn_edit_event = QPushButton("Editar")
        self.btn_edit_event.clicked.connect(self._on_edit_event)
        button_row.addWidget(self.btn_edit_event)

        self.btn_remove_event = QPushButton("Eliminar")
        self.btn_remove_event.clicked.connect(self._on_remove_event)
        button_row.addWidget(self.btn_remove_event)
        events_layout.addLayout(button_row)

        self.events_list = QListWidget()
        self.events_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        events_layout.addWidget(self.events_list)

        left_layout.addWidget(events_box, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(right_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        connection_box = QGroupBox("Conexión de hardware")
        connection_layout = QHBoxLayout(connection_box)
        connection_layout.setSpacing(8)

        self.port_combo = QComboBox()
        connection_layout.addWidget(self.port_combo, stretch=1)

        self.btn_refresh_ports = QPushButton("Actualizar puertos")
        self.btn_refresh_ports.clicked.connect(self._refresh_ports)
        connection_layout.addWidget(self.btn_refresh_ports)

        self.btn_toggle_connection = QPushButton("Conectar")
        self.btn_toggle_connection.clicked.connect(self._toggle_connection)
        connection_layout.addWidget(self.btn_toggle_connection)

        self.btn_calibrate = QPushButton("Calibrar IMU")
        self.btn_calibrate.clicked.connect(self._open_calibration)
        connection_layout.addWidget(self.btn_calibrate)

        self.btn_normalize = QPushButton("Normalizar EMG")
        self.btn_normalize.clicked.connect(self._open_emg_normalization)
        connection_layout.addWidget(self.btn_normalize)

        right_layout.addWidget(connection_box)

        preview_box = QGroupBox("Previsualización en vivo")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setSpacing(6)

        self.plot_ch0 = pg.PlotWidget(background="#1F1F21")
        self.plot_ch0.setMinimumHeight(140)
        self.curve_ch0 = self.plot_ch0.plot(pen=pg.mkPen(color=cfg.COLOR_CH0, width=1.2))
        self.curve_rms_ch0 = self.plot_ch0.plot(pen=pg.mkPen(color=cfg.COLOR_RMS_CH0, width=1.0))
        self.plot_ch0.addLegend(offset=(10, 10))
        self.plot_ch0.setYRange(-1, 1, padding=0.05)
        preview_layout.addWidget(self.plot_ch0)

        self.plot_ch1 = pg.PlotWidget(background="#1F1F21")
        self.plot_ch1.setMinimumHeight(140)
        self.curve_ch1 = self.plot_ch1.plot(pen=pg.mkPen(color=cfg.COLOR_CH1, width=1.2))
        self.curve_rms_ch1 = self.plot_ch1.plot(pen=pg.mkPen(color=cfg.COLOR_RMS_CH1, width=1.0))
        self.plot_ch1.setYRange(-1, 1, padding=0.05)
        preview_layout.addWidget(self.plot_ch1)

        self.plot_angle = pg.PlotWidget(background="#1F1F21")
        self.plot_angle.setMinimumHeight(140)
        self.curve_angle = self.plot_angle.plot(pen=pg.mkPen(color=cfg.COLOR_ANGLE, width=1.4))
        self.plot_angle.setYRange(-10, 180, padding=0.05)
        preview_layout.addWidget(self.plot_angle)

        indicators_row = QHBoxLayout()
        self.label_rms_ch0 = QLabel("RMS CH0: -- mV")
        self.label_rms_ch1 = QLabel("RMS CH1: -- mV")
        self.label_angle = QLabel("Ángulo: --°")
        for lbl in (self.label_rms_ch0, self.label_rms_ch1, self.label_angle):
            lbl.setStyleSheet("color: #D9E4E4")
        indicators_row.addWidget(self.label_rms_ch0)
        indicators_row.addWidget(self.label_rms_ch1)
        indicators_row.addWidget(self.label_angle)
        indicators_row.addStretch(1)
        preview_layout.addLayout(indicators_row)

        right_layout.addWidget(preview_box, stretch=1)

        controls_box = QGroupBox("Control de grabación")
        controls_layout = QHBoxLayout(controls_box)
        controls_layout.setSpacing(12)

        self.btn_record = QPushButton("● Grabar")
        self.btn_record.clicked.connect(self._on_record_clicked)
        controls_layout.addWidget(self.btn_record)

        self.btn_pause = QPushButton("⏸ Pausar")
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_pause.setEnabled(False)
        controls_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹ Detener")
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("🔄 Reiniciar")
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        controls_layout.addWidget(self.btn_reset)

        right_layout.addWidget(controls_box)

        status_box = QGroupBox("Estado de la sesión")
        status_layout = QHBoxLayout(status_box)
        status_layout.setSpacing(20)

        self.label_status = QLabel("Estado: Inactivo")
        self.label_timer = QLabel("Tiempo: 00:00")
        self.label_file_size = QLabel("Archivo: 0.0 MB")
        self.label_data_rate = QLabel("Tasa: 0 KB/s")
        self.label_frames = QLabel("Frames EMG/IMU: 0 / 0")
        self.label_events = QLabel("Eventos: 0")

        for lbl in (
            self.label_status,
            self.label_timer,
            self.label_file_size,
            self.label_data_rate,
            self.label_frames,
            self.label_events,
        ):
            lbl.setStyleSheet("color: #D9E4E4")
            status_layout.addWidget(lbl)

        status_layout.addStretch(1)
        right_layout.addWidget(status_box)

        splitter.setSizes([420, 780])
        self._refresh_ports()

    # ------------------------------------------------------------------
    # Patient profiles (simple caching)
    # ------------------------------------------------------------------
    def _load_patient_profiles(self) -> None:
        profiles_path = cfg.PROJECT_ROOT / "data" / "patients" / "patients.json"
        data = load_json(str(profiles_path)) if profiles_path.exists() else {}
        suggestions = sorted({entry.get("patient_id", "") for entry in data.get("patients", []) if entry.get("patient_id")})
        current = self.input_patient_id.currentText().strip()
        self._loading_patient_ids = True
        self.input_patient_id.blockSignals(True)
        self.input_patient_id.clear()
        self.input_patient_id.addItem("")
        for patient_id in suggestions:
            self.input_patient_id.addItem(patient_id)
        if current:
            index = self.input_patient_id.findText(current, QtCore.Qt.MatchFlag.MatchFixedString)
            if index >= 0:
                self.input_patient_id.setCurrentIndex(index)
            else:
                self.input_patient_id.setEditText(current)
        self.input_patient_id.blockSignals(False)
        self._loading_patient_ids = False

    def _on_patient_id_committed(self) -> None:
        if getattr(self, "_loading_patient_ids", False):
            return
        patient_id = self.input_patient_id.currentText().strip()
        if not patient_id:
            return
        if self.input_patient_id.findText(patient_id, QtCore.Qt.MatchFlag.MatchFixedString) == -1:
            self.input_patient_id.addItem(patient_id)
        patient_dir = cfg.PROJECT_ROOT / "data" / "patients" / patient_id
        profile_path = patient_dir / "profile.json"
        if profile_path.exists():
            profile_data = load_json(str(profile_path))
            self._apply_profile(profile_data)

        sessions_dir = patient_dir / "sessions"
        next_number = 1
        if sessions_dir.exists():
            session_numbers: List[int] = []
            for child in sessions_dir.iterdir():
                if not child.is_dir():
                    continue
                parts = child.name.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    session_numbers.append(int(parts[1]))
            if session_numbers:
                next_number = max(session_numbers) + 1
        self.input_session_number.setValue(next_number)

    def _apply_profile(self, profile: Dict[str, object]) -> None:
        name = str(profile.get("name", ""))
        if name:
            self.input_patient_name.setText(name)
        age = profile.get("age")
        if isinstance(age, (int, float)):
            self.input_age.setValue(int(age))
        sex = profile.get("sex")
        if isinstance(sex, str):
            index = self.input_sex.findText(sex, QtCore.Qt.MatchFlag.MatchFixedString)
            if index >= 0:
                self.input_sex.setCurrentIndex(index)
        weight = profile.get("weight")
        if isinstance(weight, (int, float)):
            self.input_weight.setValue(float(weight))
        height = profile.get("height")
        if isinstance(height, (int, float)):
            self.input_height.setValue(float(height))
        pathology = profile.get("pathology")
        if isinstance(pathology, str):
            idx = self.input_pathology.findText(pathology)
            if idx < 0:
                self.input_pathology.addItem(pathology)
                idx = self.input_pathology.findText(pathology)
            if idx >= 0:
                self.input_pathology.setCurrentIndex(idx)
        side = profile.get("affected_side")
        if isinstance(side, str):
            idx = self.input_side.findText(side)
            if idx >= 0:
                self.input_side.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def _refresh_ports(self) -> None:
        self.port_combo.clear()
        ports = get_available_ports()
        for device, label in ports:
            self.port_combo.addItem(label, device)
        if not ports:
            self.port_combo.addItem("Sin puertos disponibles", "")
        self.port_combo.setCurrentIndex(0)

    def _toggle_connection(self) -> None:
        if self.is_connected:
            self._disconnect_serial()
        else:
            self._connect_serial()

    def _connect_serial(self) -> None:
        port_device = self.port_combo.currentData()
        if not port_device:
            QMessageBox.warning(self, "Conexión", "Selecciona un puerto serial disponible.")
            return
        self.serial_thread = SerialReaderThread(port_device)
        self.serial_thread.frame_received.connect(self._on_frame_received)
        self.serial_thread.connection_status.connect(self._on_connection_status)
        self.serial_thread.start()
        self.btn_toggle_connection.setEnabled(False)

    def _disconnect_serial(self) -> None:
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait(2000)
            self.serial_thread = None
        self.is_connected = False
        self.btn_toggle_connection.setText("Conectar")
        self.label_status.setText("Estado: Desconectado")

    def _on_connection_status(self, connected: bool, message: str) -> None:
        self.is_connected = connected
        self.label_status.setText(f"Estado: {message}")
        if connected:
            self.btn_toggle_connection.setText("Desconectar")
            self.btn_toggle_connection.setEnabled(True)
        else:
            self.btn_toggle_connection.setText("Conectar")
            self.btn_toggle_connection.setEnabled(True)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def _on_frame_received(self, frame: Dict) -> None:
        if frame["type"] == "EMG":
            if self.t0_emg is None:
                self.t0_emg = frame["timestamp_us"]
            t_sec = (frame["timestamp_us"] - self.t0_emg) / 1e6
            self.current_time_emg = t_sec

            filtered_ch0, rms_ch0 = self.emg_ch0_processor.process_sample(frame["ch0"])
            filtered_ch1, rms_ch1 = self.emg_ch1_processor.process_sample(frame["ch1"])

            self.time_emg_ch0[self.idx_emg_ch0] = t_sec
            self.data_emg_ch0[self.idx_emg_ch0] = filtered_ch0
            self.data_rms_ch0[self.idx_emg_ch0] = rms_ch0
            self.idx_emg_ch0 = (self.idx_emg_ch0 + 1) % len(self.time_emg_ch0)

            self.time_emg_ch1[self.idx_emg_ch1] = t_sec
            self.data_emg_ch1[self.idx_emg_ch1] = filtered_ch1
            self.data_rms_ch1[self.idx_emg_ch1] = rms_ch1
            self.idx_emg_ch1 = (self.idx_emg_ch1 + 1) % len(self.time_emg_ch1)

            self.label_rms_ch0.setText(self._format_rms_label(0, rms_ch0))
            self.label_rms_ch1.setText(self._format_rms_label(1, rms_ch1))
            self.current_rms_values[0] = float(rms_ch0)
            self.current_rms_values[1] = float(rms_ch1)

            if self.recording_state == RecordingState.RECORDING and self.session_recorder:
                self.session_recorder.record_emg(
                    frame["timestamp_us"],
                    frame.get("seq", 0),
                    frame["ch0"],
                    frame["ch1"],
                    filtered_ch0,
                    filtered_ch1,
                    rms_ch0,
                    rms_ch1,
                )

            self.emg_count += 1

        elif frame["type"] == "IMU":
            if self.t0_imu is None:
                self.t0_imu = frame["timestamp_us"]
            t_sec = (frame["timestamp_us"] - self.t0_imu) / 1e6
            self.current_time_imu = t_sec

            angle = self.angle_calculator.update(
                frame["ax"], frame["ay"], frame["az"],
                frame["gx"], frame["gy"], frame["gz"],
                frame["timestamp_us"],
            )
            self.last_angle = angle
            self.current_raw_angle = float(self.angle_calculator.last_uncalibrated_angle)

            self.time_imu[self.idx_imu] = t_sec
            self.data_angle[self.idx_imu] = angle
            self.idx_imu = (self.idx_imu + 1) % len(self.time_imu)
            self.label_angle.setText(f"Ángulo: {angle:.1f}°")

            if self.recording_state == RecordingState.RECORDING and self.session_recorder:
                self.session_recorder.record_imu(
                    frame["timestamp_us"],
                    frame.get("seq", 0),
                    frame["ax"],
                    frame["ay"],
                    frame["az"],
                    frame["gx"],
                    frame["gy"],
                    frame["gz"],
                    angle,
                )

            self.imu_count += 1

        self._update_stats_rate()

    def _update_plots(self) -> None:
        window = cfg.WINDOW_TIME_SEC
        if self.idx_emg_ch0 > 0:
            t_data = np.roll(self.time_emg_ch0, -self.idx_emg_ch0)
            y_data = np.roll(self.data_emg_ch0, -self.idx_emg_ch0)
            rms_data = np.roll(self.data_rms_ch0, -self.idx_emg_ch0)
            mask = (t_data >= self.current_time_emg - window) & (t_data <= self.current_time_emg)
            self.curve_ch0.setData(t_data[mask], y_data[mask])
            self.curve_rms_ch0.setData(t_data[mask], rms_data[mask])
            self.plot_ch0.setXRange(max(0, self.current_time_emg - window), self.current_time_emg, padding=0)

        if self.idx_emg_ch1 > 0:
            t_data = np.roll(self.time_emg_ch1, -self.idx_emg_ch1)
            y_data = np.roll(self.data_emg_ch1, -self.idx_emg_ch1)
            rms_data = np.roll(self.data_rms_ch1, -self.idx_emg_ch1)
            mask = (t_data >= self.current_time_emg - window) & (t_data <= self.current_time_emg)
            self.curve_ch1.setData(t_data[mask], y_data[mask])
            self.curve_rms_ch1.setData(t_data[mask], rms_data[mask])
            self.plot_ch1.setXRange(max(0, self.current_time_emg - window), self.current_time_emg, padding=0)

        if self.idx_imu > 0:
            t_data = np.roll(self.time_imu, -self.idx_imu)
            angle_data = np.roll(self.data_angle, -self.idx_imu)
            mask = (t_data >= self.current_time_imu - window) & (t_data <= self.current_time_imu)
            self.curve_angle.setData(t_data[mask], angle_data[mask])
            self.plot_angle.setXRange(max(0, self.current_time_imu - window), self.current_time_imu, padding=0)

    def _update_stats_rate(self) -> None:
        current_time = QtCore.QTime.currentTime()
        elapsed = self.last_stats_update.msecsTo(current_time)
        if elapsed >= 1000:
            emg_rate = self.emg_count / (elapsed / 1000.0)
            imu_rate = self.imu_count / (elapsed / 1000.0)
            self.label_frames.setText(f"Frames EMG/IMU: {self.emg_count} / {self.imu_count} ({emg_rate:.0f}/{imu_rate:.0f} sps)")
            self.emg_count = 0
            self.imu_count = 0
            self.last_stats_update = current_time

    # ------------------------------------------------------------------
    # Recording controls
    # ------------------------------------------------------------------
    def _on_record_clicked(self) -> None:
        if self.recording_state == RecordingState.PAUSED:
            self.recording_state = RecordingState.RECORDING
            self.label_status.setText("Estado: Grabando")
            self.btn_record.setEnabled(False)
            self.btn_pause.setEnabled(True)
            return
        if not self.is_connected:
            QMessageBox.warning(self, "Grabación", "Conecta el dispositivo antes de grabar.")
            return
        if not self.angle_calculator.calibrated:
            if QMessageBox.question(self, "Calibración", "¿Deseas calibrar el IMU antes de grabar?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                self._open_calibration()
                return
        self._ensure_session_ready()
        self._start_countdown()

    def _ensure_session_ready(self) -> None:
        if self.session_recorder is not None:
            return
        patient_id = self.input_patient_id.currentText().strip()
        if not patient_id:
            QMessageBox.warning(self, "Paciente", "Ingresa un ID de paciente para continuar.")
            raise RuntimeError("Missing patient")
        session_number = self.input_session_number.value()
        timestamp = datetime.now()
        self.current_session_id = f"session_{session_number:03d}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.session_recorder = SessionRecorder(patient_id, session_number, self.current_session_id)

    def _start_countdown(self) -> None:
        self.recording_state = RecordingState.COUNTDOWN
        self.pending_countdown = 3
        self.label_status.setText("Estado: Preparando grabación (3)")
        self.countdown_timer.start()
        self.btn_record.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def _advance_countdown(self) -> None:
        self.pending_countdown -= 1
        if self.pending_countdown > 0:
            self.label_status.setText(f"Estado: Preparando grabación ({self.pending_countdown})")
        else:
            self.countdown_timer.stop()
            self._begin_recording()

    def _begin_recording(self) -> None:
        self.recording_state = RecordingState.RECORDING
        self.label_status.setText("Estado: Grabando")
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(False)
        if self.session_recorder:
            self.session_recorder.start(emg_start_us=self.t0_emg, imu_start_us=self.t0_imu)
        self._record_wallclock_start = datetime.now()

    def _on_pause_clicked(self) -> None:
        if self.recording_state != RecordingState.RECORDING:
            return
        self.recording_state = RecordingState.PAUSED
        self.label_status.setText("Estado: Pausado")
        self.btn_pause.setEnabled(False)
        self.btn_record.setEnabled(True)

    def _on_stop_clicked(self) -> None:
        if self.recording_state not in (RecordingState.RECORDING, RecordingState.PAUSED):
            return
        self._finalize_session()

    def _on_reset_clicked(self) -> None:
        if self.recording_state == RecordingState.RECORDING:
            if QMessageBox.question(self, "Reiniciar", "La grabación está en curso. ¿Descartar y reiniciar?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
        self._reset_session()

    def _reset_session(self) -> None:
        if self.session_recorder:
            session_dir = self.session_recorder.session_dir
            if session_dir.exists() and not any(session_dir.iterdir()):
                shutil.rmtree(session_dir, ignore_errors=True)
        self.recording_state = RecordingState.IDLE
        self.label_status.setText("Estado: Inactivo")
        self.btn_record.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.session_recorder = None
        self.current_session_id = None
        self._record_wallclock_start = None
        self.events_list.clear()
        self.event_counter = 0
        self.label_events.setText("Eventos: 0")
        self.label_timer.setText("Tiempo: 00:00")
        self.label_file_size.setText("Archivo: 0.0 MB")
        self.label_data_rate.setText("Tasa: 0 KB/s")

    def _finalize_session(self) -> None:
        if not self.session_recorder:
            self._reset_session()
            return
        self._persist_patient_profile()
        metadata = self._collect_metadata()
        calibration = self._collect_calibration_snapshot()
        notes = self.notes_field.toPlainText()
        paths = self.session_recorder.finalize(metadata, calibration, notes)
        QMessageBox.information(
            self,
            "Sesión guardada",
            f"Datos almacenados en:\n{paths['data']}"
        )
        self._reset_session()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _persist_patient_profile(self) -> None:
        patient_id = self.input_patient_id.currentText().strip()
        if not patient_id:
            return
        patient_dir = cfg.PROJECT_ROOT / "data" / "patients" / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        profile = {
            "patient_id": patient_id,
            "name": self.input_patient_name.text().strip(),
            "age": self.input_age.value(),
            "sex": self.input_sex.currentText(),
            "weight": self.input_weight.value(),
            "height": self.input_height.value(),
            "pathology": self.input_pathology.currentText(),
            "affected_side": self.input_side.currentText(),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(profile, str(patient_dir / "profile.json"))

        registry_path = cfg.PROJECT_ROOT / "data" / "patients" / "patients.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry = load_json(str(registry_path)) if registry_path.exists() else {"patients": []}
        patients = registry.get("patients", [])
        entry = next((item for item in patients if item.get("patient_id") == patient_id), None)
        if entry:
            entry.update({"patient_id": patient_id, "name": profile["name"]})
        else:
            patients.append({"patient_id": patient_id, "name": profile["name"]})
        registry["patients"] = patients
        save_json(registry, str(registry_path))

    def _collect_metadata(self) -> Dict[str, object]:
        timestamp_end = datetime.now()
        start_dt = self._record_wallclock_start or timestamp_end
        duration_sec = 0
        if self.session_recorder:
            duration_sec = int(self.session_recorder.elapsed_seconds())
        metadata = {
            "session_id": self.current_session_id,
            "patient_id": self.input_patient_id.currentText().strip(),
            "session_number": self.input_session_number.value(),
            "session_type": self.session_type_combo.currentText(),
            "protocol": self.protocol_combo.currentText(),
            "date": start_dt.strftime("%Y-%m-%d"),
            "time_start": start_dt.strftime("%H:%M:%S"),
            "time_end": timestamp_end.strftime("%H:%M:%S"),
            "duration_sec": duration_sec,
            "settings_snapshot": {
                "EMG_FS": cfg.EMG_FS,
                "IMU_FS": cfg.IMU_FS,
                "RMS_WINDOW_MS": cfg.RMS_WINDOW_MS,
                "WINDOW_TIME_SEC": cfg.WINDOW_TIME_SEC,
            },
            "hardware_info": {
                "serial_port": self.port_combo.currentData(),
            },
            "clinical_notes": self.notes_field.toPlainText(),
        }
        return metadata

    def _collect_calibration_snapshot(self) -> Dict[str, object]:
        return {
            "imu_calibration": {
                "calibrated": self.angle_calculator.calibrated,
                "offset": getattr(self.angle_calculator, "offset", 0.0),
                "scale": getattr(self.angle_calculator, "scale", 1.0),
            },
            "emg_calibration": {
                "mvc_ch0": self.mvc_values.get(0),
                "mvc_ch1": self.mvc_values.get(1),
            },
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_mark_event(self) -> None:
        if self.recording_state != RecordingState.RECORDING:
            QMessageBox.warning(self, "Marcadores", "Inicia la grabación para marcar eventos.")
            return
        if not self.session_recorder:
            return
        timestamp_us = int((self.t0_emg or 0) + self.session_recorder.emg_sample_count * 1e6 / max(1, cfg.EMG_FS))
        timestamp_sec = self.session_recorder.elapsed_seconds()
        event_type = self.event_type_combo.currentText()
        description = self.custom_event_input.text().strip() if event_type == "Personalizado" else event_type
        event = EventMarker(
            timestamp_us=timestamp_us,
            timestamp_relative_sec=timestamp_sec,
            event_type=event_type.lower().replace(" ", "_"),
            description=description or event_type,
            metadata={"user_marked": True},
        )
        self.session_recorder.add_event(event)
        self._append_event_list(event)

    def _append_event_list(self, event: EventMarker) -> None:
        self.event_counter += 1
        display = f"#{self.event_counter:02d} | {event.timestamp_relative_sec:05.2f}s | {event.description}"
        item = QListWidgetItem(display)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, event)
        self.events_list.addItem(item)
        self.label_events.setText(f"Eventos: {self.events_list.count()}")

    def _on_edit_event(self) -> None:
        current_item = self.events_list.currentItem()
        if not current_item or not self.session_recorder:
            return
        event = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
        new_desc, ok = QtWidgets.QInputDialog.getText(self, "Editar evento", "Descripción", text=event.description)
        if ok:
            event.description = new_desc
            current_item.setText(f"#{self.events_list.row(current_item)+1:02d} | {event.timestamp_relative_sec:05.2f}s | {event.description}")

    def _on_remove_event(self) -> None:
        current_row = self.events_list.currentRow()
        if current_row < 0 or not self.session_recorder:
            return
        if QMessageBox.question(self, "Eliminar", "¿Eliminar marcador seleccionado?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        self.events_list.takeItem(current_row)
        if self.session_recorder:
            # Simple rebuild of event list from QListWidget
            events: List[EventMarker] = []
            for idx in range(self.events_list.count()):
                item = self.events_list.item(idx)
                evt = item.data(QtCore.Qt.ItemDataRole.UserRole)
                events.append(evt)
            self.session_recorder._events = events  # Internal reassignment
        self.label_events.setText(f"Eventos: {self.events_list.count()}")

    # ------------------------------------------------------------------
    # Status bar calculations
    # ------------------------------------------------------------------
    def _update_status_bar(self) -> None:
        if self.recording_state == RecordingState.RECORDING and self.session_recorder:
            elapsed = self.session_recorder.elapsed_seconds()
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self.label_timer.setText(f"Tiempo: {mins:02d}:{secs:02d}")
            total_samples = self.session_recorder.emg_sample_count + self.session_recorder.imu_sample_count
            approx_bytes = total_samples * 32
            self.label_file_size.setText(f"Archivo: {approx_bytes / (1024*1024):.2f} MB")
            if elapsed > 0:
                self.label_data_rate.setText(f"Tasa: {approx_bytes / 1024 / max(1, elapsed):.1f} KB/s")
        elif self.recording_state == RecordingState.IDLE:
            self.label_timer.setText("Tiempo: 00:00")
            self.label_data_rate.setText("Tasa: 0 KB/s")

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------
    def _open_calibration(self) -> None:
        if not self.is_connected:
            QMessageBox.warning(self, "Calibración", "Conecta el dispositivo antes de calibrar el IMU.")
            return

        dialog = CalibrationDialog(self)
        angle_timer = QTimer(self)

        def push_angle() -> None:
            if dialog.isVisible():
                dialog.set_current_angle(self.current_raw_angle)
            else:
                angle_timer.stop()

        angle_timer.setInterval(100)
        angle_timer.timeout.connect(push_angle)
        angle_timer.start()
        result = dialog.exec()
        angle_timer.stop()

        if result != QDialog.DialogCode.Accepted or not getattr(dialog, "calibration_done", False):
            return

        calib_data = dialog.get_calibration_data()
        if not calib_data:
            return

        mode = calib_data.get("mode")
        if mode == 1:
            raw_point = calib_data.get("angle_raw_point1")
            if raw_point is not None:
                self.angle_calculator.angle = float(raw_point)
                self.angle_calculator.last_uncalibrated_angle = float(raw_point)
            self.angle_calculator.calibrate_one_point(float(calib_data.get("angle_ref_point1", 0.0)))
        elif mode == 2:
            self.angle_calculator.calibrate_two_points(
                float(calib_data.get("angle_raw_point1", 0.0)),
                float(calib_data.get("angle_ref_point1", 0.0)),
                float(calib_data.get("angle_raw_point2", 90.0)),
                float(calib_data.get("angle_ref_point2", 90.0)),
            )

        self.current_raw_angle = float(self.angle_calculator.last_uncalibrated_angle)
        try:
            self.last_angle = float(self.angle_calculator.angle)
        except AttributeError:
            pass
        self.label_angle.setText(f"Ángulo: {self.last_angle:.1f}°")

        QMessageBox.information(
            self,
            "Calibración completada",
            "La calibración del IMU se aplicó correctamente.",
        )

    def _open_emg_normalization(self) -> None:
        if self.idx_emg_ch0 == 0 and self.idx_emg_ch1 == 0:
            QMessageBox.information(
                self,
                "Sin datos EMG",
                "Recibe señal EMG antes de capturar una MVC.",
            )

        dialog = EMGNormalizationDialog(dict(self.mvc_values), self)
        dialog.mvc_computed.connect(self._on_mvc_computed)

        rms_timer = QTimer(self)

        def push_rms(force: bool = False) -> None:
            if not dialog.isVisible() and not force:
                return
            dialog.set_current_rms(
                self.current_rms_values.get(0, 0.0),
                self.current_rms_values.get(1, 0.0),
            )

        rms_timer.setInterval(100)
        rms_timer.timeout.connect(push_rms)
        rms_timer.start()
        push_rms(True)
        dialog.exec()
        rms_timer.stop()

    def _on_mvc_computed(self, channel: int, value: float) -> None:
        self.mvc_values[channel] = value
        current_rms = self.current_rms_values.get(channel, 0.0)
        if channel == 0:
            self.label_rms_ch0.setText(self._format_rms_label(channel, current_rms))
        else:
            self.label_rms_ch1.setText(self._format_rms_label(channel, current_rms))
        QMessageBox.information(
            self,
            "MVC registrada",
            f"Canal {channel}: {value * 1000:.3f} mV",
        )

    # ------------------------------------------------------------------
    def _format_rms_label(self, channel: int, rms_value: float) -> str:
        text = f"RMS CH{channel}: {rms_value * 1000:.3f} mV"
        mvc = self.mvc_values.get(channel)
        if mvc and mvc > 0:
            percent = (rms_value / mvc) * 100
            text += f" ({percent:.1f}% MVC)"
        return text

    # ------------------------------------------------------------------
    def closeEvent(self, event):  # pragma: no cover - UI lifecycle hook
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait(1500)
        super().closeEvent(event)
