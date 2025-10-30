"""Diálogo de calibración IMU con modos manual y semiautomático."""
from __future__ import annotations

import sys
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)
from PyQt6.QtGui import QFont, QImage, QPixmap

try:  # Opcional según instalación local
    import cv2
except ImportError:  # pragma: no cover - entorno sin OpenCV
    cv2 = None

try:  # Opcional según instalación local
    import mediapipe as mp
except ImportError:  # pragma: no cover - entorno sin MediaPipe
    mp = None

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\(\) is deprecated.",
    category=UserWarning,
)

from config import settings as cfg


class CalibrationDialog(QDialog):
    """Diálogo para calibración del IMU con modos manual y semiautomático."""

    MANUAL_TAB = 0
    SEMIAUTO_TAB = 1

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Calibración IMU")
        self.setModal(True)
        self.setMinimumWidth(560)

        self.calibration_done = False
        self.calibration_source: Optional[str] = None
        self.calibration_result: Dict[str, Any] = {
            "mode": cfg.CALIBRATION_POINTS,
            "angle_raw_point1": None,
            "angle_ref_point1": 0.0,
            "angle_raw_point2": None,
            "angle_ref_point2": None,
        }

        self.current_raw_angle: float = 0.0

        self._mediapipe_available = cv2 is not None and mp is not None
        self._manual_mode = cfg.CALIBRATION_POINTS
        self._manual_angle_raw_point1: Optional[float] = None
        self._manual_angle_raw_point2: Optional[float] = None
        self._manual_ref_point1: float = 0.0
        self._manual_ref_point2: float = 90.0

        self._semiauto_mode = cfg.CALIBRATION_POINTS
        self._semiauto_targets: List[float] = []
        self._semiauto_captures: List[Dict[str, float]] = []
        self._semiauto_stability: List[int] = []
        self._semiauto_current_target = 0
        self._semiauto_timer: Optional[QTimer] = None
        self._semiauto_active = False
        self._pose: Any = None
        self._video_capture: Any = None
        self._preview_size: tuple[int, int] = (424, 240)
        self._preview_fps_limit: int = 15
        self.semiauto_preview_label: Optional[QLabel] = None

        self._build_ui()
        self._manual_reset_state()
        self._update_semiauto_targets()

    # ------------------------------------------------------------------
    # Construcción de UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 12)
        main_layout.setSpacing(12)

        title = QLabel("🎯 Calibración del Ángulo de Rodilla")
        title.setFont(QFont("Avenir", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #D9E4E4; padding: 10px;")
        main_layout.addWidget(title)

        description = QLabel(
            "Selecciona el método de calibración que prefieras. El modo semiautomático requiere cámara."
        )
        description.setWordWrap(True)
        description.setStyleSheet("padding: 0 10px; color: #e4e4e4;")
        main_layout.addWidget(description)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setStyleSheet("background-color: #2B2B2B;")
        main_layout.addWidget(self.tab_widget)

        self.manual_tab = self._build_manual_tab()
        self.tab_widget.addTab(self.manual_tab, "Manual")

        self.semiauto_tab = self._build_semiauto_tab()
        self.tab_widget.addTab(self.semiauto_tab, "Semiautomática")

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        footer = QHBoxLayout()
        footer.addStretch()
        btn_close_dialog = QPushButton("Cerrar")
        btn_close_dialog.setProperty("category", "secondary")
        btn_close_dialog.clicked.connect(self.reject)
        footer.addWidget(btn_close_dialog)
        main_layout.addLayout(footer)

    def _build_manual_tab(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        info = QLabel("Captura manual de puntos utilizando la lectura instantánea del IMU.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #B8B8B8;")
        layout.addWidget(info)

        mode_group = QGroupBox("Modo de calibración")
        mode_layout = QVBoxLayout(mode_group)
        self.manual_radio_1 = QRadioButton("1 Punto (solo offset - pierna estirada = 0°)")
        self.manual_radio_2 = QRadioButton("2 Puntos (offset + escala - mayor precisión)")
        if cfg.CALIBRATION_POINTS == 1:
            self.manual_radio_1.setChecked(True)
        else:
            self.manual_radio_2.setChecked(True)
        self.manual_button_group = QButtonGroup(self)
        self.manual_button_group.addButton(self.manual_radio_1)
        self.manual_button_group.addButton(self.manual_radio_2)
        mode_layout.addWidget(self.manual_radio_1)
        mode_layout.addWidget(self.manual_radio_2)
        layout.addWidget(mode_group)

        self.manual_radio_1.toggled.connect(self._manual_on_mode_changed)
        self.manual_radio_2.toggled.connect(self._manual_on_mode_changed)

        point1_group = QGroupBox("📍 Punto 1: Pierna estirada")
        point1_layout = QVBoxLayout(point1_group)

        p1_inst = QLabel("1. Extiende completamente la rodilla\n2. Haz clic en 'Capturar Punto 1'")
        p1_inst.setStyleSheet("color: #B8B8B8;")
        point1_layout.addWidget(p1_inst)

        p1_angle_layout = QHBoxLayout()
        p1_angle_layout.addWidget(QLabel("Ángulo de referencia (°):"))
        self.manual_spin_point1 = QSpinBox()
        self.manual_spin_point1.setRange(-30, 30)
        self.manual_spin_point1.setValue(0)
        self.manual_spin_point1.setToolTip("Típicamente 0° = pierna completamente estirada")
        p1_angle_layout.addWidget(self.manual_spin_point1)
        p1_angle_layout.addStretch()
        point1_layout.addLayout(p1_angle_layout)

        self.btn_manual_capture_p1 = QPushButton("Capturar Punto 1")
        self.btn_manual_capture_p1.clicked.connect(self._manual_capture_point1)
        point1_layout.addWidget(self.btn_manual_capture_p1)

        self.manual_label_status_p1 = QLabel("❌ No capturado")
        self.manual_label_status_p1.setStyleSheet("color: #9C3428; padding: 5px;")
        point1_layout.addWidget(self.manual_label_status_p1)

        layout.addWidget(point1_group)

        self.manual_point2_group = QGroupBox("📍 Punto 2: Rodilla flexionada")
        point2_layout = QVBoxLayout(self.manual_point2_group)

        p2_inst = QLabel("1. Flexiona la rodilla a un ángulo conocido (ej. 90°)\n2. Haz clic en 'Capturar Punto 2'")
        p2_inst.setStyleSheet("color: #B8B8B8;")
        point2_layout.addWidget(p2_inst)

        p2_angle_layout = QHBoxLayout()
        p2_angle_layout.addWidget(QLabel("Ángulo de referencia (°):"))
        self.manual_spin_point2 = QSpinBox()
        self.manual_spin_point2.setRange(30, 150)
        self.manual_spin_point2.setValue(90)
        self.manual_spin_point2.setToolTip("Ángulo real de flexión (usa goniómetro si es posible)")
        p2_angle_layout.addWidget(self.manual_spin_point2)
        p2_angle_layout.addStretch()
        point2_layout.addLayout(p2_angle_layout)

        self.btn_manual_capture_p2 = QPushButton("Capturar Punto 2")
        self.btn_manual_capture_p2.clicked.connect(self._manual_capture_point2)
        point2_layout.addWidget(self.btn_manual_capture_p2)

        self.manual_label_status_p2 = QLabel("❌ No capturado")
        self.manual_label_status_p2.setStyleSheet("color: #9C3428; padding: 5px;")
        point2_layout.addWidget(self.manual_label_status_p2)

        layout.addWidget(self.manual_point2_group)

        layout.addStretch(1)

        buttons = QHBoxLayout()
        self.btn_manual_reset = QPushButton("Reiniciar capturas")
        self.btn_manual_reset.setProperty("category", "secondary")
        self.btn_manual_reset.clicked.connect(self._manual_reset_state)
        buttons.addWidget(self.btn_manual_reset)

        buttons.addStretch(1)

        self.btn_manual_finish = QPushButton("✓ Finalizar calibración")
        self.btn_manual_finish.clicked.connect(self._manual_finish)
        self.btn_manual_finish.setEnabled(False)
        buttons.addWidget(self.btn_manual_finish)

        layout.addLayout(buttons)
        return widget

    def _build_semiauto_tab(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)


        info = QLabel(
            "Solicita al paciente que realice una flexión lenta mientras MediaPipe estima el ángulo "
            "de la rodilla. El sistema capturará automáticamente los puntos necesarios para la calibración."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #B8B8B8;")
        layout.addWidget(info)

        mode_group = QGroupBox("Modo de calibración")
        mode_layout = QHBoxLayout(mode_group)
        self.semiauto_radio_1 = QRadioButton("1 Punto (solo offset)")
        self.semiauto_radio_2 = QRadioButton("2 Puntos (extensión + flexión)")
        if cfg.CALIBRATION_POINTS == 1:
            self.semiauto_radio_1.setChecked(True)
        else:
            self.semiauto_radio_2.setChecked(True)
        self.semiauto_button_group = QButtonGroup(self)
        self.semiauto_button_group.addButton(self.semiauto_radio_1)
        self.semiauto_button_group.addButton(self.semiauto_radio_2)
        mode_layout.addWidget(self.semiauto_radio_1)
        mode_layout.addWidget(self.semiauto_radio_2)
        layout.addWidget(mode_group)

        self.semiauto_radio_1.toggled.connect(self._on_semiauto_mode_changed)
        self.semiauto_radio_2.toggled.connect(self._on_semiauto_mode_changed)

        leg_layout = QHBoxLayout()
        leg_layout.addWidget(QLabel("Pierna objetivo:"))
        self.semiauto_leg_combo = QComboBox()
        self.semiauto_leg_combo.addItem("Derecha", "right")
        self.semiauto_leg_combo.addItem("Izquierda", "left")
        leg_layout.addWidget(self.semiauto_leg_combo)
        leg_layout.addStretch(1)
        layout.addLayout(leg_layout)

        preview_group = QGroupBox("Vista previa cámara")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)

        self.semiauto_preview_label = QLabel("Vista previa no disponible")
        self.semiauto_preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.semiauto_preview_label.setMinimumSize(*self._preview_size)
        self.semiauto_preview_label.setMaximumHeight(self._preview_size[1] + 12)
        self.semiauto_preview_label.setMaximumWidth(self._preview_size[0] + 16)
        self.semiauto_preview_label.setStyleSheet(
            "background-color: #1F1F21; border: 1px solid #3A3A3C; border-radius: 6px; color: #7F8C8D;"
        )
        preview_layout.addWidget(self.semiauto_preview_label)
        self._set_preview_message("Vista previa no disponible")
        layout.addWidget(preview_group)

        indicator_group = QGroupBox("Estado de detección")
        indicator_layout = QVBoxLayout(indicator_group)
        indicator_layout.setSpacing(8)

        self.pose_indicator_label = QLabel("Detección inactiva")
        self.pose_indicator_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pose_indicator_label.setMinimumHeight(42)
        indicator_layout.addWidget(self.pose_indicator_label)

        values_layout = QHBoxLayout()
        self.semiauto_pose_angle_label = QLabel("Ángulo visión: --°")
        self.semiauto_pose_angle_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        values_layout.addWidget(self.semiauto_pose_angle_label)

        self.semiauto_imu_label = QLabel("Ángulo IMU: --°")
        self.semiauto_imu_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        values_layout.addWidget(self.semiauto_imu_label)

        indicator_layout.addLayout(values_layout)
        layout.addWidget(indicator_group)

        progress_group = QGroupBox("Progreso de captura")
        progress_layout = QVBoxLayout(progress_group)
        self.semiauto_progress = QProgressBar()
        self.semiauto_progress.setRange(0, 1)
        self.semiauto_progress.setValue(0)
        self.semiauto_progress.setTextVisible(True)
        progress_layout.addWidget(self.semiauto_progress)

        self.semiauto_status_label = QLabel("Inicia la captura para registrar los puntos de calibración.")
        self.semiauto_status_label.setWordWrap(True)
        self.semiauto_status_label.setStyleSheet("color: #B8B8B8;")
        progress_layout.addWidget(self.semiauto_status_label)
        layout.addWidget(progress_group)

        layout.addStretch(1)

        buttons = QHBoxLayout()
        self.btn_semiauto_start = QPushButton("▶ Iniciar captura")
        self.btn_semiauto_start.clicked.connect(self._semiauto_start)
        buttons.addWidget(self.btn_semiauto_start)

        self.btn_semiauto_stop = QPushButton("⏹ Detener")
        self.btn_semiauto_stop.setProperty("category", "secondary")
        self.btn_semiauto_stop.clicked.connect(lambda: self._semiauto_stop(reset_progress=False, message="Captura detenida."))
        self.btn_semiauto_stop.setEnabled(False)
        buttons.addWidget(self.btn_semiauto_stop)

        buttons.addStretch(1)

        self.btn_semiauto_apply = QPushButton("✓ Aplicar calibración")
        self.btn_semiauto_apply.clicked.connect(self._semiauto_finalize)
        self.btn_semiauto_apply.setEnabled(False)
        buttons.addWidget(self.btn_semiauto_apply)

        layout.addLayout(buttons)

        if not self._mediapipe_available:
            warning = (
                "El modo semiautomático requiere las librerías 'mediapipe' y 'opencv-python'. "
                "Instálalas en tu entorno para habilitarlo."
            )
            self.btn_semiauto_start.setEnabled(False)
            self._update_semiauto_status(warning)
            self._set_pose_indicator("idle", "Librerías no disponibles")
        else:
            self._set_pose_indicator("idle", "Detección inactiva")

        return widget

    # ------------------------------------------------------------------
    # Modo manual
    # ------------------------------------------------------------------
    def _manual_on_mode_changed(self) -> None:
        self._manual_mode = 1 if self.manual_radio_1.isChecked() else 2
        self._manual_reset_state()

    def _manual_reset_state(self, preserve_mode: bool = True) -> None:
        if not preserve_mode:
            self._manual_mode = 1 if self.manual_radio_1.isChecked() else 2
        self._manual_angle_raw_point1 = None
        self._manual_angle_raw_point2 = None
        self._manual_ref_point1 = float(self.manual_spin_point1.value())
        self._manual_ref_point2 = float(self.manual_spin_point2.value())
        self.manual_label_status_p1.setText("❌ No capturado")
        self.manual_label_status_p1.setStyleSheet("color: #9C3428; padding: 5px;")
        self.manual_label_status_p2.setText("❌ No capturado")
        self.manual_label_status_p2.setStyleSheet("color: #9C3428; padding: 5px;")
        self._refresh_manual_controls()

    def _refresh_manual_controls(self) -> None:
        is_two_point = self._manual_mode == 2
        self.manual_point2_group.setEnabled(is_two_point)
        can_finish = self._manual_angle_raw_point1 is not None and (
            not is_two_point or self._manual_angle_raw_point2 is not None
        )
        self.btn_manual_capture_p2.setEnabled(is_two_point and self._manual_angle_raw_point1 is not None)
        self.btn_manual_finish.setEnabled(can_finish)

    def _manual_capture_point1(self) -> None:
        if not hasattr(self, "current_raw_angle"):
            QMessageBox.warning(self, "IMU no disponible", "No hay datos del IMU. Asegúrate de que esté conectado.")
            return
        self._manual_angle_raw_point1 = float(self.current_raw_angle)
        self._manual_ref_point1 = float(self.manual_spin_point1.value())
        self.manual_label_status_p1.setText(
            f"✓ Capturado: {self._manual_angle_raw_point1:.2f}° crudo → {self._manual_ref_point1:.1f}° real"
        )
        self.manual_label_status_p1.setStyleSheet("color: #27ae60; padding: 5px;")
        self._refresh_manual_controls()

    def _manual_capture_point2(self) -> None:
        if not hasattr(self, "current_raw_angle"):
            QMessageBox.warning(self, "IMU no disponible", "No hay datos del IMU.")
            return
        self._manual_angle_raw_point2 = float(self.current_raw_angle)
        self._manual_ref_point2 = float(self.manual_spin_point2.value())
        self.manual_label_status_p2.setText(
            f"✓ Capturado: {self._manual_angle_raw_point2:.2f}° crudo → {self._manual_ref_point2:.1f}° real"
        )
        self.manual_label_status_p2.setStyleSheet("color: #27ae60; padding: 5px;")
        self._refresh_manual_controls()

    def _manual_finish(self) -> None:
        if self._manual_angle_raw_point1 is None:
            QMessageBox.warning(self, "Calibración manual", "Debes capturar el Punto 1.")
            return
        if self._manual_mode == 2 and self._manual_angle_raw_point2 is None:
            QMessageBox.warning(self, "Calibración manual", "Debes capturar el Punto 2.")
            return

        point1 = {"raw": self._manual_angle_raw_point1, "ref": self._manual_ref_point1}
        point2 = None
        if self._manual_mode == 2:
            point2 = {"raw": self._manual_angle_raw_point2, "ref": self._manual_ref_point2}
        self._finalize_calibration(self._manual_mode, point1, point2, source="manual")

    # ------------------------------------------------------------------
    # Modo semiautomático
    # ------------------------------------------------------------------
    def _on_semiauto_mode_changed(self) -> None:
        self._update_semiauto_targets()
        self._semiauto_stop(reset_progress=True, message="Modo actualizado. Reinicia la captura.")

    def _update_semiauto_targets(self) -> None:
        self._semiauto_mode = 1 if self.semiauto_radio_1.isChecked() else 2
        ext = float(cfg.AUTO_CALIB_REFERENCE_EXT)
        flex = float(cfg.AUTO_CALIB_REFERENCE_FLEX)
        self._semiauto_targets = [ext] if self._semiauto_mode == 1 else [ext, flex]
        self._semiauto_stability = [0 for _ in self._semiauto_targets]
        self._semiauto_current_target = 0
        self._semiauto_captures.clear()
        self.semiauto_progress.setRange(0, max(1, len(self._semiauto_targets)))
        self.semiauto_progress.setValue(0)
        initial_msg = (
            "Extiende la pierna completamente y mantén estable para capturar el primer punto."
            if self._semiauto_targets
            else "Configura ángulos válidos desde la configuración."
        )
        self._update_semiauto_status(initial_msg)
        self.btn_semiauto_apply.setEnabled(False)

    def _semiauto_start(self) -> None:
        if not self._mediapipe_available:
            self._set_preview_message("Requiere MediaPipe y OpenCV")
            QMessageBox.information(
                self,
                "Calibración semiautomática",
                "Instala 'mediapipe' y 'opencv-python' para usar este modo.",
            )
            return
        if not self._semiauto_targets:
            self._set_preview_message("Configura ángulos válidos en ajustes")
            QMessageBox.warning(self, "Configuración", "No hay objetivos configurados para la calibración.")
            return

        self._semiauto_stop(reset_progress=True)

        if cv2 is None:
            self._set_preview_message("OpenCV no está disponible")
            QMessageBox.warning(self, "Calibración semiautomática", "OpenCV no está disponible en este entorno.")
            return

        camera_index = int(cfg.AUTO_CALIB_CAMERA_INDEX)
        if sys.platform.startswith("win"):
            self._video_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            self._video_capture = cv2.VideoCapture(camera_index)

        if not self._video_capture or not self._video_capture.isOpened():
            self._video_capture = None
            self._set_preview_message("No se pudo abrir la cámara")
            QMessageBox.warning(
                self,
                "Cámara no disponible",
                f"No se pudo abrir la cámara con índice {camera_index}. Verifica la configuración.",
            )
            return

        preview_width, preview_height = self._preview_size
        self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, preview_width)
        self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_height)
        target_fps = max(5, min(self._preview_fps_limit, int(cfg.AUTO_CALIB_FPS)))
        self._video_capture.set(cv2.CAP_PROP_FPS, float(target_fps))

        mp_pose = mp.solutions.pose
        self._pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=float(cfg.AUTO_CALIB_VISIBILITY_THRESHOLD),
            min_tracking_confidence=float(cfg.AUTO_CALIB_VISIBILITY_THRESHOLD),
        )

        fps = max(5, min(self._preview_fps_limit, int(cfg.AUTO_CALIB_FPS)))
        interval = max(15, int(1000 / fps))
        self._semiauto_timer = QTimer(self)
        self._semiauto_timer.timeout.connect(self._semiauto_process_frame)
        self._semiauto_timer.start(interval)

        self._semiauto_active = True
        self._semiauto_stability = [0 for _ in self._semiauto_targets]
        self._semiauto_current_target = 0
        self._semiauto_captures.clear()
        self.semiauto_progress.setRange(0, len(self._semiauto_targets))
        self.semiauto_progress.setValue(0)
        self.btn_semiauto_start.setEnabled(False)
        self.btn_semiauto_stop.setEnabled(True)
        self.btn_semiauto_apply.setEnabled(False)
        self._set_pose_indicator("idle", "Procesando cámara…")
        self._set_preview_message("Alinea tu rodilla dentro del recuadro")
        target = self._semiauto_targets[0]
        self._update_semiauto_status(
            f"Extiende la pierna hasta ~{target:.0f}° y mantén estable para capturar."
        )

    def _semiauto_stop(self, reset_progress: bool = False, message: Optional[str] = None) -> None:
        if self._semiauto_timer:
            self._semiauto_timer.stop()
            self._semiauto_timer.deleteLater()
            self._semiauto_timer = None
        if self._video_capture:
            self._video_capture.release()
            self._video_capture = None
        if self._pose:
            close_cb = getattr(self._pose, "close", None)
            if callable(close_cb):
                close_cb()
            self._pose = None
        self._semiauto_active = False
        self.btn_semiauto_start.setEnabled(self._mediapipe_available)
        self.btn_semiauto_stop.setEnabled(False)

        if reset_progress:
            self._semiauto_captures.clear()
            self._semiauto_stability = [0 for _ in self._semiauto_targets]
            self.semiauto_progress.setRange(0, max(1, len(self._semiauto_targets)))
            self.semiauto_progress.setValue(0)
            self.btn_semiauto_apply.setEnabled(False)
            self._set_preview_message("Vista previa en espera")
        else:
            self.btn_semiauto_apply.setEnabled(len(self._semiauto_captures) > 0)

        if message:
            self._update_semiauto_status(message)
        elif reset_progress:
            self._update_semiauto_status("Captura lista para reiniciarse.")
        else:
            self._update_semiauto_status("Captura detenida.")

        indicator_text = "Cámara detenida" if self._semiauto_captures else "Detección inactiva"
        self._set_pose_indicator("idle", indicator_text)

    def _semiauto_process_frame(self) -> None:
        if not self._video_capture or not self._video_capture.isOpened():
            self._set_pose_indicator("error", "Cámara desconectada")
            self._set_preview_message("Cámara desconectada")
            self._semiauto_stop(reset_progress=False, message="Se perdió la señal de la cámara.")
            return

        ret, frame = self._video_capture.read()
        if not ret or frame is None:
            self._set_pose_indicator("error", "Sin señal de video")
            self._set_preview_message("Sin señal de video")
            return

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._update_preview_frame(image_rgb)
        results = self._pose.process(image_rgb)
        if not results.pose_landmarks:
            self.semiauto_pose_angle_label.setText("Ángulo visión: --°")
            self._set_pose_indicator("error", "Pose no detectada")
            self._semiauto_reset_stability()
            return

        leg_key = self.semiauto_leg_combo.currentData()
        indexes = self._pose_indexes_for_leg(leg_key)
        landmarks = results.pose_landmarks.landmark

        if not self._landmarks_visible(landmarks, indexes):
            self.semiauto_pose_angle_label.setText("Ángulo visión: --°")
            self._set_pose_indicator("warn", "Visibilidad insuficiente")
            self._semiauto_reset_stability()
            return

        pose_angle = self._calculate_pose_angle(landmarks, indexes)
        if pose_angle is None or np.isnan(pose_angle):
            self.semiauto_pose_angle_label.setText("Ángulo visión: --°")
            self._set_pose_indicator("warn", "Ángulo no válido")
            self._semiauto_reset_stability()
            return

        self.semiauto_pose_angle_label.setText(f"Ángulo visión: {pose_angle:.1f}°")
        self._set_pose_indicator("ok", "Pose detectada")
        self._check_semiauto_target(pose_angle)

    def _semiauto_reset_stability(self) -> None:
        if 0 <= self._semiauto_current_target < len(self._semiauto_stability):
            self._semiauto_stability[self._semiauto_current_target] = 0

    def _check_semiauto_target(self, pose_angle: float) -> None:
        if not self._semiauto_active or self._semiauto_current_target >= len(self._semiauto_targets):
            return
        target = self._semiauto_targets[self._semiauto_current_target]
        tolerance = float(cfg.AUTO_CALIB_TOLERANCE_DEG)
        frames_needed = max(1, int(cfg.AUTO_CALIB_STABILITY_FRAMES))

        if abs(pose_angle - target) <= tolerance:
            self._semiauto_stability[self._semiauto_current_target] += 1
            current_frames = self._semiauto_stability[self._semiauto_current_target]
            self._update_semiauto_status(
                f"Mantén estable en {target:.0f}°. ({current_frames}/{frames_needed})"
            )
            if current_frames >= frames_needed:
                self._register_semiauto_capture(target, pose_angle)
        else:
            if self._semiauto_stability[self._semiauto_current_target] > 0:
                self._update_semiauto_status(f"Ajusta hacia {target:.0f}° para capturar el punto.")
            self._semiauto_stability[self._semiauto_current_target] = 0

    def _register_semiauto_capture(self, target: float, pose_angle: float) -> None:
        imu_angle = getattr(self, "current_raw_angle", None)
        if imu_angle is None:
            QMessageBox.warning(self, "IMU no disponible", "No se detectó ángulo del IMU durante la captura.")
            self._semiauto_stability[self._semiauto_current_target] = 0
            return

        capture = {"target": target, "pose_angle": float(pose_angle), "imu_angle": float(imu_angle)}
        self._semiauto_captures.append(capture)
        self.semiauto_progress.setValue(len(self._semiauto_captures))
        self._update_semiauto_status(
            f"Punto {len(self._semiauto_captures)} capturado ({pose_angle:.1f}°)."
        )

        self._semiauto_current_target += 1
        self._semiauto_stability = [0 for _ in self._semiauto_targets]

        if self._semiauto_current_target >= len(self._semiauto_targets):
            self._semiauto_stop(reset_progress=False, message="Puntos capturados. Revisa y aplica la calibración.")
        else:
            next_target = self._semiauto_targets[self._semiauto_current_target]
            self._update_semiauto_status(
                f"Flexiona lentamente hasta ~{next_target:.0f}° y mantén estable."
            )

    def _semiauto_finalize(self) -> None:
        if not self._semiauto_captures:
            QMessageBox.information(self, "Calibración", "No hay capturas semiautomáticas disponibles.")
            return

        mode = len(self._semiauto_captures)
        point1 = {
            "raw": self._semiauto_captures[0]["imu_angle"],
            "ref": self._semiauto_captures[0]["pose_angle"],
        }
        point2 = None
        if mode > 1:
            point2 = {
                "raw": self._semiauto_captures[1]["imu_angle"],
                "ref": self._semiauto_captures[1]["pose_angle"],
            }
        metadata = {
            "targets": [capture["target"] for capture in self._semiauto_captures],
            "leg": self.semiauto_leg_combo.currentData(),
            "auto": True,
        }
        self._finalize_calibration(mode, point1, point2, source="semiauto", metadata=metadata)

    # ------------------------------------------------------------------
    # Utilidades comunes
    # ------------------------------------------------------------------
    def _finalize_calibration(
        self,
        mode: int,
        point1: Dict[str, float],
        point2: Optional[Dict[str, float]] = None,
        *,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.calibration_result = {
            "mode": mode,
            "angle_raw_point1": point1["raw"],
            "angle_ref_point1": point1["ref"],
            "angle_raw_point2": point2["raw"] if point2 else None,
            "angle_ref_point2": point2["ref"] if point2 else None,
        }
        self.calibration_source = source
        if metadata:
            self.calibration_result["metadata"] = metadata
        self.calibration_result["source"] = source
        self.calibration_done = True
        self.accept()

    def _set_pose_indicator(self, state: str, message: str) -> None:
        palette = {
            "idle": ("#4A4A4D", "#E4E4E4"),
            "ok": ("#27ae60", "#0E0D0D"),
            "warn": ("#d68910", "#0E0D0D"),
            "error": ("#9C3428", "#FDEDEC"),
        }
        bg, fg = palette.get(state, palette["idle"])
        self.pose_indicator_label.setText(message)
        self.pose_indicator_label.setStyleSheet(
            f"border-radius: 10px; padding: 10px; font-weight: 600; background-color: {bg}; color: {fg};"
        )

    def _update_semiauto_status(self, text: str) -> None:
        self.semiauto_status_label.setText(text)

    def _set_preview_message(self, text: str) -> None:
        if self.semiauto_preview_label is None:
            return
        self.semiauto_preview_label.setPixmap(QPixmap())
        self.semiauto_preview_label.setText(text)

    def _update_preview_frame(self, image_rgb: np.ndarray) -> None:
        if self.semiauto_preview_label is None:
            return
        height, width, _ = image_rgb.shape
        bytes_per_line = width * 3
        q_image = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self._preview_size[0],
            self._preview_size[1],
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.semiauto_preview_label.setPixmap(pixmap)
        self.semiauto_preview_label.setText("")

    def _pose_indexes_for_leg(self, leg: str) -> tuple[int, int, int]:
        mp_pose = mp.solutions.pose
        if leg == "left":
            return (
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value,
            )
        return (
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        )

    @staticmethod
    def _landmark_to_array(landmark: Any) -> np.ndarray:
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)

    def _calculate_pose_angle(self, landmarks: Any, indexes: tuple[int, int, int]) -> Optional[float]:
        try:
            hip = self._landmark_to_array(landmarks[indexes[0]])
            knee = self._landmark_to_array(landmarks[indexes[1]])
            ankle = self._landmark_to_array(landmarks[indexes[2]])
        except (IndexError, AttributeError):
            return None

        vec1 = hip - knee
        vec2 = ankle - knee
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return None
        cos_angle = float(np.dot(vec1, vec2) / (norm1 * norm2))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        inner_angle = float(np.degrees(np.arccos(cos_angle)))
        return max(0.0, 180.0 - inner_angle)

    def _landmarks_visible(self, landmarks: Any, indexes: tuple[int, int, int]) -> bool:
        threshold = float(cfg.AUTO_CALIB_VISIBILITY_THRESHOLD)
        for idx in indexes:
            lm = landmarks[idx]
            visibility = getattr(lm, "visibility", 0.0)
            presence = getattr(lm, "presence", visibility)
            if max(visibility, presence) < threshold:
                return False
        return True

    def set_current_angle(self, angle: float) -> None:
        self.current_raw_angle = float(angle)
        self.semiauto_imu_label.setText(f"Ángulo IMU: {angle:.1f}°")

    def get_calibration_data(self) -> Dict[str, Any]:
        return deepcopy(self.calibration_result)

    def _on_tab_changed(self, index: int) -> None:
        if index != self.SEMIAUTO_TAB:
            self._semiauto_stop(reset_progress=False)

    def closeEvent(self, event):
        self._semiauto_stop(reset_progress=False)
        super().closeEvent(event)
