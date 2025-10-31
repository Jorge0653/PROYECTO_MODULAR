"""Offline analysis module for previously recorded EMG/IMU sessions."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

try:  # Optional exporters for image outputs
    from pyqtgraph.exporters import ImageExporter, SVGExporter
except ImportError:  # pragma: no cover - exporters optional
    ImageExporter = None
    SVGExporter = None

from config import settings as cfg
from core.session_loader import (
    PatientInfo,
    SessionDataset,
    SessionInfo,
    discover_patients,
    load_session,
)
from utils.helpers import load_json


def _color_from_settings(name: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    values = getattr(cfg, name, default)
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        return tuple(int(values[i]) for i in range(3))
    return default


@dataclass
class PlotChannel:
    name: str
    widget: pg.PlotWidget
    curve: pg.PlotDataItem
    overlay: Optional[pg.PlotDataItem] = None
    extras: Dict[str, pg.PlotDataItem] = field(default_factory=dict)

class AnalysisConfigDialog(QDialog):
    """Simple dialog to tweak analysis parameters."""

    def __init__(self, parent: Optional[QWidget], settings: Dict[str, float]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configurar análisis")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        self.fatigue_smoothing = QDoubleSpinBox()
        self.fatigue_smoothing.setRange(0.05, 5.0)
        self.fatigue_smoothing.setSingleStep(0.05)
        self.fatigue_smoothing.setValue(float(settings.get("fatigue_smoothing_sec", 0.5)))
        self.fatigue_smoothing.setSuffix(" s")
        form.addRow("Ventana suavizado fatiga", self.fatigue_smoothing)

        self.spectral_segment = QDoubleSpinBox()
        self.spectral_segment.setRange(1.0, 30.0)
        self.spectral_segment.setSingleStep(0.5)
        self.spectral_segment.setValue(float(settings.get("spectral_segment_sec", 5.0)))
        self.spectral_segment.setSuffix(" s")
        form.addRow("Duración segmento FFT", self.spectral_segment)

        self.spectral_max_freq = QDoubleSpinBox()
        self.spectral_max_freq.setRange(10.0, 2000.0)
        self.spectral_max_freq.setSingleStep(10.0)
        self.spectral_max_freq.setValue(float(settings.get("spectral_max_freq", 500.0)))
        self.spectral_max_freq.setSuffix(" Hz")
        form.addRow("Frecuencia máxima gráfica", self.spectral_max_freq)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Dict[str, float]:
        return {
            "fatigue_smoothing_sec": float(self.fatigue_smoothing.value()),
            "spectral_segment_sec": float(self.spectral_segment.value()),
            "spectral_max_freq": float(self.spectral_max_freq.value()),
        }


class SessionAnalysisWindow(QMainWindow):
    """Main window for browsing and analysing stored sessions."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Análisis de datos guardados")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setMinimumSize(1400, 820)

        self._patients: List[PatientInfo] = []
        self._current_dataset: Optional[SessionDataset] = None
        self._crosshair_lines: List[pg.InfiniteLine] = []
        self._event_markers: List[pg.InfiniteLine] = []
        self._emg_mode = "filtered"
        self.accel_curves: Dict[str, pg.PlotDataItem] = {}
        self.gyro_curves: Dict[str, pg.PlotDataItem] = {}
        self.accel_axis_checks: Dict[str, QCheckBox] = {}
        self.gyro_axis_checks: Dict[str, QCheckBox] = {}
        self._highlight_color = pg.mkColor(255, 180, 0)
        self._fatigue_channel = 0
        self._spectral_channel = 0
        self._spectral_signal = "filtered"
        self.accel_group: Optional[QGroupBox] = None
        self.gyro_group: Optional[QGroupBox] = None
        self._plot_area_layout: Optional[QVBoxLayout] = None
        self._plot_order: List[str] = []
        self._analysis_config: Dict[str, float] = {
            "fatigue_smoothing_sec": 0.5,
            "spectral_segment_sec": 5.0,
            "spectral_max_freq": 500.0,
        }

        self._build_ui()
        self._load_patients()

    # ------------------------------------------------------------------
    # UI creation
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)
        self.setCentralWidget(central)

        self.toolbar = self._build_toolbar()
        root_layout.addWidget(self.toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # Left panel -----------------------------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Buscar paciente o sesión…")
        self.search_box.textChanged.connect(self._apply_filters)
        left_layout.addWidget(self.search_box)

        self.type_filter = QComboBox()
        self.type_filter.addItem("Tipo de sesión: Todos", None)
        self.type_filter.currentIndexChanged.connect(self._apply_filters)
        left_layout.addWidget(self.type_filter)

        self.session_tree = QTreeWidget()
        self.session_tree.setHeaderLabels(["Paciente / Sesión", "Fecha", "Tipo"])
        self.session_tree.setColumnWidth(0, 220)
        self.session_tree.setUniformRowHeights(True)
        self.session_tree.setAlternatingRowColors(True)
        self.session_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        left_layout.addWidget(self.session_tree, stretch=1)

        refresh_btn = QPushButton("Actualizar lista")
        refresh_btn.clicked.connect(self._load_patients)
        left_layout.addWidget(refresh_btn)

        splitter.addWidget(left_panel)

        # Right panel ----------------------------------------------------
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(1, 3)

        self.header_group = self._build_session_header()
        right_splitter.addWidget(self.header_group)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        right_splitter.addWidget(bottom_splitter)

        plot_container = self._build_plot_area()
        bottom_splitter.addWidget(plot_container)

        self.analysis_tabs = self._build_analysis_tabs()
        bottom_splitter.addWidget(self.analysis_tabs)
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 2)

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar("Herramientas")
        toolbar.setIconSize(QtCore.QSize(20, 20))
        toolbar.setMovable(False)

        open_action = QtGui.QAction("📁 Abrir", self)
        open_action.triggered.connect(self._open_external_session)
        toolbar.addAction(open_action)

        export_button = QToolButton()
        export_button.setText("📤 Exportar")
        export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(export_button)

        data_menu = export_menu.addMenu("Datos filtrados")
        data_menu.addAction("CSV", lambda: self._export_filtered_data("csv"))
        data_menu.addAction("MAT", lambda: self._export_filtered_data("mat"))
        data_menu.addAction("EDF+", lambda: self._export_filtered_data("edf"))

        graphics_menu = export_menu.addMenu("Gráficos")
        graphics_menu.addAction("PNG", lambda: self._export_graphics("png"))
        graphics_menu.addAction("SVG", lambda: self._export_graphics("svg"))
        graphics_menu.addAction("PDF", lambda: self._export_graphics("pdf"))

        metrics_menu = export_menu.addMenu("Métricas")
        metrics_menu.addAction("JSON", lambda: self._export_metrics("json"))
        metrics_menu.addAction("XLSX", lambda: self._export_metrics("xlsx"))

        export_button.setMenu(export_menu)
        toolbar.addWidget(export_button)

        report_action = QtGui.QAction("📊 Generar reporte", self)
        report_action.triggered.connect(self._generate_report)
        toolbar.addAction(report_action)

        config_action = QtGui.QAction("⚙️ Configurar", self)
        config_action.triggered.connect(self._open_analysis_config)
        toolbar.addAction(config_action)

        toolbar.addSeparator()

        self.play_button = QtGui.QAction("Reproducir", self)
        self.play_button.setCheckable(True)
        self.play_button.triggered.connect(self._toggle_playback)
        toolbar.addAction(self.play_button)

        toolbar.addWidget(QLabel(" Velocidad:"))
        self.play_speed = QComboBox()
        self.play_speed.addItems(["0.5x", "1x", "2x", "4x"])
        self.play_speed.setCurrentIndex(1)
        toolbar.addWidget(self.play_speed)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Rango (s):"))
        self.range_spin = QtWidgets.QDoubleSpinBox()
        self.range_spin.setRange(1.0, 1200.0)
        self.range_spin.setValue(30.0)
        self.range_spin.setSingleStep(5.0)
        self.range_spin.valueChanged.connect(self._update_plot_window)
        toolbar.addWidget(self.range_spin)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" EMG vista:"))
        self.emg_view_combo = QComboBox()
        self.emg_view_combo.addItem("RMS", "rms")
        self.emg_view_combo.addItem("Filtrado", "filtered")
        self.emg_view_combo.addItem("Crudo", "raw")
        self.emg_view_combo.setCurrentIndex(1)
        self.emg_view_combo.currentIndexChanged.connect(self._on_emg_view_changed)
        toolbar.addWidget(self.emg_view_combo)

        toolbar.addSeparator()

        self.status_label = QLabel("Sin sesión seleccionada")
        toolbar.addWidget(self.status_label)

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(40)
        self.play_timer.timeout.connect(self._advance_playhead)

        return toolbar

    def _build_session_header(self) -> QGroupBox:
        group = QGroupBox("Información de la sesión")
        layout = QGridLayout(group)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(6)

        self.meta_labels: Dict[str, QLabel] = {}
        labels = {
            "patient": "Paciente",
            "session": "Sesión",
            "date": "Fecha",
            "duration": "Duración",
            "session_type": "Tipo",
            "protocol": "Protocolo",
            "notes": "Notas",
        }
        row = 0
        for key, title in labels.items():
            lbl_title = QLabel(f"{title}:")
            lbl_title.setStyleSheet("color: #9FA5B5; font-weight: 600;")
            value = QLabel("--")
            value.setStyleSheet("color: #E4E8EF;")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(lbl_title, row, 0)
            layout.addWidget(value, row, 1)
            self.meta_labels[key] = value
            row += 1

        self.quick_stats: Dict[str, QLabel] = {}
        stats_box = QGroupBox("Estadísticas rápidas")
        stats_layout = QGridLayout(stats_box)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.setHorizontalSpacing(14)
        stats_layout.setVerticalSpacing(4)

        stat_fields = [
            ("rom_total", "ROM total"),
            ("angle_range", "Ángulo min/max"),
            ("velocity_peak", "Velocidad máxima"),
            ("rms_peak", "Peaks EMG"),
            ("data_quality", "Calidad de datos"),
            ("event_count", "Eventos"),
            ("angle_cycles", "Ciclos estimados"),
        ]
        for idx, (key, title) in enumerate(stat_fields):
            title_lbl = QLabel(f"{title}:")
            value_lbl = QLabel("--")
            value_lbl.setStyleSheet("color: #D1F0FF; font-weight: 600;")
            stats_layout.addWidget(title_lbl, idx, 0)
            stats_layout.addWidget(value_lbl, idx, 1)
            self.quick_stats[key] = value_lbl

        layout.addWidget(stats_box, 0, 2, row, 1)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 1)

        return group

    def _build_plot_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._plot_area_layout = layout

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(10)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.valueChanged.connect(self._on_slider_changed)
        controls_row.addWidget(self.slider, stretch=1)

        layout.addLayout(controls_row)

        axis_container = QWidget()
        axis_layout = QHBoxLayout(axis_container)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.setSpacing(12)

        self.accel_group = QGroupBox("Acelerómetro (g)")
        accel_layout = QHBoxLayout(self.accel_group)
        accel_layout.setContentsMargins(8, 4, 8, 4)
        accel_layout.setSpacing(6)
        for axis in ("x", "y", "z"):
            checkbox = QCheckBox(axis.upper())
            checkbox.setChecked(axis in ("x", "y"))
            checkbox.toggled.connect(self._update_imu_curves)
            accel_layout.addWidget(checkbox)
            self.accel_axis_checks[axis] = checkbox
        axis_layout.addWidget(self.accel_group)

        self.gyro_group = QGroupBox("Giroscopio (°/s)")
        gyro_layout = QHBoxLayout(self.gyro_group)
        gyro_layout.setContentsMargins(8, 4, 8, 4)
        gyro_layout.setSpacing(6)
        for axis in ("x", "y", "z"):
            checkbox = QCheckBox(axis.upper())
            checkbox.setChecked(axis == "y")
            checkbox.toggled.connect(self._update_imu_curves)
            gyro_layout.addWidget(checkbox)
            self.gyro_axis_checks[axis] = checkbox
        axis_layout.addWidget(self.gyro_group)

        self.accel_group.setEnabled(False)
        self.gyro_group.setEnabled(False)
        layout.addWidget(axis_container)

        self.plots: Dict[str, PlotChannel] = {}
        pg.setConfigOptions(antialias=True)

        def create_plot(title: str, color: Tuple[int, int, int], units: Optional[str] = None) -> PlotChannel:
            widget = pg.PlotWidget(background="#1E1E1F")
            widget.showGrid(x=True, y=True, alpha=0.2)
            widget.setLabel("bottom", "Tiempo", units="s")
            if units:
                widget.setLabel("left", title, units=units)
            else:
                widget.setLabel("left", title)
            curve = widget.plot(pen=pg.mkPen(color=color, width=1.5))
            return PlotChannel(name=title, widget=widget, curve=curve)

        self.plots["emg_ch0"] = create_plot("EMG Canal 0", _color_from_settings("COLOR_CH0", (31, 119, 180)))
        self.plots["emg_ch1"] = create_plot("EMG Canal 1", _color_from_settings("COLOR_CH1", (255, 127, 14)))
        self.plots["angle"] = create_plot("Ángulo flexión", _color_from_settings("COLOR_ANGLE", (148, 103, 189)), units="°")
        self.plots["velocity"] = create_plot("Velocidad angular", (255, 215, 0), units="°/s")

        accel_plot = create_plot("Acelerómetro", (120, 200, 255), units="g")
        accel_plot.curve.setOpacity(0.0)
        self.plots["accel"] = accel_plot
        accel_colors = {
            "x": (255, 120, 120),
            "y": (120, 255, 160),
            "z": (120, 180, 255),
        }
        for axis, color in accel_colors.items():
            curve = accel_plot.widget.plot(pen=pg.mkPen(color=color, width=1.2))
            accel_plot.extras[axis] = curve
            self.accel_curves[axis] = curve

        gyro_plot = create_plot("Giroscopio", (255, 200, 120), units="°/s")
        gyro_plot.curve.setOpacity(0.0)
        self.plots["gyro"] = gyro_plot
        gyro_colors = {
            "x": (255, 150, 80),
            "y": (200, 140, 255),
            "z": (120, 220, 255),
        }
        for axis, color in gyro_colors.items():
            curve = gyro_plot.widget.plot(pen=pg.mkPen(color=color, width=1.2))
            gyro_plot.extras[axis] = curve
            self.gyro_curves[axis] = curve

        # Link X axes for synchronized zoom/pan
        for key in ("emg_ch1", "angle", "velocity", "accel", "gyro"):
            self.plots[key].widget.setXLink(self.plots["emg_ch0"].widget)

        for channel in self.plots.values():
            layout.addWidget(channel.widget, stretch=1)

        self.crosshair_pen = pg.mkPen(color=(240, 240, 240, 160), style=Qt.PenStyle.DashLine)
        for channel in self.plots.values():
            line = pg.InfiniteLine(angle=90, movable=False, pen=self.crosshair_pen)
            channel.widget.addItem(line)
            self._crosshair_lines.append(line)

        self._plot_order = list(self.plots.keys())

        self.events_list = QListWidget()
        self.events_list.setMaximumHeight(120)
        self.events_list.itemDoubleClicked.connect(self._jump_to_event)
        layout.addWidget(self.events_list)

        return container

    def _build_analysis_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # Tab 1: Metrics -------------------------------------------------
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tabs.addTab(self.metrics_table, "Métricas biomecánicas")

        # Tab 2: Fatigue -------------------------------------------------
        fatigue_panel = QWidget()
        fatigue_layout = QVBoxLayout(fatigue_panel)
        fatigue_layout.setContentsMargins(8, 8, 8, 8)
        fatigue_controls = QHBoxLayout()
        fatigue_controls.setSpacing(12)
        fatigue_controls.addWidget(QLabel("Canal EMG:"))
        self.fatigue_channel_combo = QComboBox()
        self.fatigue_channel_combo.addItem("Canal 0", 0)
        self.fatigue_channel_combo.addItem("Canal 1", 1)
        self.fatigue_channel_combo.currentIndexChanged.connect(self._on_fatigue_channel_changed)
        fatigue_controls.addWidget(self.fatigue_channel_combo)
        fatigue_controls.addSpacing(10)
        fatigue_controls.addWidget(QLabel("Resumen:"))
        self.fatigue_summary_label = QLabel("--")
        fatigue_controls.addWidget(self.fatigue_summary_label, stretch=1)
        fatigue_layout.addLayout(fatigue_controls)

        self.fatigue_plot = pg.PlotWidget(background="#1E1E1F")
        self.fatigue_plot.setLabel("left", "RMS promedio", units="mV")
        self.fatigue_plot.setLabel("bottom", "Tiempo", units="s")
        fatigue_layout.addWidget(self.fatigue_plot, stretch=1)
        tabs.addTab(fatigue_panel, "Análisis fatiga")

        # Tab 3: Events --------------------------------------------------
        events_panel = QWidget()
        events_layout = QVBoxLayout(events_panel)
        events_layout.setContentsMargins(8, 8, 8, 8)
        self.events_table = QTableWidget(0, 4)
        self.events_table.setHorizontalHeaderLabels(["Tiempo (s)", "Tipo", "Descripción", "Extra"])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.verticalHeader().setVisible(False)
        events_layout.addWidget(self.events_table, stretch=1)

        export_events_btn = QPushButton("Exportar eventos…")
        export_events_btn.clicked.connect(self._export_events_csv)
        events_layout.addWidget(export_events_btn)
        tabs.addTab(events_panel, "Eventos")

        # Tab 4: Comparative --------------------------------------------
        self.compare_table = QTableWidget(0, 8)
        self.compare_table.setHorizontalHeaderLabels([
            "Sesión",
            "Fecha",
            "ROM (°)",
            "Duración (s)",
            "Eventos",
            "Peak EMG C0 (mV)",
            "Peak EMG C1 (mV)",
            "ΔRMS C0 (%)",
        ])
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        self.compare_table.verticalHeader().setVisible(False)
        tabs.addTab(self.compare_table, "Comparativo")

        # Tab 5: Spectral ------------------------------------------------
        spectral_panel = QWidget()
        spectral_layout = QVBoxLayout(spectral_panel)
        spectral_layout.setContentsMargins(8, 8, 8, 8)
        spectral_controls = QHBoxLayout()
        spectral_controls.setSpacing(12)
        spectral_controls.addWidget(QLabel("Canal EMG:"))
        self.spectral_channel_combo = QComboBox()
        self.spectral_channel_combo.addItem("Canal 0", 0)
        self.spectral_channel_combo.addItem("Canal 1", 1)
        self.spectral_channel_combo.currentIndexChanged.connect(self._on_spectral_options_changed)
        spectral_controls.addWidget(self.spectral_channel_combo)
        spectral_controls.addWidget(QLabel("Modo:"))
        self.spectral_mode_combo = QComboBox()
        self.spectral_mode_combo.addItem("Filtrado", "filtered")
        self.spectral_mode_combo.addItem("RMS", "rms")
        self.spectral_mode_combo.addItem("Crudo", "raw")
        self.spectral_mode_combo.currentIndexChanged.connect(self._on_spectral_options_changed)
        spectral_controls.addWidget(self.spectral_mode_combo)
        spectral_controls.addSpacing(10)
        spectral_controls.addWidget(QLabel("Resumen:"))
        self.spectral_summary_label = QLabel("--")
        spectral_controls.addWidget(self.spectral_summary_label, stretch=1)
        spectral_layout.addLayout(spectral_controls)

        self.spectral_plot = pg.PlotWidget(background="#1E1E1F")
        self.spectral_plot.setLabel("left", "PSD", units="dB")
        self.spectral_plot.setLabel("bottom", "Frecuencia", units="Hz")
        spectral_layout.addWidget(self.spectral_plot, stretch=1)
        tabs.addTab(spectral_panel, "Espectral")

        return tabs

    # ------------------------------------------------------------------
    # Data loading and tree population
    # ------------------------------------------------------------------
    def _load_patients(self) -> None:
        self._patients = discover_patients()
        self._populate_tree()
        session_types = sorted({info.metadata.get("session_type", "") for patient in self._patients for info in patient.sessions})
        self.type_filter.blockSignals(True)
        self.type_filter.clear()
        self.type_filter.addItem("Tipo de sesión: Todos", None)
        for session_type in session_types:
            if session_type:
                self.type_filter.addItem(session_type, session_type)
        self.type_filter.blockSignals(False)
        self.status_label.setText(f"Pacientes encontrados: {len(self._patients)}")

    def _populate_tree(self) -> None:
        self.session_tree.clear()
        for patient in self._patients:
            patient_item = QTreeWidgetItem(self.session_tree)
            patient_item.setText(0, f"{patient.full_name} ({patient.patient_id})")
            patient_item.setData(0, Qt.ItemDataRole.UserRole, (patient.patient_id, None))
            patient_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon))
            for session in patient.sessions:
                self._add_session_item(patient_item, patient.patient_id, session)
        self.session_tree.expandAll()

    def _add_session_item(self, parent: QTreeWidgetItem, patient_id: str, session: SessionInfo) -> None:
        item = QTreeWidgetItem(parent)
        label = session.metadata.get("session_type") or session.session_id
        item.setText(0, label)
        item.setText(1, str(session.metadata.get("date", "")))
        item.setText(2, str(session.metadata.get("session_type", "")))
        item.setData(0, Qt.ItemDataRole.UserRole, (patient_id, session))
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogContentsView)
        item.setIcon(0, icon)

    def _apply_filters(self) -> None:
        query = self.search_box.text().lower().strip()
        session_type = self.type_filter.currentData()
        for i in range(self.session_tree.topLevelItemCount()):
            patient_item = self.session_tree.topLevelItem(i)
            visible_patient = False
            for j in range(patient_item.childCount()):
                session_item = patient_item.child(j)
                patient_id, session_info = session_item.data(0, Qt.ItemDataRole.UserRole)
                metadata = session_info.metadata if session_info else {}
                match_query = True
                if query:
                    text_blob = " ".join([
                        patient_item.text(0),
                        session_item.text(0),
                        session_item.text(1),
                        session_item.text(2),
                        patient_id,
                    ]).lower()
                    match_query = query in text_blob
                match_type = True
                if session_type:
                    match_type = metadata.get("session_type") == session_type
                is_visible = match_query and match_type
                session_item.setHidden(not is_visible)
                visible_patient = visible_patient or is_visible
            patient_item.setHidden(not visible_patient)

    # ------------------------------------------------------------------
    # Session selection handling
    # ------------------------------------------------------------------
    def _on_tree_selection_changed(self) -> None:
        items = self.session_tree.selectedItems()
        if not items:
            return
        item = items[0]
        patient_id, session_info = item.data(0, Qt.ItemDataRole.UserRole)
        if session_info is None:
            return
        dataset = load_session(patient_id, session_info)
        self._current_dataset = dataset
        self._render_session(dataset)

    def _render_session(self, dataset: SessionDataset) -> None:
        metadata = dataset.metadata
        profile = next((p.profile for p in self._patients if p.patient_id == dataset.patient_id), {})

        self.meta_labels["patient"].setText(f"{profile.get('name', '--')} ({dataset.patient_id})")
        self.meta_labels["session"].setText(dataset.session_info.session_id)
        self.meta_labels["date"].setText(f"{metadata.get('date', '--')} {metadata.get('time_start', '')}")
        duration = metadata.get("duration_sec") or dataset.compute_basic_metrics().get("time_span", 0)
        self.meta_labels["duration"].setText(f"{duration:.1f} s" if duration else "--")
        self.meta_labels["session_type"].setText(str(metadata.get("session_type", "--")))
        self.meta_labels["protocol"].setText(str(metadata.get("protocol", "--")))
        notes = dataset.notes.strip().splitlines()
        self.meta_labels["notes"].setText(notes[0] if notes else "--")

        self._update_plots(dataset)
        self._update_quick_stats(dataset)
        self._populate_metrics_tab(dataset)
        self._populate_fatigue_tab(dataset)
        self._populate_events_tab(dataset)
        self._populate_compare_tab(dataset)
        self._populate_spectral_tab(dataset)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _update_plots(self, dataset: SessionDataset) -> None:
        t_emg = dataset.time_axis("emg")
        t_imu = dataset.time_axis("imu")

        self.slider.blockSignals(True)
        if t_emg.size:
            self.slider.setMaximum(len(t_emg) - 1)
        elif t_imu.size:
            self.slider.setMaximum(len(t_imu) - 1)
        else:
            self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self._clear_event_markers()

        self._refresh_emg_curves(dataset)

        angle_series = dataset.angle_series()
        if angle_series.size and t_imu.size:
            limit = min(len(angle_series), len(t_imu))
            self.plots["angle"].curve.setData(t_imu[:limit], angle_series[:limit])
        else:
            self.plots["angle"].curve.clear()

        velocity = dataset.derived_series("velocity")
        if velocity.size == 0 and angle_series.size and t_imu.size > 1:
            velocity = np.gradient(angle_series, t_imu)
        if velocity.size and t_imu.size:
            limit = min(len(velocity), len(t_imu))
            self.plots["velocity"].curve.setData(t_imu[:limit], velocity[:limit])
        else:
            self.plots["velocity"].curve.clear()

        accel_available = self._update_accel_data(dataset, t_imu)
        gyro_available = self._update_gyro_data(dataset, t_imu)
        self._set_axis_controls_enabled(accel_available, gyro_available)
        self._apply_imu_visibility()

        for line in self._crosshair_lines:
            line.setPos(0.0)

        # Add event markers
        marker_pen = pg.mkPen(color=(255, 128, 0, 140), width=1.3)
        for event in dataset.events:
            ts = float(event.get("timestamp_relative_sec", 0.0))
            for channel in self.plots.values():
                marker = pg.InfiniteLine(angle=90, movable=False, pen=marker_pen)
                marker.setPos(ts)
                channel.widget.addItem(marker)
                self._event_markers.append(marker)

        self.events_list.clear()
        for event in dataset.events:
            ts = float(event.get("timestamp_relative_sec", 0.0))
            desc = event.get("description", "")
            item = QListWidgetItem(f"{ts:06.2f} s - {event.get('type', 'Evento')} - {desc}")
            item.setData(Qt.ItemDataRole.UserRole, ts)
            self.events_list.addItem(item)

        self._update_plot_window()

    def _clear_event_markers(self) -> None:
        for marker in self._event_markers:
            for channel in self.plots.values():
                if marker in channel.widget.items():
                    channel.widget.removeItem(marker)
        self._event_markers.clear()

    def _update_plot_window(self) -> None:
        if not self._current_dataset:
            return
        range_seconds = self.range_spin.value()
        center = self.slider.value()
        t_axis = self._current_dataset.time_axis("emg")
        if t_axis.size == 0:
            t_axis = self._current_dataset.time_axis("imu")
        if t_axis.size == 0:
            return
        if center >= len(t_axis):
            center = len(t_axis) - 1
        center_time = t_axis[center]
        min_time = max(0.0, center_time - range_seconds / 2)
        max_time = min(t_axis[-1], center_time + range_seconds / 2)
        for channel in self.plots.values():
            channel.widget.setXRange(min_time, max_time, padding=0.02)
        for line in self._crosshair_lines:
            line.setPos(center_time)

    def _refresh_emg_curves(self, dataset: SessionDataset) -> None:
        t_emg = dataset.time_axis("emg")
        mode = self._emg_mode or "filtered"
        if t_emg.size == 0:
            for ch in (0, 1):
                self.plots[f"emg_ch{ch}"].curve.clear()
            return
        for ch in (0, 1):
            values = dataset.emg_channel(ch, "rms" if mode == "rms" else mode)
            if values.size == 0:
                self.plots[f"emg_ch{ch}"].curve.clear()
                continue
            limit = min(len(values), len(t_emg))
            if mode == "rms":
                y_data = np.abs(values[:limit]) * 1e3
            else:
                y_data = values[:limit] * 1e3
            self.plots[f"emg_ch{ch}"].curve.setData(t_emg[:limit], y_data)
        self._update_emg_axis_labels()

    def _update_emg_axis_labels(self) -> None:
        if not hasattr(self, "emg_view_combo"):
            return
        label = self.emg_view_combo.currentText()
        for ch in (0, 1):
            self.plots[f"emg_ch{ch}"].widget.setLabel("left", f"EMG Canal {ch} ({label})", units="mV")

    def _update_accel_data(self, dataset: SessionDataset, t_imu: np.ndarray) -> bool:
        available = False
        for axis, curve in self.accel_curves.items():
            series = dataset.imu_series(f"accel_{axis}")
            checkbox = self.accel_axis_checks.get(axis)
            if series.size and t_imu.size:
                limit = min(len(series), len(t_imu))
                curve.setData(t_imu[:limit], series[:limit])
                available = True
            else:
                curve.clear()
            if checkbox:
                checkbox.blockSignals(True)
                checkbox.setEnabled(series.size > 0)
                if not series.size:
                    checkbox.setChecked(False)
                checkbox.blockSignals(False)
        return available

    def _update_gyro_data(self, dataset: SessionDataset, t_imu: np.ndarray) -> bool:
        available = False
        for axis, curve in self.gyro_curves.items():
            series = dataset.imu_series(f"gyro_{axis}")
            checkbox = self.gyro_axis_checks.get(axis)
            if series.size and t_imu.size:
                limit = min(len(series), len(t_imu))
                curve.setData(t_imu[:limit], series[:limit])
                available = True
            else:
                curve.clear()
            if checkbox:
                checkbox.blockSignals(True)
                checkbox.setEnabled(series.size > 0)
                if not series.size:
                    checkbox.setChecked(False)
                checkbox.blockSignals(False)
        return available

    def _set_axis_controls_enabled(self, accel_enabled: bool, gyro_enabled: bool) -> None:
        if self.accel_group:
            self.accel_group.setEnabled(accel_enabled)
        if self.gyro_group:
            self.gyro_group.setEnabled(gyro_enabled)

    def _apply_imu_visibility(self) -> None:
        for axis, curve in self.accel_curves.items():
            checkbox = self.accel_axis_checks.get(axis)
            visible = bool(checkbox and checkbox.isEnabled() and checkbox.isChecked())
            curve.setVisible(visible)
        for axis, curve in self.gyro_curves.items():
            checkbox = self.gyro_axis_checks.get(axis)
            visible = bool(checkbox and checkbox.isEnabled() and checkbox.isChecked())
            curve.setVisible(visible)

    def _on_emg_view_changed(self, _: int) -> None:
        self._emg_mode = self.emg_view_combo.currentData() or "filtered"
        if self._current_dataset:
            self._refresh_emg_curves(self._current_dataset)
            self._populate_fatigue_tab(self._current_dataset)
            self._populate_spectral_tab(self._current_dataset)

    def _on_fatigue_channel_changed(self, _: int) -> None:
        data = self.fatigue_channel_combo.currentData()
        self._fatigue_channel = int(data) if data is not None else 0
        if self._current_dataset:
            self._populate_fatigue_tab(self._current_dataset)

    def _on_spectral_options_changed(self, _: int) -> None:
        channel = self.spectral_channel_combo.currentData()
        mode = self.spectral_mode_combo.currentData()
        if channel is not None:
            self._spectral_channel = int(channel)
        if mode:
            self._spectral_signal = str(mode)
        if self._current_dataset:
            self._populate_spectral_tab(self._current_dataset)

    def _update_imu_curves(self, _: Optional[bool] = None) -> None:
        self._apply_imu_visibility()

    def _on_slider_changed(self, value: int) -> None:
        self._update_plot_window()

    def _advance_playhead(self) -> None:
        if not self._current_dataset:
            return
        max_val = self.slider.maximum()
        if max_val <= 0:
            return
        step = {
            0: 1,
            1: 2,
            2: 4,
            3: 8,
        }.get(self.play_speed.currentIndex(), 2)
        new_value = (self.slider.value() + step) % (max_val + 1)
        self.slider.setValue(new_value)

    def _toggle_playback(self, checked: bool) -> None:
        if checked:
            self.play_button.setText("Pausar")
            interval = int(40 / max(0.5, [0.5, 1.0, 2.0, 4.0][self.play_speed.currentIndex()]))
            self.play_timer.setInterval(max(10, interval))
            self.play_timer.start()
        else:
            self.play_button.setText("Reproducir")
            self.play_timer.stop()

    def _jump_to_event(self, item: QListWidgetItem) -> None:
        ts = item.data(Qt.ItemDataRole.UserRole)
        if ts is None:
            return
        dataset = self._current_dataset
        if not dataset:
            return
        t_axis = dataset.time_axis("emg")
        if t_axis.size == 0:
            t_axis = dataset.time_axis("imu")
        if t_axis.size == 0:
            return
        idx = int(np.argmin(np.abs(t_axis - ts)))
        self.slider.setValue(idx)

    # ------------------------------------------------------------------
    # Metrics and tabs population
    # ------------------------------------------------------------------
    def _update_quick_stats(self, dataset: SessionDataset) -> None:
        metrics = dataset.compute_basic_metrics()
        rom_total = metrics.get("rom_total")
        if rom_total is not None:
            self.quick_stats["rom_total"].setText(f"{rom_total:.2f} °")
        else:
            self.quick_stats["rom_total"].setText("--")

        if metrics.get("angle_min") is not None and metrics.get("angle_max") is not None:
            self.quick_stats["angle_range"].setText(
                f"{metrics['angle_min']:.1f}° / {metrics['angle_max']:.1f}°"
            )
        else:
            self.quick_stats["angle_range"].setText("--")

        velocity_peak = metrics.get("velocity_peak")
        if velocity_peak is not None:
            self.quick_stats["velocity_peak"].setText(f"{velocity_peak:.1f} °/s")
        else:
            self.quick_stats["velocity_peak"].setText("--")

        peak0 = metrics.get("rms_ch0_peak")
        peak1 = metrics.get("rms_ch1_peak")
        if peak0 is not None and peak1 is not None:
            self.quick_stats["rms_peak"].setText(
                f"C0 {peak0 * 1000:.2f} mV | C1 {peak1 * 1000:.2f} mV"
            )
        else:
            self.quick_stats["rms_peak"].setText("--")

        qual_emg = metrics.get("valid_emg_packets", 0)
        qual_imu = metrics.get("valid_imu_packets", 0)
        self.quick_stats["data_quality"].setText(
            f"EMG {qual_emg * 100:.0f}% | IMU {qual_imu * 100:.0f}%"
        )

        events = metrics.get("event_count")
        self.quick_stats["event_count"].setText(str(int(events)) if events is not None else "--")

        cycles = metrics.get("angle_cycles")
        if cycles is not None:
            self.quick_stats["angle_cycles"].setText(f"{cycles:.0f}")
        else:
            self.quick_stats["angle_cycles"].setText("--")

    def _populate_metrics_tab(self, dataset: SessionDataset) -> None:
        metrics = dataset.compute_basic_metrics()
        rows = [
            ("Duración", metrics.get("duration_sec"), "s"),
            ("ROM total", metrics.get("rom_total"), "°"),
            ("Ángulo mínimo", metrics.get("angle_min"), "°"),
            ("Ángulo máximo", metrics.get("angle_max"), "°"),
            ("Ángulo medio", metrics.get("angle_mean"), "°"),
            ("Desvío estándar ángulo", metrics.get("angle_std"), "°"),
            ("Velocidad máxima", metrics.get("velocity_peak"), "°/s"),
            ("Ciclos estimados", metrics.get("angle_cycles"), ""),
            ("Peak RMS C0", metrics.get("rms_ch0_peak"), "mV"),
            ("Peak RMS C1", metrics.get("rms_ch1_peak"), "mV"),
            ("RMS medio C0", metrics.get("rms_ch0_mean"), "mV"),
            ("RMS medio C1", metrics.get("rms_ch1_mean"), "mV"),
            ("Balance EMG", metrics.get("emg_balance_pct"), "pct_direct"),
            ("Ratio Q/H", metrics.get("ratio_qh"), ""),
            ("Co-contracción", metrics.get("co_contraction"), ""),
            ("Eventos detectados", metrics.get("event_count"), ""),
            ("Calidad EMG", metrics.get("valid_emg_packets"), "pct_fraction"),
            ("Calidad IMU", metrics.get("valid_imu_packets"), "pct_fraction"),
        ]
        self.metrics_table.setRowCount(len(rows))
        for row_idx, (label, value, unit) in enumerate(rows):
            lbl_item = QTableWidgetItem(label)
            lbl_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.metrics_table.setItem(row_idx, 0, lbl_item)
            if value is None:
                text = "--"
            elif unit == "mV":
                text = f"{value * 1000:.3f} {unit}"
            elif unit == "pct_fraction":
                text = f"{value * 100:.1f} %"
            elif unit == "pct_direct":
                text = f"{value:.1f} %"
            elif unit == "s":
                text = f"{value:.1f} {unit}"
            elif unit:
                text = f"{value:.3f} {unit}"
            else:
                text = f"{value:.2f}" if isinstance(value, float) else str(value)
            val_item = QTableWidgetItem(text)
            val_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.metrics_table.setItem(row_idx, 1, val_item)
        self.metrics_table.resizeRowsToContents()

    def _populate_fatigue_tab(self, dataset: SessionDataset) -> None:
        self.fatigue_plot.clear()
        channel = getattr(self, "_fatigue_channel", 0)
        rms = dataset.emg_channel(channel, "rms")
        t_axis = dataset.time_axis("emg")
        summary_text = "--"
        if rms.size == 0 or t_axis.size == 0:
            info = pg.TextItem("Sin datos EMG para análisis", anchor=(0.5, 0.5), color="w")
            self.fatigue_plot.addItem(info)
            if hasattr(self, "fatigue_summary_label"):
                self.fatigue_summary_label.setText("Sin datos disponibles")
            return

        limit = min(len(rms), len(t_axis))
        t_axis = t_axis[:limit]
        rms_mv = np.abs(rms[:limit]) * 1000.0
        base_pen = pg.mkPen(color=(80, 180, 255, 110), width=1.2)
        smooth_pen = pg.mkPen(color=(0, 210, 120), width=2)
        self.fatigue_plot.plot(t_axis, rms_mv, pen=base_pen)
        config = getattr(self, "_analysis_config", {})
        smoothing_sec = float(config.get("fatigue_smoothing_sec", 0.5))
        if t_axis.size > 1:
            dt = float(np.median(np.diff(t_axis)))
        else:
            dt = 0.0
        if dt > 0:
            window = max(3, int(round(smoothing_sec / dt)))
        else:
            window = max(3, len(rms_mv) // 100)
        if window % 2 == 0:
            window += 1
        window = min(window, len(rms_mv))
        if window < 3 or len(rms_mv) < 3:
            smooth = rms_mv
        else:
            kernel = np.ones(window, dtype=float) / max(window, 1)
            smooth = np.convolve(rms_mv, kernel, mode="same")
        self.fatigue_plot.plot(t_axis, smooth, pen=smooth_pen)

        metrics = dataset.compute_fatigue_metrics(channel)
        summary_parts: List[str] = []
        duration = metrics.get("duration")
        if duration is not None:
            summary_parts.append(f"Duración {duration:.1f}s")
        decline = metrics.get("rms_decline_pct")
        if decline is not None:
            summary_parts.append(f"ΔRMS {decline:.1f}%")
        median_freq = metrics.get("median_frequency")
        if median_freq is not None:
            summary_parts.append(f"MF {median_freq:.1f} Hz")
        slope = metrics.get("rms_slope")
        if slope is not None:
            summary_parts.append(f"Pendiente {slope * 1000:.3f} mV/s")
        if summary_parts:
            summary_text = " | ".join(summary_parts)

        if decline is not None:
            annotation = pg.TextItem(f"ΔRMS {decline:.1f}%", anchor=(1, 0), color=(255, 230, 180))
            annotation.setPos(t_axis[-1], smooth[-1] if smooth.size else rms_mv[-1])
            self.fatigue_plot.addItem(annotation)

        if hasattr(self, "fatigue_summary_label"):
            self.fatigue_summary_label.setText(summary_text)

    def _populate_spectral_tab(self, dataset: SessionDataset) -> None:
        self.spectral_plot.clear()
        channel = getattr(self, "_spectral_channel", 0)
        signal_mode = getattr(self, "_spectral_signal", "filtered")
        summary_text = "--"

        series = dataset.emg_channel(channel, "rms" if signal_mode == "rms" else signal_mode)
        times = dataset.time_axis("emg")
        if series.size == 0 or times.size == 0:
            info = pg.TextItem("Sin datos EMG disponibles", anchor=(0.5, 0.5), color="w")
            self.spectral_plot.addItem(info)
            if hasattr(self, "spectral_summary_label"):
                self.spectral_summary_label.setText("Sin datos disponibles")
            return

        limit = min(len(series), len(times))
        times = np.asarray(times[:limit], dtype=float)
        config = getattr(self, "_analysis_config", {})
        segment_sec = float(config.get("spectral_segment_sec", 5.0))
        max_freq = float(config.get("spectral_max_freq", 500.0))
        if times.size and segment_sec > 0 and (times[-1] - times[0]) > segment_sec:
            start_time = times[-1] - segment_sec
            start_idx = int(np.searchsorted(times, start_time))
        else:
            start_idx = 0
        signal = np.asarray(series[start_idx:limit], dtype=float)
        times = times[start_idx:limit]
        if times.size < 64:
            info = pg.TextItem("Datos insuficientes para FFT", anchor=(0.5, 0.5), color="w")
            self.spectral_plot.addItem(info)
            if hasattr(self, "spectral_summary_label"):
                self.spectral_summary_label.setText("Datos insuficientes")
            return

        if signal_mode == "rms":
            signal = np.abs(signal)
        else:
            signal = signal - np.mean(signal)

        diffs = np.diff(times)
        if diffs.size == 0:
            info = pg.TextItem("Frecuencia de muestreo inválida", anchor=(0.5, 0.5), color="w")
            self.spectral_plot.addItem(info)
            if hasattr(self, "spectral_summary_label"):
                self.spectral_summary_label.setText("Frecuencia inválida")
            return
        dt = float(np.mean(diffs))
        if dt <= 0:
            info = pg.TextItem("Frecuencia de muestreo inválida", anchor=(0.5, 0.5), color="w")
            self.spectral_plot.addItem(info)
            if hasattr(self, "spectral_summary_label"):
                self.spectral_summary_label.setText("Frecuencia inválida")
            return
        fs = 1.0 / dt

        window = np.hanning(signal.size)
        spectrum = np.fft.rfft(signal * window)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
        psd = (np.abs(spectrum) ** 2) / max(signal.size, 1)
        psd_db = 10 * np.log10(psd + 1e-12)
        pen = pg.mkPen(color=(255, 180, 0), width=2)
        self.spectral_plot.plot(freqs, psd_db, pen=pen)

        peak_freq = None
        if psd.size:
            peak_idx = int(np.argmax(psd))
            peak_freq = float(freqs[peak_idx])
            peak_line = pg.InfiniteLine(peak_freq, pen=pg.mkPen(color=(255, 130, 0, 140), style=Qt.PenStyle.DashLine))
            self.spectral_plot.addItem(peak_line)
            peak_label = pg.TextItem(f"Pico {peak_freq:.1f} Hz", anchor=(0, 1), color=(255, 230, 180))
            peak_label.setPos(peak_freq, float(np.max(psd_db)))
            self.spectral_plot.addItem(peak_label)

        total_power = float(np.sum(psd))
        median_freq = None
        if total_power > 0:
            cumulative = np.cumsum(psd)
            idx = int(np.searchsorted(cumulative, total_power / 2.0))
            median_freq = float(freqs[min(idx, len(freqs) - 1)])
            median_line = pg.InfiniteLine(median_freq, pen=pg.mkPen(color=(120, 220, 255, 160), style=Qt.PenStyle.DotLine))
            self.spectral_plot.addItem(median_line)

        if freqs.size:
            self.spectral_plot.setXRange(0, min(max_freq, float(freqs[-1])), padding=0.02)

        summary_parts: List[str] = []
        if peak_freq is not None:
            summary_parts.append(f"Pico {peak_freq:.1f} Hz")
        if median_freq is not None:
            summary_parts.append(f"MF {median_freq:.1f} Hz")
        if summary_parts:
            summary_text = " | ".join(summary_parts)

        if hasattr(self, "spectral_summary_label"):
            self.spectral_summary_label.setText(summary_text)

    def _populate_events_tab(self, dataset: SessionDataset) -> None:
        self.events_table.setRowCount(len(dataset.events))
        for row_idx, event in enumerate(dataset.events):
            ts = float(event.get("timestamp_relative_sec", 0.0))
            evt_type = str(event.get("type", ""))
            desc = str(event.get("description", ""))
            meta = {k: v for k, v in event.items() if k not in {"timestamp_us", "timestamp_relative_sec", "type", "description"}}
            self.events_table.setItem(row_idx, 0, QTableWidgetItem(f"{ts:.3f}"))
            self.events_table.setItem(row_idx, 1, QTableWidgetItem(evt_type))
            self.events_table.setItem(row_idx, 2, QTableWidgetItem(desc))
            self.events_table.setItem(row_idx, 3, QTableWidgetItem(str(meta) if meta else ""))
        self.events_table.resizeColumnsToContents()
        self.events_table.resizeRowsToContents()

    def _populate_compare_tab(self, dataset: SessionDataset) -> None:
        patient = next((p for p in self._patients if p.patient_id == dataset.patient_id), None)
        sessions = patient.sessions if patient else [dataset.session_info]
        self.compare_table.setRowCount(len(sessions))

        for row_idx, session_info in enumerate(sessions):
            if session_info.session_id == dataset.session_info.session_id:
                ds = dataset
            else:
                try:
                    ds = load_session(dataset.patient_id, session_info)
                except Exception as exc:  # pragma: no cover - defensive
                    warning = QTableWidgetItem(f"Error: {exc}")
                    warning.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    for col in range(self.compare_table.columnCount()):
                        self.compare_table.setItem(row_idx, col, QTableWidgetItem("--"))
                    self.compare_table.setItem(row_idx, 0, warning)
                    continue

            metrics = ds.compute_basic_metrics()
            fatigue = ds.compute_fatigue_metrics(0)
            date = str(session_info.metadata.get("date", ""))
            time_start = str(session_info.metadata.get("time_start", ""))
            date_display = f"{date} {time_start}".strip() or "--"

            cells = [
                session_info.label,
                date_display,
                f"{metrics.get('rom_total', float('nan')):.1f}" if metrics.get("rom_total") is not None else "--",
                f"{metrics.get('duration_sec', float('nan')):.1f}" if metrics.get("duration_sec") is not None else "--",
                f"{int(metrics.get('event_count', 0))}" if metrics.get("event_count") is not None else "--",
                f"{metrics.get('rms_ch0_peak', 0.0) * 1000:.2f}" if metrics.get("rms_ch0_peak") is not None else "--",
                f"{metrics.get('rms_ch1_peak', 0.0) * 1000:.2f}" if metrics.get("rms_ch1_peak") is not None else "--",
                f"{fatigue.get('rms_decline_pct', float('nan')):.1f}" if fatigue.get("rms_decline_pct") is not None else "--",
            ]

            for col_idx, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if session_info.session_id == dataset.session_info.session_id:
                    item.setBackground(QtGui.QColor(255, 255, 180, 60))
                self.compare_table.setItem(row_idx, col_idx, item)

        self.compare_table.resizeColumnsToContents()
        self.compare_table.resizeRowsToContents()

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------
    def _current_dataset_or_warn(self, title: str) -> Optional[SessionDataset]:
        if not self._current_dataset:
            QMessageBox.information(self, title, "Selecciona una sesión primero.")
            return None
        return self._current_dataset

    def _open_external_session(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de sesión", str(cfg.PROJECT_ROOT))
        if not directory:
            return
        path = Path(directory)
        if not (path / "metadata.json").exists():
            QMessageBox.warning(self, "Sesión inválida", "La carpeta seleccionada no contiene metadatos de sesión.")
            return
        metadata = load_json(str(path / "metadata.json"))
        session_info = SessionInfo(session_id=path.name, path=path, metadata=metadata or {})
        patient_id = path.parent.parent.name if path.parent.name == "sessions" else "externo"
        dataset = load_session(patient_id, session_info)
        self._current_dataset = dataset
        self._render_session(dataset)
        self.status_label.setText(f"Sesión externa: {dataset.session_info.session_id}")

    def _export_filtered_data(self, fmt: str) -> None:
        dataset = self._current_dataset_or_warn("Exportar")
        if not dataset:
            return
        fmt = fmt.lower()
        if fmt == "csv":
            self._export_filtered_data_csv(dataset)
        elif fmt == "mat":
            self._export_filtered_data_mat(dataset)
        elif fmt == "edf":
            self._export_filtered_data_edf(dataset)
        else:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Exportar", f"Formato no soportado: {fmt}")

    def _export_filtered_data_csv(self, dataset: SessionDataset) -> None:
        default_path = cfg.PROJECT_ROOT / f"{dataset.session_info.session_id}_filtrado.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar datos filtrados (CSV)",
            str(default_path),
            "CSV (*.csv)",
        )
        if not save_path:
            return

        def arr_to_list(array: np.ndarray) -> List[float]:
            if array.size == 0:
                return []
            return array.astype(float).tolist()

        t_emg = dataset.time_axis("emg")
        filtered0 = dataset.emg_channel(0, "filtered")
        filtered1 = dataset.emg_channel(1, "filtered")
        rms0 = dataset.emg_channel(0, "rms")
        rms1 = dataset.emg_channel(1, "rms")
        t_imu = dataset.time_axis("imu")
        angle = dataset.angle_series()
        accel_x = dataset.imu_series("accel_x")
        accel_y = dataset.imu_series("accel_y")
        accel_z = dataset.imu_series("accel_z")
        gyro_x = dataset.imu_series("gyro_x")
        gyro_y = dataset.imu_series("gyro_y")
        gyro_z = dataset.imu_series("gyro_z")
        velocity = dataset.derived_series("velocity")

        columns = [
            ("time_emg", arr_to_list(t_emg)),
            ("emg_filtered_ch0", arr_to_list(filtered0)),
            ("emg_filtered_ch1", arr_to_list(filtered1)),
            ("emg_rms_ch0", arr_to_list(rms0)),
            ("emg_rms_ch1", arr_to_list(rms1)),
            ("time_imu", arr_to_list(t_imu)),
            ("angle_deg", arr_to_list(angle)),
            ("accel_x_g", arr_to_list(accel_x)),
            ("accel_y_g", arr_to_list(accel_y)),
            ("accel_z_g", arr_to_list(accel_z)),
            ("gyro_x_dps", arr_to_list(gyro_x)),
            ("gyro_y_dps", arr_to_list(gyro_y)),
            ("gyro_z_dps", arr_to_list(gyro_z)),
            ("angular_velocity_dps", arr_to_list(velocity)),
        ]

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([name for name, _ in columns])
                rows = zip_longest(*(values for _, values in columns), fillvalue="")
                writer.writerows(rows)
        except OSError as exc:
            QMessageBox.critical(self, "Exportar", f"No se pudo guardar el archivo: {exc}")
            return

        QMessageBox.information(self, "Exportar", "Datos filtrados exportados en CSV.")

    def _export_filtered_data_mat(self, dataset: SessionDataset) -> None:
        try:
            from scipy import io as spio  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - dependency optional
            QMessageBox.warning(self, "Exportar", "Instala scipy para exportar a MAT.")
            return

        default_path = cfg.PROJECT_ROOT / f"{dataset.session_info.session_id}_filtrado.mat"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar datos filtrados (MAT)",
            str(default_path),
            "MAT (*.mat)",
        )
        if not save_path:
            return

        payload = {
            "time_emg": dataset.time_axis("emg"),
            "emg_filtered_ch0": dataset.emg_channel(0, "filtered"),
            "emg_filtered_ch1": dataset.emg_channel(1, "filtered"),
            "emg_rms_ch0": dataset.emg_channel(0, "rms"),
            "emg_rms_ch1": dataset.emg_channel(1, "rms"),
            "time_imu": dataset.time_axis("imu"),
            "angle_deg": dataset.angle_series(),
            "accel_x_g": dataset.imu_series("accel_x"),
            "accel_y_g": dataset.imu_series("accel_y"),
            "accel_z_g": dataset.imu_series("accel_z"),
            "gyro_x_dps": dataset.imu_series("gyro_x"),
            "gyro_y_dps": dataset.imu_series("gyro_y"),
            "gyro_z_dps": dataset.imu_series("gyro_z"),
            "angular_velocity_dps": dataset.derived_series("velocity"),
            "metadata_json": np.array([json.dumps(dataset.metadata, ensure_ascii=False)], dtype=object),
        }

        try:
            spio.savemat(save_path, payload)
        except Exception as exc:  # pragma: no cover - scipy errors
            QMessageBox.critical(self, "Exportar", f"No se pudo exportar archivo MAT: {exc}")
            return

        QMessageBox.information(self, "Exportar", "Datos filtrados exportados en MAT.")

    def _export_filtered_data_edf(self, dataset: SessionDataset) -> None:
        try:
            from pyedflib import highlevel  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - dependency optional
            QMessageBox.warning(self, "Exportar", "Instala pyedflib para exportar a EDF+.")
            return

        t_emg = dataset.time_axis("emg")
        ch0 = dataset.emg_channel(0, "filtered")
        ch1 = dataset.emg_channel(1, "filtered")
        limit = min(t_emg.size, ch0.size, ch1.size)
        if limit < 10:
            QMessageBox.warning(self, "Exportar", "No hay suficientes muestras EMG para exportar a EDF+.")
            return

        dt = float(np.mean(np.diff(t_emg[:limit]))) if limit > 1 else 0.0
        if dt <= 0:
            QMessageBox.warning(self, "Exportar", "No se pudo determinar la frecuencia de muestreo.")
            return
        sample_rate = int(round(1.0 / dt))
        if sample_rate <= 0:
            QMessageBox.warning(self, "Exportar", "Frecuencia de muestreo inválida para EDF+.")
            return

        default_path = cfg.PROJECT_ROOT / f"{dataset.session_info.session_id}.edf"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar datos filtrados (EDF+)",
            str(default_path),
            "EDF (*.edf)",
        )
        if not save_path:
            return

        data_arrays = [
            (ch0[:limit] * 1e3).astype(np.float64),
            (ch1[:limit] * 1e3).astype(np.float64),
        ]
        labels = ["EMG_C0", "EMG_C1"]
        signal_headers = highlevel.make_signal_headers(labels)
        for header, data in zip(signal_headers, data_arrays):
            peak = float(np.max(np.abs(data))) if data.size else 1.0
            peak = max(peak, 1.0)
            header["sample_rate"] = sample_rate
            header["physical_min"] = -peak
            header["physical_max"] = peak
            header["digital_min"] = -32768
            header["digital_max"] = 32767
            header["dimension"] = "mV"
            header["prefilter"] = "Filtered 20-500Hz"

        edf_header = {
            "patientname": dataset.patient_id,
            "recording_additional": dataset.session_info.session_id,
        }

        try:
            highlevel.write_edf(save_path, data_arrays, signal_headers, edf_header, file_type=highlevel.FILETYPE_EDFPLUS)
        except Exception as exc:  # pragma: no cover - pyedflib errors
            QMessageBox.critical(self, "Exportar", f"No se pudo exportar EDF+: {exc}")
            return

        QMessageBox.information(self, "Exportar", "Datos filtrados exportados en EDF+.")

    def _export_graphics(self, fmt: str) -> None:
        dataset = self._current_dataset_or_warn("Exportar gráficos")
        if not dataset:
            return
        fmt = fmt.lower()
        directory = QFileDialog.getExistingDirectory(self, "Selecciona carpeta destino", str(cfg.PROJECT_ROOT))
        if not directory:
            return

        saved = 0
        for key in self._plot_order:
            widget = self.plots[key].widget
            filename = Path(directory) / f"{dataset.session_info.session_id}_{key}.{fmt}"
            try:
                if fmt == "png":
                    pixmap = widget.grab()
                    pixmap.save(str(filename), "PNG")
                elif fmt == "svg":
                    if SVGExporter is None:
                        QMessageBox.warning(self, "Exportar", "Instala pyqtgraph[svg] para exportar SVG.")
                        return
                    exporter = SVGExporter(widget.plotItem)
                    exporter.export(str(filename))
                elif fmt == "pdf":
                    printer = QPrinter(QPrinter.OutputFormat.PdfFormat)
                    printer.setOutputFileName(str(filename))
                    printer.setPageOrientation(QtGui.QPageLayout.Orientation.Landscape)
                    pixmap = widget.grab()
                    painter = QtGui.QPainter(printer)
                    rect = painter.viewport()
                    scaled = pixmap.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    painter.drawPixmap(0, 0, scaled)
                    painter.end()
                else:  # pragma: no cover - defensive
                    QMessageBox.warning(self, "Exportar", f"Formato no soportado: {fmt}")
                    return
            except Exception as exc:  # pragma: no cover - export failures
                QMessageBox.critical(self, "Exportar", f"No se pudo exportar {filename.name}: {exc}")
                return
            saved += 1

        if saved:
            QMessageBox.information(self, "Exportar", f"Se exportaron {saved} gráficos en formato {fmt.upper()}.")

    def _export_metrics(self, fmt: str) -> None:
        dataset = self._current_dataset_or_warn("Exportar métricas")
        if not dataset:
            return
        metrics = dataset.compute_basic_metrics()
        fmt = fmt.lower()
        if fmt == "json":
            default_path = cfg.PROJECT_ROOT / "metrics.json"
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Exportar métricas (JSON)",
                str(default_path),
                "JSON (*.json)",
            )
            if not save_path:
                return
            payload = {
                "patient": dataset.patient_id,
                "session": dataset.session_info.session_id,
                "metadata": dataset.metadata,
                "metrics": metrics,
            }
            try:
                Path(save_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            except OSError as exc:
                QMessageBox.critical(self, "Exportar", f"No se pudo guardar el archivo: {exc}")
                return
            QMessageBox.information(self, "Exportar", "Métricas exportadas en JSON.")
            return

        if fmt == "xlsx":
            try:
                from openpyxl import Workbook  # type: ignore
            except ImportError:  # pragma: no cover - dependency optional
                QMessageBox.warning(self, "Exportar", "Instala openpyxl para exportar a XLSX.")
                return
            default_path = cfg.PROJECT_ROOT / "metrics.xlsx"
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Exportar métricas (XLSX)",
                str(default_path),
                "XLSX (*.xlsx)",
            )
            if not save_path:
                return
            wb = Workbook()
            ws = wb.active
            ws.title = "Métricas"
            ws.append(["Métrica", "Valor"])
            for key, value in metrics.items():
                ws.append([key, value])
            try:
                wb.save(save_path)
            except Exception as exc:  # pragma: no cover - xlsx errors
                QMessageBox.critical(self, "Exportar", f"No se pudo guardar XLSX: {exc}")
                return
            QMessageBox.information(self, "Exportar", "Métricas exportadas en XLSX.")
            return

        QMessageBox.warning(self, "Exportar", f"Formato no soportado: {fmt}")

    def _generate_report(self) -> None:
        dataset = self._current_dataset_or_warn("Generar reporte")
        if not dataset:
            return
        default_path = cfg.PROJECT_ROOT / f"reporte_{dataset.session_info.session_id}.pdf"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Generar reporte clínico",
            str(default_path),
            "PDF (*.pdf)",
        )
        if not save_path:
            return

        metrics = dataset.compute_basic_metrics()
        fatigue = dataset.compute_fatigue_metrics(self._fatigue_channel)
        spectral_channel = self._spectral_channel
        spectral_mode = self._spectral_signal

        summary_rows = "".join(
            f"<tr><td>{key}</td><td>{value:.3f}</td></tr>"
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        )

        fatigue_rows = "".join(
            f"<tr><td>{key}</td><td>{value:.3f}</td></tr>"
            for key, value in fatigue.items()
            if isinstance(value, (int, float))
        )

        html = f"""
        <h1>Reporte de análisis de rodilla</h1>
        <p><b>Paciente:</b> {dataset.patient_id}<br>
           <b>Sesión:</b> {dataset.session_info.session_id}<br>
           <b>Fecha:</b> {dataset.metadata.get('date', '--')} {dataset.metadata.get('time_start', '')}</p>
        <h2>Métricas globales</h2>
        <table border='1' cellspacing='0' cellpadding='4'>
            <tr><th>Métrica</th><th>Valor</th></tr>
            {summary_rows}
        </table>
        <h2>Fatiga (Canal {self._fatigue_channel})</h2>
        <table border='1' cellspacing='0' cellpadding='4'>
            <tr><th>Indicador</th><th>Valor</th></tr>
            {fatigue_rows}
        </table>
        <h2>Configuración actual</h2>
        <ul>
            <li>Modo espectral: {spectral_mode}</li>
            <li>Canal espectral: {spectral_channel}</li>
            <li>Eventos registrados: {len(dataset.events)}</li>
        </ul>
        """

        document = QtGui.QTextDocument()
        document.setDefaultStyleSheet("table { border-collapse: collapse; } th { background:#222; color:#fff; }")
        document.setHtml(html)

        writer = QtGui.QPdfWriter(save_path)
        writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4))
        writer.setPageOrientation(QtGui.QPageLayout.Orientation.Portrait)
        document.print(writer)

        QMessageBox.information(self, "Reporte", "Reporte clínico generado correctamente.")

    def _open_analysis_config(self) -> None:
        dialog = AnalysisConfigDialog(self, self._analysis_config)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._analysis_config.update(dialog.values())
            if self._current_dataset:
                self._populate_fatigue_tab(self._current_dataset)
                self._populate_spectral_tab(self._current_dataset)

    def _export_events_csv(self) -> None:
        if not self._current_dataset:
            QMessageBox.information(self, "Eventos", "Selecciona una sesión.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar eventos",
            str(cfg.PROJECT_ROOT / "events.csv"),
            "CSV (*.csv)",
        )
        if not save_path:
            return
        with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time_sec", "type", "description"])
            for event in self._current_dataset.events:
                writer.writerow([
                    event.get("timestamp_relative_sec", ""),
                    event.get("type", ""),
                    event.get("description", ""),
                ])
        QMessageBox.information(self, "Eventos", "Eventos exportados correctamente.")


__all__ = ["SessionAnalysisWindow"]
