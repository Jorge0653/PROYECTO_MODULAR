"""
Módulo de análisis en tiempo real: EMG + IMU + Ángulo de flexión
"""
import numpy as np
from typing import Optional, Dict
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt6.QtCore import QTimer, QTime
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QComboBox, QMessageBox,
                              QGroupBox, QFrame, QDialog, QSpinBox, QRadioButton,
                              QButtonGroup)
from PyQt6.QtGui import QFont

from core import SerialReaderThread, get_available_ports, EMGProcessor, AngleCalculator
from config import settings as cfg
from utils import save_json, load_json


class CalibrationDialog(QDialog):
    """Diálogo para calibración del IMU."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibración IMU")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        # Resultados
        self.calibration_mode = cfg.CALIBRATION_POINTS  # 1 o 2 puntos
        self.angle_raw_point1 = None
        self.angle_ref_point1 = 0.0
        self.angle_raw_point2 = None
        self.angle_ref_point2 = 90.0
        self.calibration_done = False
        
        self._create_ui()
    
    def _create_ui(self):
        """Crea la interfaz del diálogo."""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("🎯 Calibración del Ángulo de Rodilla")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; padding: 10px;")
        layout.addWidget(title)
        
        # Instrucciones
        instructions = QLabel(
            "La calibración permite mapear las lecturas del IMU a ángulos reales de flexión.\n"
            "Puedes elegir calibración de 1 o 2 puntos:"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 10px; color: #34495e;")
        layout.addWidget(instructions)
        
        # Selección de modo
        mode_group = QGroupBox("Modo de Calibración")
        mode_layout = QVBoxLayout()
        
        self.radio_1point = QRadioButton("1 Punto (solo offset - pierna estirada = 0°)")
        self.radio_2point = QRadioButton("2 Puntos (offset + escala - mayor precisión)")
        
        if cfg.CALIBRATION_POINTS == 1:
            self.radio_1point.setChecked(True)
        else:
            self.radio_2point.setChecked(True)
        
        mode_layout.addWidget(self.radio_1point)
        mode_layout.addWidget(self.radio_2point)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Configuración Punto 1
        point1_group = QGroupBox("📍 Punto 1: Pierna Estirada")
        point1_layout = QVBoxLayout()
        
        p1_inst = QLabel("1. Extiende completamente la rodilla\n2. Haz clic en 'Capturar Punto 1'")
        p1_inst.setStyleSheet("color: #555;")
        point1_layout.addWidget(p1_inst)
        
        p1_angle_layout = QHBoxLayout()
        p1_angle_layout.addWidget(QLabel("Ángulo de referencia (°):"))
        self.spin_point1 = QSpinBox()
        self.spin_point1.setRange(-30, 30)
        self.spin_point1.setValue(0)
        self.spin_point1.setToolTip("Típicamente 0° = pierna completamente estirada")
        p1_angle_layout.addWidget(self.spin_point1)
        p1_angle_layout.addStretch()
        point1_layout.addLayout(p1_angle_layout)
        
        self.btn_capture_p1 = QPushButton("Capturar Punto 1")
        self.btn_capture_p1.setStyleSheet("padding: 8px; background-color: #3498db; color: white; font-weight: bold;")
        self.btn_capture_p1.clicked.connect(self._capture_point1)
        point1_layout.addWidget(self.btn_capture_p1)
        
        self.label_p1_status = QLabel("❌ No capturado")
        self.label_p1_status.setStyleSheet("color: #e74c3c; padding: 5px;")
        point1_layout.addWidget(self.label_p1_status)
        
        point1_group.setLayout(point1_layout)
        layout.addWidget(point1_group)
        
        # Configuración Punto 2 (solo si es 2 puntos)
        self.point2_group = QGroupBox("📍 Punto 2: Rodilla Flexionada")
        point2_layout = QVBoxLayout()
        
        p2_inst = QLabel("1. Flexiona la rodilla a un ángulo conocido (ej. 90°)\n2. Haz clic en 'Capturar Punto 2'")
        p2_inst.setStyleSheet("color: #555;")
        point2_layout.addWidget(p2_inst)
        
        p2_angle_layout = QHBoxLayout()
        p2_angle_layout.addWidget(QLabel("Ángulo de referencia (°):"))
        self.spin_point2 = QSpinBox()
        self.spin_point2.setRange(30, 150)
        self.spin_point2.setValue(90)
        self.spin_point2.setToolTip("Ángulo real de flexión (usa goniómetro manual si es posible)")
        p2_angle_layout.addWidget(self.spin_point2)
        p2_angle_layout.addStretch()
        point2_layout.addLayout(p2_angle_layout)
        
        self.btn_capture_p2 = QPushButton("Capturar Punto 2")
        self.btn_capture_p2.setStyleSheet("padding: 8px; background-color: #3498db; color: white; font-weight: bold;")
        self.btn_capture_p2.clicked.connect(self._capture_point2)
        self.btn_capture_p2.setEnabled(False)
        point2_layout.addWidget(self.btn_capture_p2)
        
        self.label_p2_status = QLabel("❌ No capturado")
        self.label_p2_status.setStyleSheet("color: #e74c3c; padding: 5px;")
        point2_layout.addWidget(self.label_p2_status)
        
        self.point2_group.setLayout(point2_layout)
        layout.addWidget(self.point2_group)
        
        # Botones
        btn_layout = QHBoxLayout()
        
        self.btn_finish = QPushButton("✓ Finalizar Calibración")
        self.btn_finish.setStyleSheet("padding: 10px; background-color: #27ae60; color: white; font-weight: bold;")
        self.btn_finish.clicked.connect(self._finish_calibration)
        self.btn_finish.setEnabled(False)
        btn_layout.addWidget(self.btn_finish)
        
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("padding: 10px;")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        
        # Conectar cambio de modo
        self.radio_1point.toggled.connect(self._on_mode_changed)
        self.radio_2point.toggled.connect(self._on_mode_changed)
        self._on_mode_changed()  # Aplicar estado inicial
    
    def _on_mode_changed(self):
        """Actualiza UI según el modo seleccionado."""
        if self.radio_1point.isChecked():
            self.calibration_mode = 1
            self.point2_group.setEnabled(False)
        else:
            self.calibration_mode = 2
            self.point2_group.setEnabled(True)
        
        # Resetear estados
        self.angle_raw_point1 = None
        self.angle_raw_point2 = None
        self.label_p1_status.setText("❌ No capturado")
        self.label_p1_status.setStyleSheet("color: #e74c3c; padding: 5px;")
        self.label_p2_status.setText("❌ No capturado")
        self.label_p2_status.setStyleSheet("color: #e74c3c; padding: 5px;")
        self.btn_capture_p2.setEnabled(False)
        self.btn_finish.setEnabled(False)
    
    def set_current_angle(self, angle: float):
        """Actualiza el ángulo crudo actual (llamado desde la ventana principal)."""
        self.current_raw_angle = angle
    
    def _capture_point1(self):
        """Captura el punto 1."""
        if not hasattr(self, 'current_raw_angle'):
            QMessageBox.warning(self, "Error", "No hay datos del IMU. Asegúrate de que esté conectado.")
            return
        
        self.angle_raw_point1 = self.current_raw_angle
        self.angle_ref_point1 = self.spin_point1.value()
        
        self.label_p1_status.setText(f"✓ Capturado: {self.angle_raw_point1:.2f}° crudo → {self.angle_ref_point1}° real")
        self.label_p1_status.setStyleSheet("color: #27ae60; padding: 5px;")
        
        if self.calibration_mode == 1:
            self.btn_finish.setEnabled(True)
        else:
            self.btn_capture_p2.setEnabled(True)
    
    def _capture_point2(self):
        """Captura el punto 2."""
        if not hasattr(self, 'current_raw_angle'):
            QMessageBox.warning(self, "Error", "No hay datos del IMU.")
            return
        
        self.angle_raw_point2 = self.current_raw_angle
        self.angle_ref_point2 = self.spin_point2.value()
        
        self.label_p2_status.setText(f"✓ Capturado: {self.angle_raw_point2:.2f}° crudo → {self.angle_ref_point2}° real")
        self.label_p2_status.setStyleSheet("color: #27ae60; padding: 5px;")
        
        self.btn_finish.setEnabled(True)
    
    def _finish_calibration(self):
        """Finaliza la calibración."""
        if self.calibration_mode == 1:
            if self.angle_raw_point1 is None:
                QMessageBox.warning(self, "Error", "Debes capturar el Punto 1.")
                return
        else:
            if self.angle_raw_point1 is None or self.angle_raw_point2 is None:
                QMessageBox.warning(self, "Error", "Debes capturar ambos puntos.")
                return
        
        self.calibration_done = True
        self.accept()
    
    def get_calibration_data(self) -> Dict:
        """Retorna los datos de calibración."""
        return {
            'mode': self.calibration_mode,
            'angle_raw_point1': self.angle_raw_point1,
            'angle_ref_point1': self.angle_ref_point1,
            'angle_raw_point2': self.angle_raw_point2,
            'angle_ref_point2': self.angle_ref_point2
        }


class RealtimeAnalysisWindow(QMainWindow):
    """Ventana de análisis en tiempo real."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis en Tiempo Real - EMG + IMU")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Estados
        self.serial_thread: Optional[SerialReaderThread] = None
        self.is_connected = False
        
        # Procesadores
        self.emg_ch0_processor = EMGProcessor()
        self.emg_ch1_processor = EMGProcessor()
        self.angle_calculator = AngleCalculator()
        
        # Buffers (arrays numpy pre-alocados)
        emg_buffer = cfg.EMG_BUFFER_SIZE
        imu_buffer = cfg.IMU_BUFFER_SIZE
        self.time_emg_ch0 = np.zeros(emg_buffer, dtype=np.float64)
        self.data_emg_ch0 = np.zeros(emg_buffer, dtype=np.float64)
        self.data_rms_ch0 = np.zeros(emg_buffer, dtype=np.float64)
        
        self.time_emg_ch1 = np.zeros(emg_buffer, dtype=np.float64)
        self.data_emg_ch1 = np.zeros(emg_buffer, dtype=np.float64)
        self.data_rms_ch1 = np.zeros(emg_buffer, dtype=np.float64)
        
        self.time_imu = np.zeros(imu_buffer, dtype=np.float64)
        self.data_angle = np.zeros(imu_buffer, dtype=np.float64)
        
        # Índices circulares
        self.idx_emg_ch0 = 0
        self.idx_emg_ch1 = 0
        self.idx_imu = 0
        
        # Tiempo de referencia
        self.t0_emg: Optional[float] = None
        self.t0_imu: Optional[float] = None
        
        # Tiempo actual
        self.current_time_emg = 0.0
        self.current_time_imu = 0.0
        
        # Ángulo crudo actual (para calibración)
        self.current_raw_angle = 0.0
        
        # Estadísticas
        self.emg_count = 0
        self.imu_count = 0
        self.last_stats_update = QTime.currentTime()
        
        # UI
        self._create_ui()
        
        # Timer de actualización
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plots)
        self.update_timer.start(cfg.UPDATE_INTERVAL_MS)
    
    def _create_ui(self):
        """Crea la interfaz."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # ========== TOOLBAR ==========
        self._create_toolbar(main_layout)
        
        # ========== CONFIGURACIÓN DE PYQTGRAPH ==========
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', False)
        pg.setConfigOption('useOpenGL', False)
        
        # ========== GRÁFICAS ==========
        
        # Canal EMG 0
        self.plot_ch0 = pg.PlotWidget(title='Canal EMG 0 (AIN0)')
        self.plot_ch0.setLabel('left', 'Voltaje', units='V')
        self.plot_ch0.setLabel('bottom', 'Tiempo', units='s')
        self.plot_ch0.showGrid(x=True, y=True, alpha=0.3)
        self.plot_ch0.setYRange(-0.5, 0.5, padding=0.05)
        self.plot_ch0.addLegend()
        
        self.curve_ch0 = self.plot_ch0.plot(pen=pg.mkPen(color=cfg.COLOR_CH0, width=1.5), name='EMG CH0')
        self.curve_rms_ch0 = self.plot_ch0.plot(pen=pg.mkPen(color=cfg.COLOR_RMS_CH0, width=2), name='RMS CH0')
        
        main_layout.addWidget(self.plot_ch0)
        
        # Canal EMG 1
        self.plot_ch1 = pg.PlotWidget(title='Canal EMG 1 (AIN1)')
        self.plot_ch1.setLabel('left', 'Voltaje', units='V')
        self.plot_ch1.setLabel('bottom', 'Tiempo', units='s')
        self.plot_ch1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_ch1.setYRange(-0.5, 0.5, padding=0.05)
        self.plot_ch1.addLegend()
        
        self.curve_ch1 = self.plot_ch1.plot(pen=pg.mkPen(color=cfg.COLOR_CH1, width=1.5), name='EMG CH1')
        self.curve_rms_ch1 = self.plot_ch1.plot(pen=pg.mkPen(color=cfg.COLOR_RMS_CH1, width=2), name='RMS CH1')
        
        main_layout.addWidget(self.plot_ch1)
        
        # Ángulo de Flexión
        self.plot_angle = pg.PlotWidget(title='Ángulo de Flexión de Rodilla')
        self.plot_angle.setLabel('left', 'Ángulo', units='°')
        self.plot_angle.setLabel('bottom', 'Tiempo', units='s')
        self.plot_angle.showGrid(x=True, y=True, alpha=0.3)
        self.plot_angle.setYRange(-10, 140, padding=0.05)
        
        self.curve_angle = self.plot_angle.plot(pen=pg.mkPen(color=cfg.COLOR_ANGLE, width=2))
        
        main_layout.addWidget(self.plot_angle)
        
        # ========== PANEL DE MÉTRICAS ==========
        metrics_layout = QHBoxLayout()
        
        # RMS CH0
        self.label_rms_ch0 = self._create_metric_label("RMS CH0: 0.000 mV", cfg.COLOR_RMS_CH0)
        metrics_layout.addWidget(self.label_rms_ch0)
        
        # RMS CH1
        self.label_rms_ch1 = self._create_metric_label("RMS CH1: 0.000 mV", cfg.COLOR_RMS_CH1)
        metrics_layout.addWidget(self.label_rms_ch1)
        
        # Ángulo actual
        self.label_angle = self._create_metric_label("Ángulo: 0.0°", cfg.COLOR_ANGLE)
        metrics_layout.addWidget(self.label_angle)
        
        main_layout.addLayout(metrics_layout)
        
        # ========== BARRA DE ESTADO ==========
        self.status_label = QLabel("⚫ Desconectado")
        self.stats_label = QLabel("EMG: 0 sps | IMU: 0 sps")
        self.statusBar().addWidget(self.status_label)
        self.statusBar().addPermanentWidget(self.stats_label)
    
    def _create_toolbar(self, parent_layout):
        """Crea el toolbar de control."""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(10, 10, 10, 10)
        
        # Puerto serial
        toolbar_layout.addWidget(QLabel("Puerto:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(200)
        self._refresh_ports()
        toolbar_layout.addWidget(self.port_combo)
        
        # Botón actualizar puertos
        btn_refresh = QPushButton("🔄")
        btn_refresh.setToolTip("Actualizar lista de puertos")
        btn_refresh.clicked.connect(self._refresh_ports)
        toolbar_layout.addWidget(btn_refresh)
        
        # Botón conectar/desconectar
        self.btn_connect = QPushButton("Conectar")
        self.btn_connect.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_connect.clicked.connect(self._toggle_connection)
        toolbar_layout.addWidget(self.btn_connect)
        
        toolbar_layout.addSpacing(20)
        
        # Botón calibrar
        self.btn_calibrate = QPushButton("🎯 Calibrar IMU")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setStyleSheet("padding: 8px;")
        self.btn_calibrate.clicked.connect(self._open_calibration)
        toolbar_layout.addWidget(self.btn_calibrate)
        
        # Botón limpiar
        btn_clear = QPushButton("Limpiar Gráficas")
        btn_clear.setStyleSheet("padding: 8px;")
        btn_clear.clicked.connect(self._clear_buffers)
        toolbar_layout.addWidget(btn_clear)
        
        toolbar_layout.addStretch()
        
        parent_layout.addWidget(toolbar_widget)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        parent_layout.addWidget(line)
    
    def _create_metric_label(self, text: str, color: tuple) -> QLabel:
        """Crea un label para métricas."""
        label = QLabel(text)
        label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        label.setStyleSheet(f"color: rgb{color}; padding: 10px; border: 2px solid rgb{color}; border-radius: 5px;")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return label
    
    def _refresh_ports(self):
        """Actualiza la lista de puertos."""
        self.port_combo.clear()
        ports = get_available_ports()
        
        for device, description in ports:
            self.port_combo.addItem(description, device)
        
        if len(ports) == 0:
            self.port_combo.addItem("No hay puertos disponibles", None)
    
    def _toggle_connection(self):
        """Conecta o desconecta del puerto serial."""
        if self.is_connected:
            # Desconectar
            if self.serial_thread:
                self.serial_thread.stop()
                self.serial_thread.wait()
                self.serial_thread = None
            
            self.is_connected = False
            self.btn_connect.setText("Conectar")
            self.btn_connect.setStyleSheet("font-weight: bold; padding: 8px;")
            self.port_combo.setEnabled(True)
            self.btn_calibrate.setEnabled(False)
        else:
            # Conectar
            port = self.port_combo.currentData()
            if port is None:
                QMessageBox.warning(self, "Error", "No hay puerto serial seleccionado")
                return
            
            self._clear_buffers()
            
            self.serial_thread = SerialReaderThread(port)
            self.serial_thread.frame_received.connect(self._on_frame_received)
            self.serial_thread.connection_status.connect(self._on_connection_status)
            self.serial_thread.start()
    
    def _on_connection_status(self, connected: bool, message: str):
        """Maneja cambios en el estado de conexión."""
        self.status_label.setText(message)
        
        if connected:
            self.is_connected = True
            self.btn_connect.setText("Desconectar")
            self.btn_connect.setStyleSheet("font-weight: bold; padding: 8px; background-color: #e74c3c; color: white;")
            self.port_combo.setEnabled(False)
            self.btn_calibrate.setEnabled(True)
        else:
            self.is_connected = False
            self.btn_connect.setText("Conectar")
            self.btn_connect.setStyleSheet("font-weight: bold; padding: 8px;")
            self.port_combo.setEnabled(True)
            self.btn_calibrate.setEnabled(False)
    
    def _on_frame_received(self, frame: Dict):
        """Procesa un frame recibido."""
        if frame['type'] == 'EMG':
            if self.t0_emg is None:
                self.t0_emg = frame['timestamp_us']
            
            t_sec = (frame['timestamp_us'] - self.t0_emg) / 1e6
            self.current_time_emg = t_sec
            
            # Procesar Canal 0
            filtered_ch0, rms_ch0 = self.emg_ch0_processor.process_sample(frame['ch0'])
            self.time_emg_ch0[self.idx_emg_ch0] = t_sec
            self.data_emg_ch0[self.idx_emg_ch0] = filtered_ch0
            self.data_rms_ch0[self.idx_emg_ch0] = rms_ch0
            self.idx_emg_ch0 = (self.idx_emg_ch0 + 1) % cfg.EMG_BUFFER_SIZE
            
            # Procesar Canal 1
            filtered_ch1, rms_ch1 = self.emg_ch1_processor.process_sample(frame['ch1'])
            self.time_emg_ch1[self.idx_emg_ch1] = t_sec
            self.data_emg_ch1[self.idx_emg_ch1] = filtered_ch1
            self.data_rms_ch1[self.idx_emg_ch1] = rms_ch1
            self.idx_emg_ch1 = (self.idx_emg_ch1 + 1) % cfg.EMG_BUFFER_SIZE
            
            self.emg_count += 1
        
        elif frame['type'] == 'IMU':
            if self.t0_imu is None:
                self.t0_imu = frame['timestamp_us']
            
            t_sec = (frame['timestamp_us'] - self.t0_imu) / 1e6
            self.current_time_imu = t_sec
            
            # Calcular ángulo
            angle = self.angle_calculator.update(
                frame['ax'], frame['ay'], frame['az'],
                frame['gx'], frame['gy'], frame['gz'],
                frame['timestamp_us']
            )
            
            self.current_raw_angle = angle
            
            self.time_imu[self.idx_imu] = t_sec
            self.data_angle[self.idx_imu] = angle
            self.idx_imu = (self.idx_imu + 1) % cfg.IMU_BUFFER_SIZE
            
            self.imu_count += 1
        
        # Actualizar estadísticas cada segundo
        current_time = QTime.currentTime()
        elapsed = self.last_stats_update.msecsTo(current_time)
        
        if elapsed >= 1000:
            emg_rate = self.emg_count / (elapsed / 1000.0)
            imu_rate = self.imu_count / (elapsed / 1000.0)
            self.stats_label.setText(f"EMG: {emg_rate:.1f} sps | IMU: {imu_rate:.1f} sps")
            
            self.emg_count = 0
            self.imu_count = 0
            self.last_stats_update = current_time
    
    def _update_plots(self):
        """Actualiza las gráficas."""
        # ===== EMG CH0 =====
        if self.idx_emg_ch0 > 0:
            x_max = self.current_time_emg
            x_min = max(0, x_max - cfg.WINDOW_TIME_SEC)
            
            t_data = np.roll(self.time_emg_ch0, -self.idx_emg_ch0)
            y_data = np.roll(self.data_emg_ch0, -self.idx_emg_ch0)
            rms_data = np.roll(self.data_rms_ch0, -self.idx_emg_ch0)
            
            mask = (t_data >= x_min) & (t_data <= x_max) & (t_data > 0)
            
            self.curve_ch0.setData(t_data[mask], y_data[mask])
            self.curve_rms_ch0.setData(t_data[mask], rms_data[mask])
            self.plot_ch0.setXRange(x_min, x_max, padding=0)
            
            # Actualizar métrica RMS
            if np.any(mask):
                current_rms = rms_data[mask][-1] if mask.sum() > 0 else 0.0
                self.label_rms_ch0.setText(f"RMS CH0: {current_rms * 1000:.3f} mV")
        
        # ===== EMG CH1 =====
        if self.idx_emg_ch1 > 0:
            x_max = self.current_time_emg
            x_min = max(0, x_max - cfg.WINDOW_TIME_SEC)
            
            t_data = np.roll(self.time_emg_ch1, -self.idx_emg_ch1)
            y_data = np.roll(self.data_emg_ch1, -self.idx_emg_ch1)
            rms_data = np.roll(self.data_rms_ch1, -self.idx_emg_ch1)
            
            mask = (t_data >= x_min) & (t_data <= x_max) & (t_data > 0)
            
            self.curve_ch1.setData(t_data[mask], y_data[mask])
            self.curve_rms_ch1.setData(t_data[mask], rms_data[mask])
            self.plot_ch1.setXRange(x_min, x_max, padding=0)
            
            # Actualizar métrica RMS
            if np.any(mask):
                current_rms = rms_data[mask][-1] if mask.sum() > 0 else 0.0
                self.label_rms_ch1.setText(f"RMS CH1: {current_rms * 1000:.3f} mV")
        
        # ===== ÁNGULO =====
        if self.idx_imu > 0:
            x_max = self.current_time_imu
            x_min = max(0, x_max - cfg.WINDOW_TIME_SEC)
            
            t_data = np.roll(self.time_imu, -self.idx_imu)
            angle_data = np.roll(self.data_angle, -self.idx_imu)
            
            mask = (t_data >= x_min) & (t_data <= x_max) & (t_data > 0)
            
            self.curve_angle.setData(t_data[mask], angle_data[mask])
            self.plot_angle.setXRange(x_min, x_max, padding=0)
            
            # Actualizar métrica de ángulo
            if np.any(mask):
                current_angle = angle_data[mask][-1] if mask.sum() > 0 else 0.0
                status = "✓ Calibrado" if self.angle_calculator.calibrated else "⚠ No calibrado"
                self.label_angle.setText(f"Ángulo: {current_angle:.1f}° ({status})")
    
    def _clear_buffers(self):
        """Limpia todos los buffers."""
        self.time_emg_ch0.fill(0)
        self.data_emg_ch0.fill(0)
        self.data_rms_ch0.fill(0)
        
        self.time_emg_ch1.fill(0)
        self.data_emg_ch1.fill(0)
        self.data_rms_ch1.fill(0)
        
        self.time_imu.fill(0)
        self.data_angle.fill(0)
        
        self.idx_emg_ch0 = 0
        self.idx_emg_ch1 = 0
        self.idx_imu = 0
        
        self.t0_emg = None
        self.t0_imu = None
        
        self.current_time_emg = 0.0
        self.current_time_imu = 0.0
        
        # Resetear procesadores
        self.emg_ch0_processor.reset()
        self.emg_ch1_processor.reset()
        self.angle_calculator.reset()
    
    def _open_calibration(self):
        """Abre el diálogo de calibración."""
        dialog = CalibrationDialog(self)
        
        # Actualizar ángulo crudo en tiempo real dentro del diálogo
        def update_angle_in_dialog():
            if dialog.isVisible():
                dialog.set_current_angle(self.current_raw_angle)
        
        timer = QTimer()
        timer.timeout.connect(update_angle_in_dialog)
        timer.start(100)  # Actualizar cada 100 ms
        
        result = dialog.exec()
        timer.stop()
        
        if result == QDialog.DialogCode.Accepted and dialog.calibration_done:
            calib_data = dialog.get_calibration_data()
            
            # Aplicar calibración
            if calib_data['mode'] == 1:
                self.angle_calculator.calibrate_one_point(calib_data['angle_ref_point1'])
            else:
                self.angle_calculator.calibrate_two_points(
                    calib_data['angle_raw_point1'], calib_data['angle_ref_point1'],
                    calib_data['angle_raw_point2'], calib_data['angle_ref_point2']
                )
            
            QMessageBox.information(
                self,
                "Calibración Exitosa",
                f"IMU calibrado correctamente ({calib_data['mode']} punto{'s' if calib_data['mode'] == 2 else ''}).\n"
                "Los ángulos ahora reflejan la flexión real de la rodilla."
            )
    
    def closeEvent(self, event):
        """Maneja el cierre de la ventana."""
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait()
        
        event.accept()