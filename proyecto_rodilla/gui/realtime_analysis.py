"""
Módulo de análisis en tiempo real: EMG + IMU + Ángulo de flexión
"""
import sys
import numpy as np
from typing import Optional, Dict
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt6.QtCore import QTimer, QTime
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QMessageBox,
    QFrame,
    QDialog,
)
from PyQt6.QtGui import QFont

from core import SerialReaderThread, get_available_ports, EMGProcessor, AngleCalculator
from config import settings as cfg
from utils import save_json, load_json
from .settings_window import SettingsWindow
from .calibration_dialog import CalibrationDialog


class RealtimeAnalysisWindow(QMainWindow):
    """Ventana de análisis en tiempo real."""
    window_reload_requested = QtCore.pyqtSignal()
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Análisis en Tiempo Real - EMG + IMU")
        self.setWindowState(QtCore.Qt.WindowState.WindowMaximized)
        self.setMinimumSize(1200, 800)
        
        # Estados
        self.serial_thread: Optional[SerialReaderThread] = None
        self.is_connected = False
        self.settings_window: Optional[SettingsWindow] = None
        
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
        self.current_raw_angle = self.angle_calculator.last_uncalibrated_angle
        
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
        pg.setConfigOption('background', "#1E1F20") # Color de fondo oscuro
        pg.setConfigOption('foreground', '#E4E4E4') # Color de líneas y texto claro
        pg.setConfigOption('antialias', False) # Desactivar antialiasing para mejor rendimiento
        pg.setConfigOption('useOpenGL', True) # Activar OpenGL si está disponible para mejor rendimiento
         
        
        # ========== GRÁFICAS ==========
        
        # Canal EMG 0
        self.plot_ch0 = pg.PlotWidget(title='Canal EMG 0 (Cuadríceps)')
        self.plot_ch0.setLabel('left', 'Voltaje', units='V') 
        self.plot_ch0.setLabel('bottom', 'Tiempo', units='s')
        self.plot_ch0.showGrid(x=True, y=True, alpha=0.3)
        self.plot_ch0.setYRange(-0.5, 0.5, padding=0.05)
        self.plot_ch0.addLegend()
        
        self.curve_ch0 = self.plot_ch0.plot(pen=pg.mkPen(color=cfg.COLOR_CH0, width=1.5), name='EMG CH0')
        self.curve_rms_ch0 = self.plot_ch0.plot(pen=pg.mkPen(color=cfg.COLOR_RMS_CH0, width=2), name='RMS CH0')
        
        main_layout.addWidget(self.plot_ch0)
        
        # Canal EMG 1
        self.plot_ch1 = pg.PlotWidget(title='Canal EMG 1 (Isquiotibiales)')
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
        self.plot_angle.setYRange(-10, 180, padding=0.05)
        
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
        toolbar_widget.setObjectName("controlToolbar")
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(12, 10, 12, 10)
        toolbar_layout.setSpacing(12)

        label_port = QLabel("Puerto:")
        toolbar_layout.addWidget(label_port)

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(220)
        self._refresh_ports()
        toolbar_layout.addWidget(self.port_combo)

        btn_refresh = QPushButton("⟳")
        btn_refresh.setProperty("category", "minimal")
        btn_refresh.setFixedWidth(40)
        btn_refresh.setToolTip("Actualizar lista de puertos")
        btn_refresh.clicked.connect(self._refresh_ports)
        toolbar_layout.addWidget(btn_refresh)

        self.btn_connect = QPushButton("Conectar")
        self.btn_connect.setProperty("category", "primary")
        self.btn_connect.setProperty("state", "disconnected")
        self.btn_connect.setToolTip("Iniciar conexión serial")
        self.btn_connect.clicked.connect(self._toggle_connection)
        toolbar_layout.addWidget(self.btn_connect)

        toolbar_layout.addSpacing(12)

        self.btn_calibrate = QPushButton("🎯 Calibrar IMU")
        self.btn_calibrate.setProperty("category", "primary")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.clicked.connect(self._open_calibration)
        toolbar_layout.addWidget(self.btn_calibrate)

        btn_clear = QPushButton("Limpiar gráficas")
        btn_clear.setProperty("category", "secondary")
        btn_clear.setToolTip("Restablecer buffers y gráficas")
        btn_clear.clicked.connect(self._clear_buffers)
        toolbar_layout.addWidget(btn_clear)

        toolbar_layout.addStretch(1)

        self.btn_settings = QPushButton("⚙️ Configuración")
        self.btn_settings.setProperty("category", "secondary")
        self.btn_settings.setToolTip("Abrir configuración del sistema")
        self.btn_settings.clicked.connect(self._open_settings)
        toolbar_layout.addWidget(self.btn_settings)

        parent_layout.addWidget(toolbar_widget)

        toolbar_widget.setStyleSheet(
            """
            QWidget#controlToolbar {
                background-color: #1F1F21;
                border: 1px solid #2D2D2F;
                border-radius: 12px;
            }
            QWidget#controlToolbar QLabel {
                color: #D9E4E4;
                font-family: "Avenir";
                font-size: 10.5pt;
            }
            QWidget#controlToolbar QComboBox {
                background-color: #2C2C2E;
                border: 1px solid #3A3A3C;
                border-radius: 6px;
                padding: 4px 9px;
                color: #F5F5F7;
                min-height: 30px;
            }
            QWidget#controlToolbar QComboBox:disabled {
                background-color: #272729;
                color: #7B7B80;
                border-color: #3C3C3F;
            }
            QWidget#controlToolbar QPushButton {
                background-color: #32598C;
                color: #FFFFFF;
                font-family: "Avenir";
                font-weight: 600;
                padding: 8px 16px;
                border: none;
                border-radius: 8px;
            }
            QWidget#controlToolbar QPushButton:hover {
                background-color: #233E62;
            }
            QWidget#controlToolbar QPushButton:pressed {
                background-color: #192B45;
            }
            QWidget#controlToolbar QPushButton[category="secondary"] {
                background-color: #3A3A3C;
                color: #D0D0D5;
                font-weight: 500;
            }
            QWidget#controlToolbar QPushButton[category="secondary"]:hover {
                background-color: #4A4A4D;
            }
            QWidget#controlToolbar QPushButton[category="secondary"]:pressed {
                background-color: #2F2F31;
            }
            QWidget#controlToolbar QPushButton[category="minimal"] {
                background-color: #2C2C2E;
                color: #D9E4E4;
                font-size: 13pt;
                padding: 6px 0;
            }
            QWidget#controlToolbar QPushButton[category="minimal"]:hover {
                background-color: #3A3A3C;
            }
            QWidget#controlToolbar QPushButton[state="connected"] {
                background-color: #9C3428;
            }
            QWidget#controlToolbar QPushButton[state="connected"]:hover {
                background-color: #822B22;
            }
            QWidget#controlToolbar QPushButton[state="connected"]:pressed {
                background-color: #71261D;
            }
        """
        )

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("border-top: 1px solid #2F2F31;")
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
            self.btn_connect.setProperty("state", "connected")
            self.btn_connect.setToolTip("Finalizar conexión serial")
            self._repolish(self.btn_connect)
            self.port_combo.setEnabled(False)
            self.btn_calibrate.setEnabled(True)
        else:
            self.is_connected = False
            self.btn_connect.setText("Conectar")
            self.btn_connect.setProperty("state", "disconnected")
            self.btn_connect.setToolTip("Iniciar conexión serial")
            self._repolish(self.btn_connect)
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
            
            self.current_raw_angle = self.angle_calculator.last_uncalibrated_angle
            
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
        self.current_raw_angle = self.angle_calculator.last_uncalibrated_angle
    
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
                if calib_data.get('angle_raw_point1') is not None:
                    raw_point = float(calib_data['angle_raw_point1'])
                    self.angle_calculator.angle = raw_point
                    self.angle_calculator.last_uncalibrated_angle = raw_point
                self.angle_calculator.calibrate_one_point(calib_data['angle_ref_point1'])
            else:
                self.angle_calculator.calibrate_two_points(
                    calib_data['angle_raw_point1'], calib_data['angle_ref_point1'],
                    calib_data['angle_raw_point2'], calib_data['angle_ref_point2']
                )

            self.current_raw_angle = self.angle_calculator.last_uncalibrated_angle
            
            QMessageBox.information(
                self,
                "Calibración Exitosa",
                f"IMU calibrado correctamente ({calib_data['mode']} punto{'s' if calib_data['mode'] == 2 else ''}).\n"
                "Los ángulos ahora reflejan la flexión real de la rodilla."
            )

    def _open_settings(self):
        """Abre el panel de configuración asociado a esta ventana."""
        if self.settings_window is None or not self.settings_window.isVisible():
            self.settings_window = SettingsWindow(self)
            self.settings_window.settings_applied.connect(self._handle_settings_applied)
            self.settings_window.finished.connect(self._on_settings_dialog_closed)
            self.settings_window.show()
        else:
            self.settings_window.raise_()
            self.settings_window.activateWindow()

    def _handle_settings_applied(self) -> None:
        """Solicita reiniciar la ventana tras guardar configuración."""
        self.window_reload_requested.emit()
        self.close()

    def _on_settings_dialog_closed(self, _: int) -> None:
        """Limpia la referencia del diálogo al cerrarse."""
        self.settings_window = None

    @staticmethod
    def _repolish(widget: QtWidgets.QWidget) -> None:
        """Fuerza a Qt a recalcular estilos después de cambiar propiedades."""
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
    
    def closeEvent(self, event):
        """Maneja el cierre de la ventana."""
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait()
        if self.settings_window and self.settings_window.isVisible():
            self.settings_window.close()
        self.settings_window = None
        if self.update_timer.isActive():
            self.update_timer.stop()
        event.accept()