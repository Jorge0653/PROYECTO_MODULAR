"""
Ventana principal con menú del sistema
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                              QPushButton, QLabel, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont


class MainWindow(QMainWindow):
    """Ventana principal del sistema de evaluación de rodilla."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Evaluación y Rehabilitación de Rodilla")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setMinimumSize(600, 900)
        
        # Referencia a ventanas secundarias
        self.realtime_window = None
        self.recording_window = None
        self.data_analysis_window = None
        self.settings_window = None
        
        self._create_ui()
    
    def _create_ui(self):
        """Crea la interfaz del menú principal."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        
        # ========== TÍTULO ==========
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)  # menor separación entre título y subtítulo

        title_label = QLabel("SISTEMA DE EVALUACIÓN DE RODILLA")
        title_font = QFont("Avenir", 19, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #D9E4E4; padding-top: 8px; padding-bottom: 0px;")
        title_layout.addWidget(title_label)
        
        # Subtítulo
        subtitle = QLabel("EMG + IMU + Goniometría 3D")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #6F7474; font-size: 12pt; padding-top: 0px; font-family: 'Avenir';")
        title_layout.addWidget(subtitle)

        main_layout.addWidget(title_container)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("border-top: 1px solid #3A3A3C;")
        main_layout.addWidget(line)
        
        main_layout.addSpacing(20)
        
        # ========== BOTONES DEL MENÚ ==========
        
        # 1. Análisis en Tiempo Real (IMPLEMENTADO)
        btn_realtime, container_realtime = self._create_menu_button(
            "📊  Análisis en Tiempo Real",
            "Monitoreo continuo de EMG, ángulo de rodilla y métricas en vivo",
            enabled=True
        )
        btn_realtime.clicked.connect(self._open_realtime_analysis)
        main_layout.addWidget(container_realtime)
        # 2. Ejercicios Guiados (PLACEHOLDER)
        btn_exercises, container_exercises = self._create_menu_button(
            "🎯  Ejercicios Guiados",
            "Sesiones con feedback en tiempo real (sentadilla, step-up, marcha)",
            enabled=False
        )
        btn_exercises.clicked.connect(self._show_not_implemented)
        main_layout.addWidget(container_exercises)
        # 3. Grabación de Sesión
        btn_record, container_record = self._create_menu_button(
            "💾  Grabación de Sesión",
            "Grabar datos completos para análisis posterior",
            enabled=True
        )
        btn_record.clicked.connect(self._open_session_recording)
        main_layout.addWidget(container_record)
        # 4. Análisis Offline (PLACEHOLDER)
        btn_offline, container_offline = self._create_menu_button(
            "📂  Análisis de Datos Guardados",
            "Procesar y analizar sesiones previamente grabadas",
            enabled=True
        )
        btn_offline.clicked.connect(self._open_data_analysis)
        main_layout.addWidget(container_offline)
        # 5. Configuración del sistema
        btn_config, container_config = self._create_menu_button(
            "⚙️   Configuración del Sistema",
            "Parámetros de filtrado, calibración y visualización",
            enabled=True
        )
        btn_config.clicked.connect(self._open_settings)
        main_layout.addWidget(container_config)
        main_layout.addStretch()
        
        # Separador
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        line2.setStyleSheet("border-top: 1px solid #3A3A3C;")
        main_layout.addWidget(line2)
        
        # ========== BOTÓN SALIR ==========
        btn_exit = QPushButton("❌  Salir")
        btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #9C3428;
                color: #e4e4e4;
                font-size: 13pt;
                font-weight: bold;
                font-family: "Avenir";
                padding: 15px;
                border-radius: 3px;
                border: none;
            }
            QPushButton:hover {
                background-color: #822B22;
                border-radius: 6px;
                
            }
            QPushButton:pressed {
                background-color: #71261D;
                border-radius: 9px;
                
            }
        """)
        btn_exit.clicked.connect(self._confirm_exit)
        main_layout.addWidget(btn_exit)
        
        # ========== FOOTER ==========
        footer = QLabel("CUCEI - Universidad de Guadalajara | Ingeniería Biomédica © 2025")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #969696; font-size: 9pt; padding-top: 10px; font-family: 'Times New Roman';")
        main_layout.addWidget(footer)
    
    def _create_menu_button(self, text: str, description: str, enabled: bool = True) -> tuple[QPushButton, QWidget]:
        """Crea un botón del menú con estilo consistente."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        btn = QPushButton(text)
        
        if enabled:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #32598C;
                    color: #e4e4e4;
                    font-size: 14pt;
                    font-family: "Avenir";
                    font-weight: bold;
                    padding: 15px;
                    text-align: left;
                    border-radius: 6px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #233E62;
                    border-radius: 12px;          
                }
                QPushButton:pressed {
                    background-color: #182B43;
                    border-radius: 15px;  
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #98A2A9;
                    color: #777E84;
                    font-size: 14pt;
                    font-family: "Avenir";
                    font-weight: bold;
                    padding: 15px;
                    text-align: left;
                    border-radius: 6px;
                    border: none;
                    
                }
            """)
        
        # Descripción debajo del botón 
        desc_label = QLabel(f"    {description}")
        desc_label.setStyleSheet("color: #747474; font-size: 10pt; padding-left: 24px; font-family: 'Avenir';padding-top: 3px;")
        desc_label.setWordWrap(True)
        
        layout.addWidget(btn)
        layout.addWidget(desc_label)
        
        return btn, container

    def _open_realtime_analysis(self):
        """Abre la ventana de análisis en tiempo real."""
        from .realtime_analysis import RealtimeAnalysisWindow
        
        if self.realtime_window is None or not self.realtime_window.isVisible():
            self.realtime_window = RealtimeAnalysisWindow(self)
            self.realtime_window.window_reload_requested.connect(self._reload_realtime_analysis)
            self.realtime_window.show()
        else:
            self.realtime_window.raise_()
            self.realtime_window.activateWindow()

    def _open_session_recording(self):
        """Abre la ventana de grabación de sesión."""
        from .session_recording import SessionRecordingWindow

        if self.recording_window is None or not self.recording_window.isVisible():
            self.recording_window = SessionRecordingWindow(self)
            self.recording_window.show()
        else:
            self.recording_window.raise_()
            self.recording_window.activateWindow()

    def _open_data_analysis(self) -> None:
        """Abre la ventana de análisis de sesiones guardadas."""
        from .data_analysis_window import SessionAnalysisWindow

        if self.data_analysis_window is None or not self.data_analysis_window.isVisible():
            self.data_analysis_window = SessionAnalysisWindow(self)
            self.data_analysis_window.show()
        else:
            self.data_analysis_window.raise_()
            self.data_analysis_window.activateWindow()

    def _open_settings(self):
        """Abre el panel de configuración."""
        from .settings_window import SettingsWindow

        if self.settings_window is None or not self.settings_window.isVisible():
            self.settings_window = SettingsWindow(self)
            self.settings_window.settings_applied.connect(self._on_settings_applied_from_main)
            self.settings_window.finished.connect(self._on_settings_dialog_closed)
            self.settings_window.show()
        else:
            self.settings_window.raise_()
            self.settings_window.activateWindow()
    
    def _show_not_implemented(self):
        """Muestra mensaje de función no implementada."""
        QMessageBox.information(
            self,
            "Función no disponible",
            "Esta funcionalidad estará disponible en futuras versiones.\n\n",
            QMessageBox.StandardButton.Ok
        )
    
    def _confirm_exit(self):
        """Confirma antes de salir."""
        reply = QMessageBox.question(
            self,
            "Confirmar salida",
            "¿Estás seguro de que deseas cerrar el sistema?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Cerrar ventanas secundarias si están abiertas
            if self.realtime_window and self.realtime_window.isVisible():
                self.realtime_window.close()
            if self.recording_window and self.recording_window.isVisible():
                self.recording_window.close()
            if self.data_analysis_window and self.data_analysis_window.isVisible():
                self.data_analysis_window.close()
            if self.settings_window and self.settings_window.isVisible():
                self.settings_window.close()
            
            self.close()
    
    def closeEvent(self, event):
        """Maneja el cierre de la ventana."""
        # Asegurar que ventanas secundarias se cierren
        if self.realtime_window and self.realtime_window.isVisible():
            self.realtime_window.close()
        if self.recording_window and self.recording_window.isVisible():
            self.recording_window.close()
        if self.data_analysis_window and self.data_analysis_window.isVisible():
            self.data_analysis_window.close()
        if self.settings_window and self.settings_window.isVisible():
            self.settings_window.close()
        
        event.accept()

    def _reload_realtime_analysis(self):
        """Recrea la ventana de análisis en tiempo real tras aplicar configuración."""
        if self.realtime_window:
            self.realtime_window.close()
            self.realtime_window = None
        QTimer.singleShot(0, self._open_realtime_analysis)

    def _on_settings_applied_from_main(self) -> None:
        """Reinicia módulos activos tras guardar configuración desde el menú principal."""
        if self.realtime_window and self.realtime_window.isVisible():
            self._reload_realtime_analysis()
        if self.data_analysis_window and self.data_analysis_window.isVisible():
            self._reload_data_analysis()

    def _on_settings_dialog_closed(self, _: int) -> None:
        """Limpia la referencia al cerrar el diálogo de configuración."""
        self.settings_window = None

    def _reload_data_analysis(self) -> None:
        """Recrea la ventana de análisis offline aplicando los ajustes más recientes."""
        if self.data_analysis_window:
            self.data_analysis_window.close()
            self.data_analysis_window = None
        QTimer.singleShot(0, self._open_data_analysis)