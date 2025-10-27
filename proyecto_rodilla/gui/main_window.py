"""
Ventana principal con menú del sistema
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                              QPushButton, QLabel, QMessageBox, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class MainWindow(QMainWindow):
    """Ventana principal del sistema de evaluación de rodilla."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Evaluación y Rehabilitación de Rodilla")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(700, 500)
        
        # Referencia a ventanas secundarias
        self.realtime_window = None
        
        self._create_ui()
    
    def _create_ui(self):
        """Crea la interfaz del menú principal."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        
        # ========== TÍTULO ==========
        title_label = QLabel("SISTEMA DE EVALUACIÓN DE RODILLA")
        title_font = QFont("Arial", 18, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 20px;")
        main_layout.addWidget(title_label)
        
        # Subtítulo
        subtitle = QLabel("EMG + IMU + Goniometría 3D")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; font-size: 12pt; padding-bottom: 20px;")
        main_layout.addWidget(subtitle)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)
        
        main_layout.addSpacing(20)
        
        # ========== BOTONES DEL MENÚ ==========
        
        # 1. Análisis en Tiempo Real (IMPLEMENTADO)
        # 1. Análisis en Tiempo Real (IMPLEMENTADO)
        btn_realtime, container_realtime = self._create_menu_button(
            "📊  Análisis en Tiempo Real",
            "Monitoreo continuo de EMG, ángulo de rodilla y métricas en vivo",
            enabled=True
        )
        btn_realtime.clicked.connect(self._open_realtime_analysis)
        main_layout.addWidget(container_realtime)
        # 2. Ejercicios Guiados (PLACEHOLDER)
        # 2. Ejercicios Guiados (PLACEHOLDER)
        btn_exercises, container_exercises = self._create_menu_button(
            "🎯  Ejercicios Guiados",
            "Sesiones con feedback en tiempo real (sentadilla, step-up, marcha)",
            enabled=False
        )
        btn_exercises.clicked.connect(self._show_not_implemented)
        main_layout.addWidget(container_exercises)
        # 3. Grabación de Sesión (PLACEHOLDER)
        # 3. Grabación de Sesión (PLACEHOLDER)
        btn_record, container_record = self._create_menu_button(
            "💾  Grabación de Sesión",
            "Grabar datos completos para análisis posterior",
            enabled=False
        )
        btn_record.clicked.connect(self._show_not_implemented)
        main_layout.addWidget(container_record)
        # 4. Análisis Offline (PLACEHOLDER)
        # 4. Análisis Offline (PLACEHOLDER)
        btn_offline, container_offline = self._create_menu_button(
            "📂  Análisis de Datos Guardados",
            "Procesar y analizar sesiones previamente grabadas",
            enabled=False
        )
        btn_offline.clicked.connect(self._show_not_implemented)
        main_layout.addWidget(container_offline)
        # 5. Configuración (PLACEHOLDER)
        # 5. Configuración (PLACEHOLDER)
        btn_config, container_config = self._create_menu_button(
            "⚙️   Configuración del Sistema",
            "Parámetros de filtrado, calibración y visualización",
            enabled=False
        )
        btn_config.clicked.connect(self._show_not_implemented)
        main_layout.addWidget(container_config)
        main_layout.addStretch()
        
        # Separador
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line2)
        
        # ========== BOTÓN SALIR ==========
        btn_exit = QPushButton("❌  Salir")
        btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        btn_exit.clicked.connect(self._confirm_exit)
        main_layout.addWidget(btn_exit)
        
        # ========== FOOTER ==========
        footer = QLabel("CUCEI - Universidad de Guadalajara | Proyecto Modular 2025")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #95a5a6; font-size: 9pt; padding-top: 10px;")
        main_layout.addWidget(footer)
    
    def _create_menu_button(self, text: str, description: str, enabled: bool = True) -> QPushButton:
    #def _create_menu_button(self, text: str, description: str, enabled: bool = True):
        """Crea un botón del menú con estilo consistente."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        btn = QPushButton(text)
        
        if enabled:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    font-size: 14pt;
                    font-weight: bold;
                    padding: 15px;
                    text-align: left;
                    border-radius: 8px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #21618c;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #bdc3c7;
                    color: #7f8c8d;
                    font-size: 14pt;
                    font-weight: bold;
                    padding: 15px;
                    text-align: left;
                    border-radius: 8px;
                    border: none;
                }
            """)
        
        # Descripción
        desc_label = QLabel(f"    {description}")
        desc_label.setStyleSheet("color: #7f8c8d; font-size: 10pt; padding-left: 5px;")
        desc_label.setWordWrap(True)
        
        layout.addWidget(btn)
        layout.addWidget(btn)
        layout.addWidget(desc_label)
        
        return btn, container
    def _open_realtime_analysis(self):
        """Abre la ventana de análisis en tiempo real."""
        from .realtime_analysis import RealtimeAnalysisWindow
        
        if self.realtime_window is None or not self.realtime_window.isVisible():
            self.realtime_window = RealtimeAnalysisWindow()
            self.realtime_window.show()
        else:
            self.realtime_window.raise_()
            self.realtime_window.activateWindow()
    
    def _show_not_implemented(self):
        """Muestra mensaje de función no implementada."""
        QMessageBox.information(
            self,
            "Función no disponible",
            "Esta funcionalidad estará disponible en futuras versiones.\n\n"
            "Por ahora, puedes usar el módulo de Análisis en Tiempo Real.",
            QMessageBox.StandardButton.Ok
        )
    
    def _confirm_exit(self):
        """Confirma antes de salir."""
        reply = QMessageBox.question(
            self,
            "Confirmar salida",
            "¿Estás seguro de que deseas salir del sistema?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Cerrar ventanas secundarias si están abiertas
            if self.realtime_window and self.realtime_window.isVisible():
                self.realtime_window.close()
            
            self.close()
    
    def closeEvent(self, event):
        """Maneja el cierre de la ventana."""
        # Asegurar que ventanas secundarias se cierren
        if self.realtime_window and self.realtime_window.isVisible():
            self.realtime_window.close()
        
        event.accept()