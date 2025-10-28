"""
Sistema de Evaluación y Rehabilitación de Rodilla
Punto de entrada principal

Autor: Joshua Uriel de la Rosa Barragán
Institución: CUCEI - Universidad de Guadalajara
Proyecto: Módulo Integrador 2025
"""
import sys
from PyQt6.QtWidgets import QApplication
from gui import MainWindow


def main():
    """Punto de entrada principal."""
    app = QApplication(sys.argv)
    app.setStyle('qtmodern-dark')  # Tema oscuro moderno
    
    # Configurar estilo global
    app.setStyleSheet("""
        QMainWindow {
            background-color: #ecf0f1;
        }
        QWidget {
            font-family: Arial, sans-serif;
        }
        QMessageBox {
            background-color: #363432;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()