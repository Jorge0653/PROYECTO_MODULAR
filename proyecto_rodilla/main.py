"""
Sistema de Evaluación y Rehabilitación de Rodilla
Punto de entrada principal

Autores: 
    Daniela Carolina Ruiz García
    Jorge Eduardo Magaña Torres
    Joshua Uriel de la Rosa Barragán
         
Institución: CUCEI - Universidad de Guadalajara
Carrera: Ingeniería en Biomédica
Proyecto: Proyecto Modular 2025B
"""
import sys
from PyQt6.QtWidgets import QApplication
from gui import MainWindow


def main():
    """Punto de entrada principal."""
    app = QApplication(sys.argv)
    app.setStyle('fusion')  # Tema oscuro moderno

    # Configurar estilo global
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1F1F21;
        }
        QWidget {
            font-family: Avenir;
        }
    """)

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()