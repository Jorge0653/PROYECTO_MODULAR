"""
Thread de lectura serial asíncrona
"""
import serial
import serial.tools.list_ports
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional
from .frame_decoder import FrameDecoder
from config import SERIAL_BAUD, SERIAL_TIMEOUT


class SerialReaderThread(QThread):
    """Thread para lectura asíncrona del puerto serial."""
    
    frame_received = pyqtSignal(dict)  # Señal con frame decodificado
    connection_status = pyqtSignal(bool, str)  # (conectado, mensaje)
    
    def __init__(self, port: str, baud: int = SERIAL_BAUD):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = False
        self.serial_conn: Optional[serial.Serial] = None
        self.decoder = FrameDecoder()
    
    def run(self):
        """Bucle principal del thread."""
        self.running = True
        
        try:
            self.serial_conn = serial.Serial(
                self.port, self.baud, timeout=SERIAL_TIMEOUT,
                rtscts=False, dsrdtr=False
            )
            self.connection_status.emit(True, f"✓ Conectado a {self.port}")
            
            while self.running:
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    frames = self.decoder.feed(data)
                    
                    for frame in frames:
                        self.frame_received.emit(frame)
                else:
                    self.msleep(1)
                    
        except serial.SerialException as e:
            self.connection_status.emit(False, f"❌ Error serial: {str(e)}")
        except Exception as e:
            self.connection_status.emit(False, f"❌ Error: {str(e)}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.connection_status.emit(False, "⚫ Desconectado")
    
    def stop(self):
        """Detiene el thread de lectura."""
        self.running = False


def get_available_ports():
    """Retorna lista de puertos seriales disponibles."""
    ports = serial.tools.list_ports.comports()
    return [(port.device, f"{port.device} - {port.description}") for port in sorted(ports)]