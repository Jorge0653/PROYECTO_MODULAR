"""
Paquete core: procesamiento de señales y comunicación
"""
from .frame_decoder import FrameDecoder, crc16_ccitt
from .signal_processing import EMGProcessor, AngleCalculator
from .serial_reader import SerialReaderThread, get_available_ports

__all__ = [
    'FrameDecoder',
    'crc16_ccitt',
    'EMGProcessor',
    'AngleCalculator',
    'SerialReaderThread',
    'get_available_ports'
]