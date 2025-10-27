"""
Decodificador del protocolo binario ESP32-S3
"""
import struct
from typing import Optional, Dict, List
from config import (PREAMBLE, FRAME_TYPE_EMG, FRAME_TYPE_IMU,
                    VREF, PGA_GAIN, ADC_RESOLUTION)


def crc16_ccitt(data: bytes, initial_crc: int = 0xFFFF) -> int:
    """Calcula CRC16 CCITT-FALSE (polinomio 0x1021, inicial 0xFFFF)."""
    crc = initial_crc
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
        crc &= 0xFFFF
    return crc


class FrameDecoder:
    """Decodifica el protocolo binario del ESP32-S3."""
    
    def __init__(self):
        self.buffer = bytearray()
    
    def feed(self, data: bytes) -> List[Dict]:
        """
        Alimenta bytes al decodificador y retorna lista de frames válidos.
        
        Returns:
            Lista de diccionarios con frames decodificados
        """
        self.buffer.extend(data)
        frames = []
        
        while len(self.buffer) >= 7:  # Mínimo: preamble(2) + type(1) + crc(2) = 5 + payload mínimo
            # Buscar preámbulo
            idx = self.buffer.find(PREAMBLE)
            if idx == -1:
                self.buffer = self.buffer[-1:]  # Guardar último byte
                break
            
            if idx > 0:
                self.buffer = self.buffer[idx:]
            
            if len(self.buffer) < 7:
                break
            
            frame_type = self.buffer[2]
            
            # Determinar tamaño de payload según tipo
            if frame_type == FRAME_TYPE_EMG:
                payload_size = 14  # SEQ(2) + TS(4) + A(4) + B(4)
            elif frame_type == FRAME_TYPE_IMU:
                payload_size = 18  # SEQ(2) + TS(4) + 6*int16(12)
            else:
                # Tipo desconocido, descartar y buscar siguiente
                self.buffer = self.buffer[2:]
                continue
            
            total_frame_size = 2 + 1 + payload_size + 2  # Preamble + Type + Payload + CRC
            
            if len(self.buffer) < total_frame_size:
                break  # Frame incompleto
            
            frame_data = self.buffer[:total_frame_size]
            
            # Verificar CRC
            crc_received = struct.unpack('<H', frame_data[-2:])[0]
            crc_calculated = crc16_ccitt(frame_data[2:-2])
            
            if crc_received == crc_calculated:
                payload = frame_data[3:-2]
                decoded = self._decode_payload(frame_type, payload)
                if decoded:
                    frames.append(decoded)
            
            # Avanzar buffer
            self.buffer = self.buffer[total_frame_size:]
        
        return frames
    
    def _decode_payload(self, frame_type: int, payload: bytes) -> Optional[Dict]:
        """Decodifica el payload según el tipo de frame."""
        try:
            if frame_type == FRAME_TYPE_EMG:
                seq, ts, raw_a, raw_b = struct.unpack('<HIii', payload)
                
                # Conversión a voltaje
                ch0_v = (raw_a / ADC_RESOLUTION) * (VREF / PGA_GAIN)
                ch1_v = (raw_b / ADC_RESOLUTION) * (VREF / PGA_GAIN)
                
                return {
                    'type': 'EMG',
                    'seq': seq,
                    'timestamp_us': ts,
                    'ch0': ch0_v,
                    'ch1': ch1_v
                }
            
            elif frame_type == FRAME_TYPE_IMU:
                seq, ts, ax, ay, az, gx, gy, gz = struct.unpack('<HIhhhhhh', payload)
                
                # Conversión según datasheet MPU6050
                # Accel: ±2g → 16384 LSB/g
                # Gyro: ±250°/s → 131 LSB/°/s
                return {
                    'type': 'IMU',
                    'seq': seq,
                    'timestamp_us': ts,
                    'ax': ax / 16384.0,  # en g
                    'ay': ay / 16384.0,
                    'az': az / 16384.0,
                    'gx': gx / 131.0,    # en °/s
                    'gy': gy / 131.0,
                    'gz': gz / 131.0
                }
        except struct.error:
            return None
        
        return None