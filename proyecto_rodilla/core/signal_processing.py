"""
Procesamiento de señales: filtros, RMS y cálculo de ángulo de flexión.
"""
import numpy as np
from scipy import signal
from collections import deque
from typing import Optional
from config import settings as cfg


class EMGProcessor:
    """Procesador de señales EMG: filtrado, detrend, RMS"""
    
    def __init__(self, fs: Optional[float] = None):
        self.fs = fs if fs is not None else cfg.EMG_FS
        
        # Diseño de filtros (solo una vez)
        # Pasa-altas (Butterworth 4° orden)
        self.sos_hp = signal.butter(4, cfg.EMG_HIGHPASS_CUTOFF, 'hp', fs=self.fs, output='sos')
        
        # Pasa-bajas (Butterworth 4° orden)
        self.sos_lp = signal.butter(4, cfg.EMG_LOWPASS_CUTOFF, 'lp', fs=self.fs, output='sos')
        
        # Notch (IIR)
        self.b_notch, self.a_notch = signal.iirnotch(cfg.EMG_NOTCH_FREQ, cfg.EMG_NOTCH_Q, self.fs)
        
        # Estados de los filtros (para procesamiento continuo)
        self.zi_hp = signal.sosfilt_zi(self.sos_hp)
        self.zi_lp = signal.sosfilt_zi(self.sos_lp)
        self.zi_notch = signal.lfilter_zi(self.b_notch, self.a_notch)
        
        # Buffer para RMS
        self.rms_buffer = deque(maxlen=cfg.RMS_WINDOW_SAMPLES)
    
    def process_sample(self, sample: float) -> tuple:
        """
        Procesa una muestra EMG individual.
        
        Returns:
            (muestra_filtrada, rms_actual)
        """
        # Detrend (restar media móvil simple o DC offset)
        # Para tiempo real, usamos un filtro pasa-altas que elimina DC
        
        # Aplicar filtros en cascada
        filtered, self.zi_hp = signal.sosfilt(self.sos_hp, [sample], zi=self.zi_hp)
        filtered, self.zi_notch = signal.lfilter(self.b_notch, self.a_notch, filtered, zi=self.zi_notch)
        filtered, self.zi_lp = signal.sosfilt(self.sos_lp, filtered, zi=self.zi_lp)
        
        filtered_sample = filtered[0]
        
        # Calcular RMS con ventana móvil
        self.rms_buffer.append(filtered_sample ** 2)
        rms_value = np.sqrt(np.mean(self.rms_buffer)) if len(self.rms_buffer) > 0 else 0.0
        
        return filtered_sample, rms_value
    
    def reset(self):
        """Reinicia estados de los filtros"""
        self.zi_hp = signal.sosfilt_zi(self.sos_hp)
        self.zi_lp = signal.sosfilt_zi(self.sos_lp)
        self.zi_notch = signal.lfilter_zi(self.b_notch, self.a_notch)
        self.rms_buffer.clear()


class AngleCalculator:
    """
    Calcula el ángulo de flexión de rodilla usando fusión accel + gyro.
    IMU montado con X lateral, Y superior (hacia la rodilla) y Z anterior (hacia el frente).
    """
    
    def __init__(self, alpha: Optional[float] = None, fs: Optional[float] = None):
        self.alpha = alpha if alpha is not None else cfg.COMPLEMENTARY_FILTER_ALPHA
        imu_fs = fs if fs is not None else cfg.IMU_FS
        self.dt = 1.0 / imu_fs if imu_fs else 0.02
        
        # Estado
        self.angle = 0.0  # Ángulo actual (°)
        self.last_time = None
        
        # Calibración
        self.calibrated = False
        self.offset = 0.0      # Para calibración de 1 punto
        self.scale = 1.0       # Para calibración de 2 puntos
        self.angle_ref1 = 0.0  # Ángulo de referencia punto 1
        self.angle_ref2 = 90.0 # Ángulo de referencia punto 2
    
    def calculate_angle_accel(self, ax: float, ay: float, az: float) -> float:
        """
        Calcula ángulo de inclinación solo con acelerómetro.
        Asume IMU en espinilla, eje X lateral, eje Y superior y eje Z anterior.
        """
        # Ángulo en plano sagital (flexión/extensión)
        # Usar arctan2 para obtener ángulo respecto a gravedad
        angle = np.degrees(np.arctan2(-az, ay))
        return angle
    
    def update(self, ax: float, ay: float, az: float, 
               gx: float, gy: float, gz: float, timestamp_us: float) -> float:
        """
        Actualiza el ángulo usando filtro complementario.
        
        Args:
            ax, ay, az: Aceleración en g
            gx, gy, gz: Velocidad angular en °/s
            timestamp_us: Timestamp en microsegundos
        
        Returns:
            Ángulo de flexión calibrado (°)
        """
        # Calcular dt real (en caso de que no sea constante)
        if self.last_time is not None:
            dt = (timestamp_us - self.last_time) / 1e6  # a segundos
        else:
            dt = self.dt
        
        self.last_time = timestamp_us
        
        # Ángulo del acelerómetro (referencia absoluta pero ruidoso)
        angle_accel = self.calculate_angle_accel(ax, ay, az)
        
        # Integración del giroscopio (rotación de flexión alrededor del eje lateral X)
        gyro_rate = -gx
        angle_gyro = self.angle + gyro_rate * dt
        
        # Filtro complementario
        self.angle = self.alpha * angle_gyro + (1 - self.alpha) * angle_accel
        
        # Aplicar calibración
        if self.calibrated:
            calibrated_angle = (self.angle - self.offset) * self.scale
        else:
            calibrated_angle = self.angle
        
        return calibrated_angle
    
    def calibrate_one_point(self, angle_ref: float = 0.0):
        """
        Calibración de 1 punto (solo offset).
        
        Args:
            angle_ref: Ángulo de referencia real (°), típicamente 0° = pierna estirada
        """
        self.offset = self.angle - angle_ref
        self.scale = 1.0
        self.calibrated = True
    
    def calibrate_two_points(self, angle_raw1: float, angle_ref1: float,
                            angle_raw2: float, angle_ref2: float):
        """
        Calibración de 2 puntos (offset + escala).
        
        Args:
            angle_raw1: Ángulo crudo en punto 1
            angle_ref1: Ángulo de referencia real en punto 1 (°)
            angle_raw2: Ángulo crudo en punto 2
            angle_ref2: Ángulo de referencia real en punto 2 (°)
        """
        # Mapeo lineal: angle_cal = (angle_raw - offset) * scale
        # angle_ref1 = (angle_raw1 - offset) * scale
        # angle_ref2 = (angle_raw2 - offset) * scale
        
        self.scale = (angle_ref2 - angle_ref1) / (angle_raw2 - angle_raw1)
        self.offset = angle_raw1 - (angle_ref1 / self.scale)
        self.angle_ref1 = angle_ref1
        self.angle_ref2 = angle_ref2
        self.calibrated = True
    
    def reset(self):
        """Reinicia el calculador"""
        self.angle = 0.0
        self.last_time = None
        self.calibrated = False
        self.offset = 0.0
        self.scale = 1.0