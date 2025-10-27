"""
Configuración centralizada del sistema de evaluación de rodilla
"""

# ==================== CONFIGURACIÓN SERIAL ====================
SERIAL_BAUD = 921600
SERIAL_TIMEOUT = 1.0

# ==================== PROTOCOLO DE COMUNICACIÓN ====================
PREAMBLE = bytes([0xA5, 0x5A])
FRAME_TYPE_EMG = 0x01
FRAME_TYPE_IMU = 0x02

# ==================== PARÁMETROS EMG ====================
EMG_FS = 500  # Frecuencia de muestreo (Hz)
EMG_CHANNELS = 2  # Número de canales

# Filtros EMG
EMG_HIGHPASS_CUTOFF = 20.0  # Hz - elimina drift
EMG_LOWPASS_CUTOFF = 249.0  # Hz - anti-aliasing
EMG_NOTCH_FREQ = 60.0       # Hz - interferencia de línea (México)
EMG_NOTCH_Q = 30.0          # Factor de calidad del filtro notch

# RMS
RMS_WINDOW_MS = 100         # ms
RMS_WINDOW_SAMPLES = int(EMG_FS * RMS_WINDOW_MS / 1000)  # 50 muestras

# ==================== PARÁMETROS IMU ====================
IMU_FS = 50  # Frecuencia de muestreo efectiva (Hz)

# Filtro complementario para fusión accel + gyro
COMPLEMENTARY_FILTER_ALPHA = 0.98  # 0.98 es típico (98% gyro, 2% accel)

# Calibración
CALIBRATION_POINTS = 2  # Puede ser 1 o 2 (configurable)
# 1 punto: solo offset (pierna estirada = 0°)
# 2 puntos: offset + escala (pierna estirada = 0°, flexión conocida = X°)

# ==================== PARÁMETROS ADS1256 ====================
VREF = 3.275
PGA_GAIN = 1
ADC_RESOLUTION = 2**23

# ==================== VISUALIZACIÓN ====================
WINDOW_TIME_SEC = 5.0       # Ventana temporal (segundos)
UPDATE_FPS = 25             # Actualización de gráficas (FPS)
UPDATE_INTERVAL_MS = int(1000 / UPDATE_FPS)  # 40 ms

EMG_BUFFER_SIZE = int(EMG_FS * WINDOW_TIME_SEC)  # 2500 muestras
IMU_BUFFER_SIZE = int(IMU_FS * WINDOW_TIME_SEC)  # 250 muestras

# ==================== COLORES (RGB) ====================
COLOR_CH0 = (31, 119, 180)    # Azul
COLOR_CH1 = (255, 127, 14)    # Naranja
COLOR_RMS_CH0 = (44, 160, 44) # Verde
COLOR_RMS_CH1 = (214, 39, 40) # Rojo
COLOR_ANGLE = (148, 103, 189) # Púrpura

# ==================== PATHS ====================
CONFIG_FILE = "system_config.json"
CALIBRATION_FILE = "imu_calibration.json"