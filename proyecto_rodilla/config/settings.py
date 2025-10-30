"""
Configuración centralizada y editable del sistema de evaluación de rodilla.

Las variables expuestas a nivel de módulo conservan compatibilidad con el código
		[
			"IMU_FS",
			"COMPLEMENTARY_FILTER_ALPHA",
			"CALIBRATION_POINTS",
			"AUTO_CALIB_CAMERA_INDEX",
			"AUTO_CALIB_FPS",
			"AUTO_CALIB_REFERENCE_EXT",
			"AUTO_CALIB_REFERENCE_FLEX",
			"AUTO_CALIB_TOLERANCE_DEG",
			"AUTO_CALIB_STABILITY_FRAMES",
			"AUTO_CALIB_VISIBILITY_THRESHOLD",
		],
Utiliza las funciones ``get_settings`` y ``update_settings`` para interactuar con
los valores en tiempo de ejecución y persistirlos en ``system_config.json``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Directorios base
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==================== DEFINICIÓN DE VALORES ====================
DEFAULT_SETTINGS: Dict[str, Any] = {
	# Conexión serial
	"SERIAL_BAUD": 921600,
	"SERIAL_TIMEOUT": 1.0,

	# Protocolo de comunicación
	"PREAMBLE": [0xA5, 0x5A],  # Se almacena como lista para facilitar serialización JSON
	"FRAME_TYPE_EMG": 0x01,
	"FRAME_TYPE_IMU": 0x02,

	# Parámetros EMG
	"EMG_FS": 1635.78,
	"EMG_CHANNELS": 2,
	"EMG_HIGHPASS_CUTOFF": 20.0,
	"EMG_LOWPASS_CUTOFF": 500.0,
	"EMG_NOTCH_FREQ": 60.0,
	"EMG_NOTCH_Q": 30.0,
	"RMS_WINDOW_MS": 100,

	# Parámetros IMU
	"IMU_FS": 50,
	"COMPLEMENTARY_FILTER_ALPHA": 0.02,
	"CALIBRATION_POINTS": 2,
	"AUTO_CALIB_CAMERA_INDEX": 3,
	"AUTO_CALIB_FPS": 20,
	"AUTO_CALIB_REFERENCE_EXT": 0.0,
	"AUTO_CALIB_REFERENCE_FLEX": 90.0,
	"AUTO_CALIB_TOLERANCE_DEG": 2.0,
	"AUTO_CALIB_STABILITY_FRAMES": 12,
	"AUTO_CALIB_VISIBILITY_THRESHOLD": 0.75,

	# ADC ADS1256
	"VREF": 3.275,
	"PGA_GAIN": 1,
	"ADC_RESOLUTION": 2 ** 23,

	# Visualización
	"WINDOW_TIME_SEC": 5.0,
	"UPDATE_FPS": 25,
	"COLOR_CH0": [31, 119, 180], 
	"COLOR_CH1": [255, 127, 14],
	"COLOR_RMS_CH0": [44, 160, 44],
	"COLOR_RMS_CH1": [214, 39, 40],
	"COLOR_ANGLE": [148, 103, 189],

	# Archivos
	"CONFIG_FILE": "system_config.json",
	"CALIBRATION_FILE": "imu_calibration.json",
}

# Valores derivados que dependen de otros parámetros
DERIVED_KEYS: Dict[str, Any] = {
	"RMS_WINDOW_SAMPLES": lambda s: int(s["EMG_FS"] * s["RMS_WINDOW_MS"] / 1000),
	"UPDATE_INTERVAL_MS": lambda s: int(1000 / max(1, s["UPDATE_FPS"])),
	"EMG_BUFFER_SIZE": lambda s: int(s["EMG_FS"] * s["WINDOW_TIME_SEC"]),
	"IMU_BUFFER_SIZE": lambda s: int(s["IMU_FS"] * s["WINDOW_TIME_SEC"]),
}

# Metadatos para UI y validaciones
SETTINGS_SCHEMA: Dict[str, Dict[str, Any]] = {
	"SERIAL_BAUD": {
		"section": "Conexión Serial",
		"label": "Baud rate",
		"type": "int",
		"min": 9600,
		"max": 4_000_000,
		"step": 100,
		"description": "Velocidad de comunicación con el microcontrolador.",
	},
	"SERIAL_TIMEOUT": {
		"section": "Conexión Serial",
		"label": "Timeout (s)",
		"type": "float",
		"min": 0.01,
		"max": 10.0,
		"step": 0.01,
		"description": "Tiempo de espera para lecturas seriales.",
	},
	"PREAMBLE": {
		"section": "Protocolo",
		"label": "Preámbulo",
		"type": "bytes",
		"editable": True,
		"description": "Bytes iniciales que identifican cada trama.",
	},
	"FRAME_TYPE_EMG": {
		"section": "Protocolo",
		"label": "Tipo de frame EMG",
		"type": "int",
		"min": 0,
		"max": 255,
		"description": "Identificador para tramas de electromiografía.",
	},
	"FRAME_TYPE_IMU": {
		"section": "Protocolo",
		"label": "Tipo de frame IMU",
		"type": "int",
		"min": 0,
		"max": 255,
		"description": "Identificador para tramas de datos inerciales.",
	},
	"EMG_FS": {
		"section": "EMG",
		"label": "Frecuencia de muestreo (Hz)",
		"type": "float",
		"min": 100,
		"max": 4000,
		"step": 10,
		"description": "Frecuencia de muestreo para EMG.",
	},
	"EMG_CHANNELS": {
		"section": "EMG",
		"label": "Canales EMG",
		"type": "int",
		"min": 1,
		"max": 8,
		"editable": False,
	},
	"EMG_HIGHPASS_CUTOFF": {
		"section": "EMG",
		"label": "High-pass cutoff (Hz)",
		"type": "float",
		"min": 0.05,
		"max": 200.0,
		"step": 0.5,
	},
	"EMG_LOWPASS_CUTOFF": {
		"section": "EMG",
		"label": "Low-pass cutoff (Hz)",
		"type": "float",
		"min": 40.0,
		"max": 600.0,
		"step": 1.0,
	},
	"EMG_NOTCH_FREQ": {
		"section": "EMG",
		"label": "Frecuencia notch (Hz)",
		"type": "float",
		"min": 40.0,
		"max": 70.0,
		"step": 0.5,
	},
	"EMG_NOTCH_Q": {
		"section": "EMG",
		"label": "Factor Q notch",
		"type": "float",
		"min": 1.0,
		"max": 100.0,
		"step": 0.5,
	},
	"RMS_WINDOW_MS": {
		"section": "EMG",
		"label": "Ventana RMS (ms)",
		"type": "int",
		"min": 10,
		"max": 1000,
		"step": 5,
	},
	"RMS_WINDOW_SAMPLES": {
		"section": "EMG",
		"label": "Ventana RMS (muestras)",
		"type": "derived",
		"editable": False,
		"description": "Calculado automáticamente a partir de la frecuencia de muestreo.",
	},
	"IMU_FS": {
		"section": "IMU",
		"label": "Frecuencia IMU (Hz)",
		"type": "int",
		"min": 10,
		"max": 500,
		"step": 5,
	},
	"COMPLEMENTARY_FILTER_ALPHA": {
		"section": "IMU",
		"label": "Alpha filtro complementario",
		"type": "float",
		"min": 0.0,
		"max": 1.0,
		"step": 0.01,
	},
	"CALIBRATION_POINTS": {
		"section": "IMU",
		"label": "Puntos de calibración",
		"type": "choice",
		"options": [1, 2],
	},
	"AUTO_CALIB_CAMERA_INDEX": {
		"section": "IMU",
		"label": "Cámara semiautomática (índice)",
		"type": "int",
		"min": 0,
		"max": 10,
		"description": "Índice de cámara utilizado por MediaPipe durante la calibración semiautomática.",
	},
	"AUTO_CALIB_FPS": {
		"section": "IMU",
		"label": "FPS captura semiautomática",
		"type": "int",
		"min": 5,
		"max": 60,
		"description": "Velocidad de captura (cuadros por segundo) para procesamiento de visión.",
	},
	"AUTO_CALIB_REFERENCE_EXT": {
		"section": "IMU",
		"label": "Objetivo extensión (°)",
		"type": "float",
		"min": -20.0,
		"max": 30.0,
		"step": 0.5,
		"description": "Ángulo objetivo detectado por visión para la pierna extendida.",
	},
	"AUTO_CALIB_REFERENCE_FLEX": {
		"section": "IMU",
		"label": "Objetivo flexión (°)",
		"type": "float",
		"min": 40.0,
		"max": 150.0,
		"step": 0.5,
		"description": "Ángulo objetivo detectado por visión para la rodilla flexionada.",
	},
	"AUTO_CALIB_TOLERANCE_DEG": {
		"section": "IMU",
		"label": "Tolerancia detección (°)",
		"type": "float",
		"min": 1.0,
		"max": 20.0,
		"step": 0.5,
		"description": "Margen aceptado entre el ángulo objetivo y el estimado por visión.",
	},
	"AUTO_CALIB_STABILITY_FRAMES": {
		"section": "IMU",
		"label": "Fotogramas de estabilidad",
		"type": "int",
		"min": 3,
		"max": 120,
		"description": "Frames consecutivos requeridos para aceptar un punto objetivo.",
	},
	"AUTO_CALIB_VISIBILITY_THRESHOLD": {
		"section": "IMU",
		"label": "Umbral visibilidad pose",
		"type": "float",
		"min": 0.0,
		"max": 1.0,
		"step": 0.05,
		"description": "Confianza mínima de MediaPipe para considerar válida la pose detectada.",
	},
	"VREF": {
		"section": "ADC",
		"label": "Voltaje de referencia (V)",
		"type": "float",
		"min": 1.0,
		"max": 3.3,
		"step": 0.001,
	},
	"PGA_GAIN": {
		"section": "ADC",
		"label": "Ganancia PGA",
		"type": "int",
		"min": 1,
		"max": 64,
		
	},
	"ADC_RESOLUTION": {
		"section": "ADC",
		"label": "Resolución ADC",
		"type": "int",
		"editable": False,
		"description": "Valor fijo del convertidor ADS1256.",
	},
	"WINDOW_TIME_SEC": {
		"section": "Visualización",
		"label": "Ventana de tiempo (s)",
		"type": "float",
		"min": 1.0,
		"max": 30.0,
		"step": 0.5,
	},
	"UPDATE_FPS": {
		"section": "Visualización",
		"label": "FPS gráficos",
		"type": "int",
		"min": 1,
		"max": 120,
	},
	"UPDATE_INTERVAL_MS": {
		"section": "Visualización",
		"label": "Intervalo de actualización (ms)",
		"type": "derived",
		"editable": False,
	},
	"EMG_BUFFER_SIZE": {
		"section": "Visualización",
		"label": "Buffer EMG",
		"type": "derived",
		"editable": False,
	},
	"IMU_BUFFER_SIZE": {
		"section": "Visualización",
		"label": "Buffer IMU",
		"type": "derived",
		"editable": False,
	},
	"COLOR_CH0": {
		"section": "Visualización",
		"label": "Color EMG CH0",
		"type": "color",
	},
	"COLOR_CH1": {
		"section": "Visualización",
		"label": "Color EMG CH1",
		"type": "color",
	},
	"COLOR_RMS_CH0": {
		"section": "Visualización",
		"label": "Color RMS CH0",
		"type": "color",
	},
	"COLOR_RMS_CH1": {
		"section": "Visualización",
		"label": "Color RMS CH1",
		"type": "color",
	},
	"COLOR_ANGLE": {
		"section": "Visualización",
		"label": "Color ángulo",
		"type": "color",
	},
	"CONFIG_FILE": {
		"section": "Archivos",
		"label": "Archivo de configuración",
		"type": "str",
		"editable": False,
		"description": "Ruta relativa del archivo persistente de configuración.",
	},
	"CALIBRATION_FILE": {
		"section": "Archivos",
		"label": "Archivo de calibración IMU",
		"type": "str",
	},
}

SETTINGS_LAYOUT: List[Tuple[str, List[str]]] = [
	("Conexión Serial", ["SERIAL_BAUD", "SERIAL_TIMEOUT"]),
	("Protocolo", ["PREAMBLE", "FRAME_TYPE_EMG", "FRAME_TYPE_IMU"]),
	(
		"EMG",
		[
			"EMG_FS",
			"EMG_CHANNELS",
			"EMG_HIGHPASS_CUTOFF",
			"EMG_LOWPASS_CUTOFF",
			"EMG_NOTCH_FREQ",
			"EMG_NOTCH_Q",
			"RMS_WINDOW_MS",
			"RMS_WINDOW_SAMPLES",
		],
	),
	("IMU", ["IMU_FS", "COMPLEMENTARY_FILTER_ALPHA", "CALIBRATION_POINTS", "AUTO_CALIB_CAMERA_INDEX", "AUTO_CALIB_FPS", "AUTO_CALIB_REFERENCE_EXT", "AUTO_CALIB_REFERENCE_FLEX", "AUTO_CALIB_TOLERANCE_DEG", "AUTO_CALIB_STABILITY_FRAMES", "AUTO_CALIB_VISIBILITY_THRESHOLD"]),
	("ADC", ["VREF", "PGA_GAIN", "ADC_RESOLUTION"]),
	(
		"Visualización",
		[
			"WINDOW_TIME_SEC",
			"UPDATE_FPS",
			"UPDATE_INTERVAL_MS",
			"EMG_BUFFER_SIZE",
			"IMU_BUFFER_SIZE",
			"COLOR_CH0",
			"COLOR_CH1",
			"COLOR_RMS_CH0",
			"COLOR_RMS_CH1",
			"COLOR_ANGLE",
		],
	),
	("Archivos", ["CONFIG_FILE", "CALIBRATION_FILE"]),
]

_runtime_settings: Dict[str, Any] = {}


def _config_path(settings: Dict[str, Any]) -> Path:
	file_name = settings.get("CONFIG_FILE", DEFAULT_SETTINGS["CONFIG_FILE"])
	return PROJECT_ROOT / file_name


def _normalize_settings(values: Dict[str, Any]) -> Dict[str, Any]:
	normalized = deepcopy(values)
	# Asegurar que los colores y listas se serialicen correctamente
	for key in [
		"COLOR_CH0",
		"COLOR_CH1",
		"COLOR_RMS_CH0",
		"COLOR_RMS_CH1",
		"COLOR_ANGLE",
	]:
		if key in normalized and isinstance(normalized[key], tuple):
			normalized[key] = list(normalized[key])

	if "PREAMBLE" in normalized and isinstance(normalized["PREAMBLE"], (bytes, bytearray)):
		normalized["PREAMBLE"] = list(normalized["PREAMBLE"])

	return normalized


def _compute_derived(settings: Dict[str, Any]) -> None:
	for key, fn in DERIVED_KEYS.items():
		settings[key] = fn(settings)


def _apply_to_module(settings: Dict[str, Any]) -> None:
	for key, value in settings.items():
		if key == "PREAMBLE" and isinstance(value, list):
			globals()[key] = bytes(value)
		elif key.startswith("COLOR_") and isinstance(value, list):
			globals()[key] = tuple(value)
		else:
			globals()[key] = value


def _load_from_disk(settings: Dict[str, Any]) -> Dict[str, Any]:
	path = _config_path(settings)
	if not path.exists():
		return settings

	try:
		with path.open("r", encoding="utf-8") as fh:
			data = json.load(fh)
	except (json.JSONDecodeError, OSError):
		return settings

	for key, value in data.items():
		if key in settings:
			settings[key] = value

	return settings


def load_settings() -> Dict[str, Any]:
	"""Carga los valores desde disco y los aplica al módulo."""
	global _runtime_settings
	base = _normalize_settings(DEFAULT_SETTINGS)
	loaded = _load_from_disk(base)
	_compute_derived(loaded)
	_runtime_settings = loaded
	_apply_to_module(_runtime_settings)
	return deepcopy(_runtime_settings)


def get_settings() -> Dict[str, Any]:
	"""Obtiene una copia de los parámetros en memoria."""
	if not _runtime_settings:
		load_settings()
	return deepcopy(_runtime_settings)


def get_settings_schema() -> Dict[str, Dict[str, Any]]:
	"""Retorna el esquema de metadatos para construcción de UI."""
	return deepcopy(SETTINGS_SCHEMA)


def get_settings_layout() -> List[Tuple[str, List[str]]]:
	"""Retorna la distribución seccional para la interfaz de configuración."""
	return deepcopy(SETTINGS_LAYOUT)


def _coerce_value(key: str, value: Any) -> Any:
	meta = SETTINGS_SCHEMA.get(key, {})
	value_type = meta.get("type")

	if value_type == "int":
		return int(value)
	if value_type == "float":
		return float(value)
	if value_type == "choice":
		return value
	if value_type == "color":
		if isinstance(value, (list, tuple)) and len(value) == 3:
			return [int(v) for v in value]
		raise ValueError(f"{key}: formato de color inválido")
	if value_type == "bytes":
		if isinstance(value, (bytes, bytearray)):
			return list(value)
		if isinstance(value, Iterable):
			return [int(v) & 0xFF for v in value]
		raise ValueError(f"{key}: formato de bytes inválido")
	if value_type == "str":
		return str(value)
	return value


def update_settings(new_values: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
	"""Actualiza valores en memoria y opcionalmente los persiste."""
	if not _runtime_settings:
		load_settings()

	updated = deepcopy(_runtime_settings)
	for key, value in new_values.items():
		if key not in DEFAULT_SETTINGS and key not in DERIVED_KEYS:
			continue
		if key in DERIVED_KEYS:
			continue  # Derived values no se asignan directamente
		try:
			coerced = _coerce_value(key, value)
		except (TypeError, ValueError) as exc:
			raise ValueError(f"Valor inválido para {key}: {exc}") from exc
		updated[key] = coerced

	_compute_derived(updated)
	_apply_to_module(updated)
	_runtime_settings.clear()
	_runtime_settings.update(updated)

	if persist:
		persistable = {
			key: value
			for key, value in _normalize_settings(updated).items()
			if key in DEFAULT_SETTINGS
		}
		path = _config_path(updated)
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("w", encoding="utf-8") as fh:
			json.dump(persistable, fh, indent=4, ensure_ascii=False)

	return deepcopy(_runtime_settings)


def reset_to_defaults(persist: bool = True) -> Dict[str, Any]:
	"""Vuelve a los valores predeterminados."""
	defaults = _normalize_settings(DEFAULT_SETTINGS)
	_compute_derived(defaults)
	_apply_to_module(defaults)
	_runtime_settings.clear()
	_runtime_settings.update(defaults)

	if persist:
		path = _config_path(defaults)
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("w", encoding="utf-8") as fh:
			json.dump(
				{
					key: value
					for key, value in _normalize_settings(defaults).items()
					if key in DEFAULT_SETTINGS
				},
				fh,
				indent=4,
				ensure_ascii=False,
			)

	return deepcopy(_runtime_settings)


# Inicializar los valores al importar el módulo
load_settings()

__all__ = list(DEFAULT_SETTINGS.keys()) + list(DERIVED_KEYS.keys()) + [
	"load_settings",
	"get_settings",
	"get_settings_schema",
	"get_settings_layout",
	"update_settings",
	"reset_to_defaults",
] 