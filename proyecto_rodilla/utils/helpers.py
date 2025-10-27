"""
Funciones auxiliares
"""
import json
from pathlib import Path


def save_json(data: dict, filepath: str):
    """Guarda diccionario en archivo JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> dict:
    """Carga diccionario desde archivo JSON."""
    if not Path(filepath).exists():
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)