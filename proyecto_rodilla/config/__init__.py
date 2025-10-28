"""
Paquete de configuración
"""
from . import settings as settings
from .settings import *  # noqa: F401,F403

__all__ = list(settings.__all__) + ["settings"]