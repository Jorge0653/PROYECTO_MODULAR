"""Diálogo para normalizar EMG mediante MVC."""
from __future__ import annotations

from typing import Dict, Optional

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class EMGNormalizationDialog(QDialog):
    """Gestiona capturas de MVC para normalizar canales EMG."""

    mvc_computed = QtCore.pyqtSignal(int, float)

    def __init__(self, mvc_values: Optional[Dict[int, Optional[float]]] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Normalizar EMG (MVC)")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._mvc_values: Dict[int, Optional[float]] = {0: None, 1: None}
        if mvc_values:
            self._mvc_values.update(mvc_values)

        self._current_rms: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._active_channel: Optional[int] = None
        self._remaining_ms: int = 0
        self._current_max: float = 0.0

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_measure_tick)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        description = QLabel(
            "Realiza una Contracción Voluntaria Máxima (MVC) para cada canal."
            " El sistema tomará el valor pico de RMS durante 3 segundos."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        channels_box = QGroupBox("Canales EMG")
        channels_layout = QGridLayout()
        channels_layout.setHorizontalSpacing(14)
        channels_layout.setVerticalSpacing(10)
        channels_box.setLayout(channels_layout)
        layout.addWidget(channels_box)

        self._channel_labels: Dict[int, QLabel] = {}
        self._channel_buttons: Dict[int, QPushButton] = {}

        for idx in (0, 1):
            header = QLabel(f"Canal {idx}")
            header.setStyleSheet("font-weight: 600;")
            channels_layout.addWidget(header, idx, 0)

            value_label = QLabel(self._format_mvc_value(idx))
            self._channel_labels[idx] = value_label
            channels_layout.addWidget(value_label, idx, 1)

            button = QPushButton("Medir MVC (3 s)")
            button.clicked.connect(lambda _, ch=idx: self._start_measurement(ch))
            self._channel_buttons[idx] = button
            channels_layout.addWidget(button, idx, 2)

        self.status_label = QLabel("Selecciona un canal para iniciar.")
        self.current_value_label = QLabel("RMS actual: -- mV | Pico: -- mV")
        font = self.current_value_label.font()
        font.setPointSize(font.pointSize() + 1)
        self.current_value_label.setFont(font)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.current_value_label)
        layout.addLayout(info_layout)

        footer = QHBoxLayout()
        footer.addStretch(1)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.close)
        footer.addWidget(btn_close)
        layout.addLayout(footer)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def set_current_rms(self, ch0: float, ch1: float) -> None:
        """Recibe los valores RMS actuales para alimentar la medición."""
        self._current_rms[0] = ch0
        self._current_rms[1] = ch1

        if self._active_channel is not None and self._remaining_ms > 0:
            current = self._current_rms[self._active_channel]
            self._current_max = max(self._current_max, abs(current))
            self._update_current_value_label(current)

    # ------------------------------------------------------------------
    # Lógica de medición
    # ------------------------------------------------------------------
    def _start_measurement(self, channel: int) -> None:
        if self._active_channel is not None:
            QMessageBox.information(
                self,
                "Medición en curso",
                "Espera a que termine la medición actual para iniciar otra.",
            )
            return

        self._active_channel = channel
        self._remaining_ms = 3000
        self._current_max = 0.0
        self._timer.start()

        self.status_label.setText(
            f"Canal {channel}: realiza la contracción máxima. Tiempo restante: 3.0 s"
        )
        self._update_current_value_label(self._current_rms[channel])
        self._channel_buttons[channel].setEnabled(False)

    def _on_measure_tick(self) -> None:
        if self._active_channel is None:
            self._timer.stop()
            return

        self._remaining_ms = max(0, self._remaining_ms - self._timer.interval())
        seconds = self._remaining_ms / 1000.0
        self.status_label.setText(
            f"Canal {self._active_channel}: mantiene la contracción. Tiempo restante: {seconds:.1f} s"
        )

        if self._remaining_ms == 0:
            self._finalize_measurement()

    def _finalize_measurement(self) -> None:
        channel = self._active_channel
        if channel is None:
            return

        self._timer.stop()
        peak = self._current_max if self._current_max > 0 else abs(self._current_rms[channel])
        self._mvc_values[channel] = peak if peak > 0 else None
        self._channel_labels[channel].setText(self._format_mvc_value(channel))
        self._channel_buttons[channel].setEnabled(True)

        self.status_label.setText("Medición finalizada. Puedes repetir si lo deseas.")
        self.current_value_label.setText(
            f"RMS actual: -- mV | Pico: {peak * 1000:.3f} mV" if peak else "No se detectó señal"
        )

        if peak:
            self.mvc_computed.emit(channel, peak)
        else:
            QMessageBox.warning(
                self,
                "Sin señal",
                "No se detectó actividad suficiente durante la medición.",
            )

        self._active_channel = None
        self._remaining_ms = 0
        self._current_max = 0.0

    def closeEvent(self, event):  # pragma: no cover - interacción UI
        """Detiene temporizadores si el diálogo se cierra abruptamente."""
        if self._timer.isActive():
            self._timer.stop()
        self._active_channel = None
        self._remaining_ms = 0
        self._current_max = 0.0
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def _format_mvc_value(self, channel: int) -> str:
        value = self._mvc_values.get(channel)
        return f"MVC actual: {value * 1000:.3f} mV" if value else "MVC actual: -- mV"

    def _update_current_value_label(self, current_value: float) -> None:
        self.current_value_label.setText(
            f"RMS actual: {current_value * 1000:.3f} mV | Pico: {self._current_max * 1000:.3f} mV"
        )
