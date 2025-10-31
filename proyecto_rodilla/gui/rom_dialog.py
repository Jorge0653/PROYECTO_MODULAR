"""Diálogo para capturar el rango de movimiento (ROM) de la rodilla."""
from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class ROMDialog(QDialog):
    """Permite registrar extensión y flexión máximas para calcular el ROM."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Medir ROM")
        self.setModal(True)
        self.setMinimumWidth(360)

        self._current_angle: float = 0.0
        self._stage: str = "extend"  # extend -> flex
        self._min_angle: Optional[float] = None
        self._max_angle: Optional[float] = None
        self._extension_angle: Optional[float] = None
        self._flexion_angle: Optional[float] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        instructions = QLabel(
            "Para calcular el ROM:")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        step_label = QLabel(
            "1. Extiende la rodilla al máximo y presiona \"Registrar extensión\".\n"
            "2. Flexiona la rodilla al máximo y presiona \"Registrar flexión\"."
        )
        step_label.setWordWrap(True)
        layout.addWidget(step_label)

        self.stage_message = QLabel("Extiende la rodilla al máximo.")
        self.stage_message.setWordWrap(True)
        layout.addWidget(self.stage_message)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        layout.addLayout(grid)

        current_title = QLabel("Ángulo actual:")
        self.current_angle_label = QLabel("--°")
        self.current_angle_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        grid.addWidget(current_title, 0, 0)
        grid.addWidget(self.current_angle_label, 0, 1)

        self.extension_value_label = QLabel("Extensión registrada: --°")
        self.flexion_value_label = QLabel("Flexión registrada: --°")
        grid.addWidget(self.extension_value_label, 1, 0, 1, 2)
        grid.addWidget(self.flexion_value_label, 2, 0, 1, 2)

        buttons_row = QGridLayout()
        buttons_row.setHorizontalSpacing(10)
        layout.addLayout(buttons_row)

        self.btn_extension = QPushButton("Registrar extensión")
        self.btn_extension.clicked.connect(self._capture_extension)
        buttons_row.addWidget(self.btn_extension, 0, 0)

        self.btn_flexion = QPushButton("Registrar flexión")
        self.btn_flexion.clicked.connect(self._capture_flexion)
        self.btn_flexion.setEnabled(False)
        buttons_row.addWidget(self.btn_flexion, 0, 1)

        self.btn_reset = QPushButton("Reiniciar")
        self.btn_reset.clicked.connect(self._reset_measurement)
        buttons_row.addWidget(self.btn_reset, 0, 2)

        self.summary_label = QLabel("ROM actual: --°")
        self.summary_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.summary_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def set_current_angle(self, angle: float) -> None:
        """Actualiza el ángulo actual recibido desde la ventana principal."""
        self._current_angle = angle
        self.current_angle_label.setText(f"{angle:.1f}°")

        if self._stage == "extend":
            self._min_angle = angle if self._min_angle is None else min(self._min_angle, angle)
        elif self._stage == "flex":
            self._max_angle = angle if self._max_angle is None else max(self._max_angle, angle)

    def get_rom_value(self) -> Optional[float]:
        """Retorna el ROM calculado si ambas capturas fueron realizadas."""
        if self._extension_angle is None or self._flexion_angle is None:
            return None
        return abs(self._flexion_angle - self._extension_angle)

    def get_measurements(self) -> tuple[Optional[float], Optional[float]]:
        """Devuelve las mediciones crudas (extensión, flexión)."""
        return self._extension_angle, self._flexion_angle

    # ------------------------------------------------------------------
    # Slots internos
    # ------------------------------------------------------------------
    def _capture_extension(self) -> None:
        if self._min_angle is None:
            self._min_angle = self._current_angle
        self._extension_angle = self._min_angle
        self.extension_value_label.setText(f"Extensión registrada: {self._extension_angle:.1f}°")

        self._stage = "flex"
        self._max_angle = self._current_angle
        self.stage_message.setText("Flexiona la rodilla al máximo.")

        self.btn_extension.setEnabled(False)
        self.btn_flexion.setEnabled(True)

    def _capture_flexion(self) -> None:
        if self._extension_angle is None:
            QMessageBox.warning(self, "Acción inválida", "Registra primero la extensión.")
            return

        if self._max_angle is None:
            self._max_angle = self._current_angle
        self._flexion_angle = self._max_angle
        self.flexion_value_label.setText(f"Flexión registrada: {self._flexion_angle:.1f}°")

        rom = self.get_rom_value()
        if rom is not None:
            self.summary_label.setText(f"ROM actual: {rom:.1f}°")

        self._stage = "done"
        self.stage_message.setText("ROM capturado. Puedes aceptar o reiniciar.")

        self.btn_flexion.setEnabled(False)
        self.btn_extension.setEnabled(True)

    def _reset_measurement(self) -> None:
        self._stage = "extend"
        self._min_angle = None
        self._max_angle = None
        self._extension_angle = None
        self._flexion_angle = None

        self.extension_value_label.setText("Extensión registrada: --°")
        self.flexion_value_label.setText("Flexión registrada: --°")
        self.summary_label.setText("ROM actual: --°")
        self.stage_message.setText("Extiende la rodilla al máximo.")

        self.btn_extension.setEnabled(True)
        self.btn_flexion.setEnabled(False)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def accept(self) -> None:  # pragma: no cover - lógica UI
        if self._extension_angle is None or self._flexion_angle is None:
            QMessageBox.warning(
                self,
                "Medición incompleta",
                "Registra la extensión y la flexión para calcular el ROM.",
            )
            return
        super().accept()
