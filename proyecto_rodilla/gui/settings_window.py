"""Ventana para editar los parámetros de configuración del sistema."""
from __future__ import annotations

from typing import Any, Dict, Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import settings as cfg


class ColorInput(QWidget):
    """Editor sencillo para tuplas RGB."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._spins: list[QSpinBox] = []
        for _ in range(3):
            spin = QSpinBox(self)
            spin.setRange(0, 255)
            spin.setFixedWidth(100)
            layout.addWidget(spin)
            self._spins.append(spin)

    def set_value(self, value: Iterable[int]) -> None:
        for spin, val in zip(self._spins, value):
            spin.setValue(int(val))

    def get_value(self) -> tuple[int, int, int]:
        return tuple(spin.value() for spin in self._spins)


class SettingsWindow(QDialog):
    """Diálogo de edición de parámetros de configuración."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Configuración del Sistema")
        self.resize(720, 620)
        self.setModal(True)

        self._schema = cfg.get_settings_schema()
        self._layout = cfg.get_settings_layout()
        self._values = cfg.get_settings()
        self._editors: Dict[str, QWidget] = {}

        self._build_ui()
        self._populate_from_values()

    # ------------------------------------------------------------------
    # Construcción de UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        self.tabs = QTabWidget(self)
        main_layout.addWidget(self.tabs)

        for section, keys in self._layout:
            section_widget = QWidget(self)
            form = QFormLayout(section_widget)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            form.setContentsMargins(8, 8, 8, 8)
            form.setSpacing(12)

            for key in keys:
                widget = self._create_editor(key)
                label_text = self._schema.get(key, {}).get("label", key)
                form.addRow(label_text, widget)

            self.tabs.addTab(section_widget, section)

        button_bar = QHBoxLayout()
        button_bar.addStretch(1)

        self.btn_reset = QPushButton("Restablecer valores")
        self.btn_reset.clicked.connect(self._on_reset)
        button_bar.addWidget(self.btn_reset)

        self.btn_save = QPushButton("Guardar cambios")
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self._on_save)
        button_bar.addWidget(self.btn_save)

        self.btn_close = QPushButton("Cerrar")
        self.btn_close.clicked.connect(self.close)
        button_bar.addWidget(self.btn_close)

        main_layout.addLayout(button_bar)

    def _create_editor(self, key: str) -> QWidget:
        meta = self._schema.get(key, {})
        editable = meta.get("editable", True) and meta.get("type") != "derived"
        value = self._values.get(key)
        description = meta.get("description", "")
        widget: QWidget

        if not editable:
            widget = QLabel(self._format_value(key, value))
            widget.setEnabled(False)
        else:
            value_type = meta.get("type")
            if value_type == "int":
                widget = QSpinBox()
                widget.setRange(int(meta.get("min", -10**9)), int(meta.get("max", 10**9)))
                widget.setSingleStep(int(meta.get("step", 1)))
            elif value_type == "float":
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setRange(float(meta.get("min", -10**6)), float(meta.get("max", 10**6)))
                widget.setSingleStep(float(meta.get("step", 0.1)))
            elif value_type == "choice":
                widget = QComboBox()
                for option in meta.get("options", []):
                    widget.addItem(str(option), option)
            elif value_type == "color":
                widget = ColorInput()
            elif value_type == "bytes":
                widget = QLineEdit(self._format_bytes(value))
                widget.setPlaceholderText("A5 5A")
            else:
                widget = QLineEdit(str(value) if value is not None else "")

        if description:
            widget.setToolTip(description)

        self._editors[key] = widget
        return widget

    # ------------------------------------------------------------------
    # Sincronización valores <-> UI
    # ------------------------------------------------------------------
    def _populate_from_values(self) -> None:
        for key, widget in self._editors.items():
            value = self._values.get(key)
            self._set_widget_value(key, widget, value)

    def _set_widget_value(self, key: str, widget: QWidget, value: Any) -> None:
        meta = self._schema.get(key, {})
        if isinstance(widget, QLabel):
            widget.setText(self._format_value(key, value))
            return
        if isinstance(widget, QSpinBox):
            widget.setValue(int(value))
            return
        if isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
            return
        if isinstance(widget, QComboBox):
            idx = widget.findData(value)
            if idx < 0:
                idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
            return
        if isinstance(widget, ColorInput):
            widget.set_value(value or (0, 0, 0))
            return
        if isinstance(widget, QLineEdit):
            if meta.get("type") == "bytes":
                widget.setText(self._format_bytes(value))
            else:
                widget.setText("" if value is None else str(value))

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------
    def _on_save(self) -> None:
        try:
            updates = self._collect_updates()
        except ValueError as exc:
            QMessageBox.warning(self, "Valor inválido", str(exc))
            return

        if not updates:
            QMessageBox.information(self, "Configuración", "No hay cambios por guardar.")
            return

        try:
            cfg.update_settings(updates)
        except ValueError as exc:
            QMessageBox.warning(self, "Configuración", str(exc))
            return

        self._values = cfg.get_settings()
        self._populate_from_values()
        QMessageBox.information(
            self,
            "Configuración",
            "Parámetros guardados correctamente. Reinicia los módulos abiertos para aplicar cambios.",
        )

    def _on_reset(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Restablecer configuración",
            "¿Deseas volver a los valores predeterminados?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        cfg.reset_to_defaults()
        self._values = cfg.get_settings()
        self._populate_from_values()

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def _collect_updates(self) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        for key, widget in self._editors.items():
            meta = self._schema.get(key, {})
            if isinstance(widget, QLabel):
                continue
            if meta.get("type") == "derived" or not meta.get("editable", True):
                continue
            value = self._extract_value(widget, meta)
            if value is None and meta.get("type") == "bytes":
                raise ValueError(f"{meta.get('label', key)} requiere al menos un byte")
            if value is not None and value != self._values.get(key):
                updates[key] = value
        return updates

    def _extract_value(self, widget: QWidget, meta: Dict[str, Any]) -> Any:
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return data if data is not None else widget.currentText()
        if isinstance(widget, ColorInput):
            return widget.get_value()
        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            if meta.get("type") == "bytes":
                return self._parse_bytes(text) if text else None
            return text
        return None

    def _format_value(self, key: str, value: Any) -> str:
        meta = self._schema.get(key, {})
        value_type = meta.get("type")
        if value_type == "color":
            rgb = value or (0, 0, 0)
            return f"{rgb[0]}, {rgb[1]}, {rgb[2]}"
        if value_type == "bytes":
            return self._format_bytes(value)
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def _format_bytes(value: Any) -> str:
        if isinstance(value, (bytes, bytearray)):
            data = list(value)
        elif isinstance(value, Iterable):
            data = list(value)
        else:
            return ""
        return " ".join(f"{int(b) & 0xFF:02X}" for b in data)

    @staticmethod
    def _parse_bytes(text: str) -> list[int]:
        if not text:
            return []
        parts = text.replace(",", " ").split()
        result: list[int] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("0x"):
                value = int(part, 16)
            else:
                value = int(part, 16) if all(c in "0123456789abcdefABCDEF" for c in part) else int(part)
            result.append(value & 0xFF)
        if not result:
            raise ValueError("Ingresa al menos un byte en formato hexadecimal")
        return result