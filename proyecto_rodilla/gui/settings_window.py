"""Ventana para editar los parámetros de configuración del sistema."""
from __future__ import annotations

import sys
from typing import Any, Dict, Iterable

from PyQt6.QtCore import Qt, pyqtSignal
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
    QColorDialog,
)
from PyQt6.QtGui import QColor

from config import settings as cfg


class ColorPicker(QWidget):
    """Selector de color con vista previa y diálogo del sistema."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._color: tuple[int, int, int] = (0, 0, 0)

        self._preview = QLabel()
        self._preview.setFixedSize(58, 24)
        self._preview.setStyleSheet("border: 1px solid #1F1F21; border-radius: 4px;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._preview)

        self._button = QPushButton("Seleccionar color")
        self._button.setMinimumWidth(140)
        self._button.clicked.connect(self._open_dialog)
        layout.addWidget(self._button)

        layout.addStretch(1)

        self._update_preview()

    def set_value(self, value: Iterable[int] | None) -> None:
        if value is None:
            self._color = (0, 0, 0)
        else:
            rgb = tuple(int(v) for v in value)
            if len(rgb) != 3:
                raise ValueError("El color debe tener formato RGB")
            self._color = tuple(max(0, min(255, v)) for v in rgb)
        self._update_preview()

    def get_value(self) -> tuple[int, int, int]:
        return self._color

    # ------------------------------------------------------------------
    def _open_dialog(self) -> None:
        initial = QColor(*self._color)
        color = QColorDialog.getColor(initial, self, "Seleccionar color")
        if color.isValid():
            self._color = (color.red(), color.green(), color.blue())
            self._update_preview()

    def _update_preview(self) -> None:
        r, g, b = self._color
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "#0E0D0D" if brightness > 150 else "#E4E4E4"
        self._preview.setStyleSheet(
            "border: 1px solid #1F1F21; border-radius: 4px; "
            f"background-color: rgb({r}, {g}, {b}); color: {text_color};"
        )
        self._preview.setText(f"#{r:02X}{g:02X}{b:02X}")


class SettingsWindow(QDialog):
    """Diálogo de edición de parámetros de configuración."""

    settings_applied = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Configuración del Sistema")
        self.resize(720, 620)
        self.setModal(True)
        self._apply_styles()

        self._schema = cfg.get_settings_schema()
        self._layout = cfg.get_settings_layout()
        self._values = cfg.get_settings()
        self._editors: Dict[str, QWidget] = {}
        self._camera_entries: list[tuple[int, str]] = []
        self._camera_scan_error: str | None = None

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
        self.btn_reset.setProperty("category", "secondary")
        self.btn_reset.clicked.connect(self._on_reset)
        button_bar.addWidget(self.btn_reset)

        self.btn_save = QPushButton("Guardar cambios")
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self._on_save)
        button_bar.addWidget(self.btn_save)

        self.btn_close = QPushButton("Cerrar")
        self.btn_close.setProperty("category", "secondary")
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
            if key == "AUTO_CALIB_CAMERA_INDEX":
                widget = self._create_camera_selector(value)
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
                    widget = ColorPicker()
                elif value_type == "bytes":
                    widget = QLineEdit(self._format_bytes(value))
                    widget.setPlaceholderText("A5 5A")
                else:
                    widget = QLineEdit(str(value) if value is not None else "")

        if description:
            widget.setToolTip(description)

        widget.setProperty("config_key", key)

        self._editors[key] = widget
        return widget

    def _create_camera_selector(self, current_value: Any) -> QComboBox:
        combo = QComboBox()
        self._camera_entries, self._camera_scan_error = self._discover_cameras()

        if self._camera_entries:
            for index, label in self._camera_entries:
                combo.addItem(f"{label} (#{index})", index)
        else:
            placeholder = self._camera_scan_error or "No se detectaron cámaras disponibles"
            combo.addItem(placeholder, None)
            model = combo.model()
            if hasattr(model, "item"):
                placeholder_item = model.item(0)
                if placeholder_item is not None:
                    placeholder_item.setEnabled(False)

        if current_value is not None and combo.findData(current_value) < 0:
            combo.addItem(f"Índice configurado ({current_value})", current_value)

        if self._camera_scan_error:
            combo.setToolTip(self._camera_scan_error)

        return combo

    def _discover_cameras(self, limit: int = 8) -> tuple[list[tuple[int, str]], str | None]:
        try:
            import cv2  # type: ignore
        except ImportError:
            return [], "Instala 'opencv-python' para detectar cámaras automáticamente."

        cameras: list[tuple[int, str]] = []
        backend = getattr(cv2, "CAP_DSHOW", None) if sys.platform.startswith("win") else None

        for index in range(limit):
            if backend is not None:
                capture = cv2.VideoCapture(index, backend)
            else:
                capture = cv2.VideoCapture(index)

            if not capture or not capture.isOpened():
                if capture:
                    capture.release()
                continue

            cameras.append((index, f"Cámara {index}"))
            capture.release()

        if not cameras:
            return [], "No se detectaron cámaras disponibles"
        return cameras, None

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
            config_key = widget.property("config_key")
            idx = widget.findData(value)
            if idx < 0:
                idx = widget.findText(str(value))
            if idx < 0 and config_key == "AUTO_CALIB_CAMERA_INDEX" and value is not None:
                widget.addItem(f"Índice configurado ({value})", value)
                idx = widget.count() - 1
            if idx >= 0:
                widget.setCurrentIndex(idx)
            return
        if isinstance(widget, ColorPicker):
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

        self.settings_applied.emit()
        self.accept()

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
        if isinstance(widget, ColorPicker):
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

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1C1C1E;
                color: #E4E4E4;
                font-family: "Avenir";
                font-size: 10.5pt;
            }
            QLabel {
                color: #D9E4E4;
                font-size: 10.5pt;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2C2C2E;
                border: 1px solid #3A3A3C;
                border-radius: 5px;
                padding: 6px 8px;
                color: #F5F5F7;
                selection-background-color: #32598C;
                selection-color: #FFFFFF;
            }
            QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
                background-color: #2A2A2A;
                color: #767676;
                border-color: #3F3F41;
            }
            QTabWidget::pane {
                border: 1px solid #323234;
                border-radius: 6px;
                margin-top: 8px;
            }
            QTabBar::tab {
                background: #29292B;
                color: #CCCCD0;
                border: 1px solid #323234;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #32598C;
                color: #FFFFFF;
            }
            QTabBar::tab:hover {
                background: #233E62;
            }
            QPushButton {
                background-color: #32598C;
                color: #FFFFFF;
                font-weight: 600;
                padding: 10px 18px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #233E62;
            }
            QPushButton:pressed {
                background-color: #1A2D47;
            }
            QPushButton[category="secondary"] {
                background-color: #3A3A3C;
                color: #D0D0D5;
                font-weight: 500;
            }
            QPushButton[category="secondary"]:hover {
                background-color: #4A4A4D;
            }
            QPushButton[category="secondary"]:pressed {
                background-color: #2F2F31;
            }
            QMessageBox {
                background-color: #1C1C1E;
                font-family: "Avenir";
                font-size: 9.5pt;
            }
            QMessageBox QLabel {
                min-width: 240px;
                font-size: 9.5pt;
            }
            QMessageBox QPushButton {
                padding: 3px 6px;
                font-size: 9.5pt;
            }
        """
        )