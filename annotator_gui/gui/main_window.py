# ============================================================================
# gui/main_window.py
# ============================================================================
""""
Main window for the tissue annotation application
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QComboBox,
                             QRadioButton, QButtonGroup, QFileDialog,
                             QMessageBox, QGroupBox, QLineEdit, QDialog)
from PyQt5.QtCore import Qt
import numpy as np
from datetime import datetime

from gui.canvas import ImageCanvas
from core.data_manager import DataManager


class AddMaskTypeDialog(QDialog):
    """Dialog for adding a new mask type"""

    def __init__(self, parent=None, existing_types=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Mask Type")
        self.setGeometry(200, 200, 300, 150)
        self.existing_types = existing_types or []
        self.result_text = None

        self.init_ui()

    def init_ui(self):
        """Initialize dialog UI"""
        layout = QVBoxLayout()

        label = QLabel("Enter mask type name:")
        layout.addWidget(label)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("e.g., lesion, artifact, etc.")
        layout.addWidget(self.text_input)

        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_input)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def accept_input(self):
        """Validate and accept input"""
        text = self.text_input.text().strip()

        if not text:
            QMessageBox.warning(self, "Error", "Mask type name cannot be empty")
            return

        if text in self.existing_types:
            QMessageBox.warning(self, "Error", f"Mask type '{text}' already exists")
            return

        if not text.replace('_', '').isalnum():
            QMessageBox.warning(self, "Error", "Mask type name can only contain letters, numbers, and underscores")
            return

        self.result_text = text
        self.accept()

    def get_result(self):
        """Get the entered mask type"""
        return self.result_text


class TissueAnnotatorGUI(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tissue Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.setAcceptDrops(True)

        self.data_manager = DataManager()
        self.canvas = None
        self.param_combo = None
        self.mask_type_combo = None
        self.label_stats = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        self.canvas = ImageCanvas(self)
        main_layout.addWidget(self.canvas, stretch=4)

        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

    def _create_control_panel(self):
        """Create the control panel widget"""
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(300)

        control_layout.addWidget(self._create_file_group())
        control_layout.addWidget(self._create_mask_type_group())
        control_layout.addWidget(self._create_parameter_group())
        control_layout.addWidget(self._create_mode_group())
        control_layout.addWidget(self._create_mask_controls_group())
        control_layout.addWidget(self._create_statistics_group())
        control_layout.addWidget(self._create_instructions_group())
        control_layout.addStretch()

        return control_panel

    def _create_file_group(self):
        """Create file control group"""
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()

        drop_label = QLabel("Drag & drop NPZ file here\nor use the button below")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                background-color: #f0f0f0;
            }
        """)
        file_layout.addWidget(drop_label)

        self.btn_load = QPushButton("Load NPZ File")
        self.btn_load.clicked.connect(self.load_file)
        file_layout.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save Masks")
        self.btn_save.clicked.connect(self.save_mask)
        self.btn_save.setEnabled(False)
        file_layout.addWidget(self.btn_save)

        file_group.setLayout(file_layout)
        return file_group

    def _create_mask_type_group(self):
        """Create mask type selection group"""
        mask_type_group = QGroupBox("Mask Type")
        mask_type_layout = QVBoxLayout()

        self.mask_type_combo = QComboBox()
        self.mask_type_combo.currentTextChanged.connect(self.change_mask_type)
        mask_type_layout.addWidget(self.mask_type_combo)

        self.btn_add_mask_type = QPushButton("Add New Mask Type")
        self.btn_add_mask_type.clicked.connect(self.add_mask_type)
        self.btn_add_mask_type.setEnabled(False)
        mask_type_layout.addWidget(self.btn_add_mask_type)

        mask_type_group.setLayout(mask_type_layout)
        return mask_type_group

    def _create_parameter_group(self):
        """Create parameter selection group"""
        param_group = QGroupBox("Parameter")
        param_layout = QVBoxLayout()

        self.param_combo = QComboBox()
        self.param_combo.currentTextChanged.connect(self.change_parameter)
        param_layout.addWidget(self.param_combo)

        param_group.setLayout(param_layout)
        return param_group

    def _create_mode_group(self):
        """Create annotation mode selection group"""
        mode_group = QGroupBox("Annotation Mode")
        mode_layout = QVBoxLayout()

        self.mode_group = QButtonGroup()
        self.radio_polygon = QRadioButton("Polygon")
        self.radio_freehand = QRadioButton("Freehand")
        self.radio_polygon.setChecked(True)

        self.mode_group.addButton(self.radio_polygon)
        self.mode_group.addButton(self.radio_freehand)

        self.radio_polygon.toggled.connect(self.change_mode)

        mode_layout.addWidget(self.radio_polygon)
        mode_layout.addWidget(self.radio_freehand)

        mode_group.setLayout(mode_layout)
        return mode_group

    def _create_mask_controls_group(self):
        """Create mask control group"""
        mask_group = QGroupBox("Mask Controls")
        mask_layout = QVBoxLayout()

        self.btn_toggle = QPushButton("Toggle Mask Visibility")
        self.btn_toggle.clicked.connect(self.toggle_mask)
        mask_layout.addWidget(self.btn_toggle)

        opacity_label = QLabel("Mask Opacity:")
        mask_layout.addWidget(opacity_label)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.change_opacity)
        mask_layout.addWidget(self.opacity_slider)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo_last)
        self.btn_undo.setEnabled(False)
        mask_layout.addWidget(self.btn_undo)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_mask)
        self.btn_clear.setEnabled(False)
        mask_layout.addWidget(self.btn_clear)

        mask_group.setLayout(mask_layout)
        return mask_group

    def _create_statistics_group(self):
        """Create statistics display group"""
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()

        self.label_stats = QLabel("No data loaded")
        self.label_stats.setWordWrap(True)
        stats_layout.addWidget(self.label_stats)

        stats_group.setLayout(stats_layout)
        return stats_group

    def _create_instructions_group(self):
        """Create instructions group"""
        instr_group = QGroupBox("Instructions")
        instr_layout = QVBoxLayout()

        instructions = QLabel(
            "Polygon Mode:\n"
            "- Click to add vertices\n"
            "- Click near first point to close\n"
            "- Undo button: remove last vertex\n"
            "- Backspace/Delete: remove last vertex\n"
            "- Enter: close polygon\n"
            "- Escape: cancel drawing\n\n"
            "Freehand Mode:\n"
            "- Click and drag to draw\n"
            "- Release to close region"
        )
        instructions.setWordWrap(True)
        instr_layout.addWidget(instructions)

        instr_group.setLayout(instr_layout)
        return instr_group

    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].toLocalFile().endswith('.npz'):
                event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle file drop event"""
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            if filepath.endswith('.npz'):
                self._load_file_internal(filepath)

    def load_file(self):
        """Load NPZ file via file dialog"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select NPZ File", "", "NPZ Files (*.npz)"
        )

        if filepath:
            self._load_file_internal(filepath)

    def _load_file_internal(self, filepath):
        """Internal method to load NPZ file"""
        try:
            success, message = self.data_manager.load_file(filepath)

            if not success:
                QMessageBox.warning(self, "Error", message)
                return

            self.param_combo.clear()
            self.param_combo.addItems(self.data_manager.param_names)

            self.mask_type_combo.clear()
            self.mask_type_combo.addItems(self.data_manager.available_mask_types)

            self.btn_save.setEnabled(True)
            self.btn_undo.setEnabled(True)
            self.btn_clear.setEnabled(True)
            self.btn_add_mask_type.setEnabled(True)

            self.change_parameter(self.data_manager.param_names[0])

            filename = filepath.split('/')[-1].split('\\')[-1]
            self.setWindowTitle(f"Tissue Annotation Tool - {filename}")

            if "existing mask" in message.lower():
                QMessageBox.information(self, "Info", message)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def add_mask_type(self):
        """Add a new mask type"""
        dialog = AddMaskTypeDialog(self, self.data_manager.available_mask_types)
        if dialog.exec_() == QDialog.Accepted:
            new_type = dialog.get_result()
            if new_type:
                success = self.data_manager.add_mask_type(new_type)
                if success:
                    self.mask_type_combo.addItem(new_type)
                    self.mask_type_combo.setCurrentText(new_type)
                    QMessageBox.information(self, "Success", f"Mask type '{new_type}' added successfully")
                else:
                    QMessageBox.warning(self, "Error", f"Mask type '{new_type}' already exists")

    def change_mask_type(self, mask_type):
        """Switch to a different mask type"""
        if mask_type and self.data_manager.set_current_mask_type(mask_type):
            if self.data_manager.current_param:
                self.canvas.set_mask(self.data_manager.get_current_mask())
                self.update_statistics()

    def change_parameter(self, param_name):
        """Change displayed parameter"""
        if param_name and param_name in self.data_manager.data_dict:
            self.data_manager.current_param = param_name
            image_data = self.data_manager.data_dict[param_name]
            self.canvas.set_image(image_data)
            mask = self.data_manager.get_current_mask()
            if mask is not None:
                self.canvas.set_mask(mask)
            self.update_statistics()

    def change_mode(self):
        """Change annotation mode"""
        if self.radio_polygon.isChecked():
            self.canvas.mode = 'polygon'
        else:
            self.canvas.mode = 'freehand'
        self.canvas.clear_current_drawing()

    def toggle_mask(self):
        """Toggle mask visibility"""
        self.canvas.show_mask = not self.canvas.show_mask
        self.canvas.update()

    def change_opacity(self, value):
        """Change mask opacity"""
        self.canvas.mask_opacity = value / 100.0
        self.canvas.update()

    def save_mask_history(self):
        """Save current mask state for undo"""
        if self.canvas.mask is not None:
            self.data_manager.masks[self.data_manager.current_mask_type] = self.canvas.mask
            self.data_manager.save_mask_history()

    def undo_last(self):
        """Undo last operation"""
        if self.canvas.polygon_points:
            self.canvas.polygon_points.pop()
            self.canvas.update()
            return

        if self.data_manager.undo():
            mask = self.data_manager.get_current_mask()
            self.canvas.set_mask(mask)
            self.canvas.clear_current_drawing()
            self.update_statistics()
        else:
            QMessageBox.information(self, "Undo", "Nothing to undo")

    def clear_mask(self):
        """Clear all annotations for current mask type"""
        reply = QMessageBox.question(
            self, "Clear All",
            f"Are you sure you want to clear all annotations for '{self.data_manager.current_mask_type}' mask?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.data_manager.clear_mask()
            mask = self.data_manager.get_current_mask()
            self.canvas.set_mask(mask)
            self.canvas.clear_current_drawing()
            self.update_statistics()

    def save_mask(self):
        """Save all masks to NPZ file"""
        self.data_manager.masks[self.data_manager.current_mask_type] = self.canvas.mask

        success, message = self.data_manager.save_mask()

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)

    def update_statistics(self):
        """Update statistics display"""
        mask = self.data_manager.get_current_mask()
        if mask is not None:
            self.data_manager.masks[self.data_manager.current_mask_type] = self.canvas.mask

        stats = self.data_manager.get_statistics()
        self.label_stats.setText(stats)