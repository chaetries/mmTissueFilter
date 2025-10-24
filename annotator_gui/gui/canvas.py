# ============================================================================
# gui/canvas.py
# ============================================================================
"""
Canvas widget for image display and annotation
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPainter, QPen, QColor
import numpy as np
from matplotlib.path import Path


class ImageCanvas(QWidget):
    """Custom widget for displaying and annotating images"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMinimumSize(800, 600)

        # Image data
        self.image = None
        self.mask = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)

        # Drawing state
        self.polygon_points = []
        self.freehand_points = []
        self.is_drawing = False
        self.mode = 'polygon'

        # Display settings
        self.show_mask = True
        self.mask_opacity = 0.5

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_image(self, image_array):
        """Convert numpy array to QImage and display"""
        img_normalized = self._normalize_image(image_array)
        img_normalized = np.ascontiguousarray(img_normalized)

        height, width = img_normalized.shape
        bytes_per_line = width

        self.image = QImage(img_normalized.tobytes(), width, height,
                           bytes_per_line, QImage.Format_Grayscale8)
        self._image_data = img_normalized
        self.update()

    def _normalize_image(self, image_array):
        """Normalize image array to 0-255 range"""
        img_min = image_array.min()
        img_max = image_array.max()

        if img_max > img_min:
            normalized = ((image_array - img_min) / (img_max - img_min) * 255)
        else:
            normalized = np.zeros_like(image_array)

        return normalized.astype(np.uint8)

    def set_mask(self, mask_array):
        """Set the annotation mask"""
        self.mask = mask_array.copy()
        self.update()

    def clear_current_drawing(self):
        """Clear current polygon/freehand path"""
        self.polygon_points = []
        self.freehand_points = []
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton and self.image:
            img_point = self.screen_to_image(event.pos())

            if self.mode == 'polygon':
                self._handle_polygon_click(img_point)
            elif self.mode == 'freehand':
                self._start_freehand_drawing(img_point)

    def _handle_polygon_click(self, img_point):
        """Handle polygon mode click"""
        if not (0 <= img_point.x() < self.mask.shape[1] and
                0 <= img_point.y() < self.mask.shape[0]):
            return

        self.polygon_points.append(img_point)

        if len(self.polygon_points) > 2:
            first = self.polygon_points[0]
            dist = np.sqrt((img_point.x() - first.x())**2 +
                         (img_point.y() - first.y())**2)
            if dist < 10 / self.scale:
                self.close_polygon()
                return

        self.update()

    def _start_freehand_drawing(self, img_point):
        """Start freehand drawing"""
        self.is_drawing = True
        self.freehand_points = [img_point]

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.is_drawing and self.mode == 'freehand' and self.image:
            img_point = self.screen_to_image(event.pos())
            self.freehand_points.append(img_point)
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            if len(self.freehand_points) > 2:
                self.close_polygon()

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if len(self.polygon_points) > 2:
                self.close_polygon()
        elif event.key() == Qt.Key_Escape:
            self.clear_current_drawing()
        elif event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            if self.polygon_points:
                self.polygon_points.pop()
                self.update()

    def screen_to_image(self, screen_point):
        """Convert screen coordinates to image coordinates"""
        x = int((screen_point.x() - self.offset.x()) / self.scale)
        y = int((screen_point.y() - self.offset.y()) / self.scale)
        return QPoint(x, y)

    def image_to_screen(self, img_point):
        """Convert image coordinates to screen coordinates"""
        x = int(img_point.x() * self.scale + self.offset.x())
        y = int(img_point.y() * self.scale + self.offset.y())
        return QPoint(x, y)

    def close_polygon(self):
        """Fill the polygon in the mask"""
        points = self.polygon_points if self.mode == 'polygon' else self.freehand_points

        if len(points) < 3 or self.mask is None:
            return

        if self.parent_window:
            self.parent_window.save_mask_history()

        vertices = [[p.x(), p.y()] for p in points]
        self._fill_polygon(vertices)
        self.clear_current_drawing()

        if self.parent_window:
            self.parent_window.update_statistics()

    def _fill_polygon(self, vertices):
        """Fill polygon region in mask using matplotlib Path"""
        path = Path(vertices)

        height, width = self.mask.shape
        y, x = np.mgrid[:height, :width]
        points_array = np.c_[x.ravel(), y.ravel()]

        inside = path.contains_points(points_array)
        inside = inside.reshape(height, width)

        self.mask = self.mask | inside
        self.update()

    def paintEvent(self, event):
        """Paint the canvas"""
        painter = QPainter(self)

        if not self.image:
            painter.drawText(self.rect(), Qt.AlignCenter,
                            "Drag and drop NPZ file or use Load button")
            return

        self._calculate_transform()
        self._draw_image(painter)
        self._draw_mask_overlay(painter)
        self._draw_current_annotations(painter)

    def _calculate_transform(self):
        """Calculate scale and offset for image display"""
        img_width = self.image.width()
        img_height = self.image.height()

        scale_x = self.width() / img_width
        scale_y = self.height() / img_height
        self.scale = min(scale_x, scale_y) * 0.95

        scaled_width = int(img_width * self.scale)
        scaled_height = int(img_height * self.scale)
        self.offset = QPoint((self.width() - scaled_width) // 2,
                            (self.height() - scaled_height) // 2)

    def _draw_image(self, painter):
        """Draw the base image"""
        scaled_width = int(self.image.width() * self.scale)
        scaled_height = int(self.image.height() * self.scale)

        painter.drawImage(self.offset.x(), self.offset.y(),
                         self.image.scaled(scaled_width, scaled_height))

    def _draw_mask_overlay(self, painter):
        """Draw the mask overlay"""
        if self.mask is None or not self.show_mask:
            return

        mask_img = (self.mask * 255).astype(np.uint8)
        height, width = self.mask.shape

        mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = 255
        mask_rgba[:, :, 3] = (mask_img * self.mask_opacity).astype(np.uint8)
        mask_rgba = np.ascontiguousarray(mask_rgba)

        bytes_per_line = width * 4
        mask_qimg = QImage(mask_rgba.tobytes(), width, height,
                          bytes_per_line, QImage.Format_RGBA8888)
        self._mask_data = mask_rgba

        scaled_width = int(width * self.scale)
        scaled_height = int(height * self.scale)

        painter.drawImage(self.offset.x(), self.offset.y(),
                        mask_qimg.scaled(scaled_width, scaled_height))

    def _draw_current_annotations(self, painter):
        """Draw current polygon or freehand path"""
        painter.setPen(QPen(QColor(255, 255, 0), 2))

        if self.polygon_points:
            for i in range(len(self.polygon_points)):
                p1 = self.image_to_screen(self.polygon_points[i])
                if i < len(self.polygon_points) - 1:
                    p2 = self.image_to_screen(self.polygon_points[i + 1])
                    painter.drawLine(p1, p2)
                painter.drawEllipse(p1, 4, 4)

        if self.freehand_points and self.is_drawing:
            for i in range(len(self.freehand_points) - 1):
                p1 = self.image_to_screen(self.freehand_points[i])
                p2 = self.image_to_screen(self.freehand_points[i + 1])
                painter.drawLine(p1, p2)