import sys
import subprocess
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QTextEdit,
    QSplitter,
    QLineEdit,
    QPushButton,
    QComboBox,
    QScrollArea,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


def run_script(input_param):
    result = subprocess.run(['python', 'aco/bf-aco-cleanv2.py', input_param], capture_output=True, text=True)
    if result.stderr:
        return f"Error: {result.stderr}"
    return result.stdout


class SplitViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A Beautiful Fleet")
        self.setGeometry(100, 100, 1200, 800)
        self.scaleFactor = 0.5

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        img_widget = QWidget()
        img_layout = QVBoxLayout(img_widget)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.img_label)
        self.scroll_area.setWidgetResizable(True)
        img_layout.addWidget(self.scroll_area)

        self.image_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps') 
        self.image_files = self.get_image_files(self.image_folder)

        if self.image_files:
            self.update_image(self.image_files[0])

        self.image_selector = QComboBox()
        self.image_selector.addItems(self.image_files)
        self.image_selector.currentTextChanged.connect(self.on_image_select)
        img_layout.addWidget(self.image_selector)

        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_button)
        img_layout.addLayout(zoom_layout)

        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter dataset name")
        input_layout.addWidget(self.input_field)

        self.run_button = QPushButton("Start")
        self.run_button.clicked.connect(self.run_script)
        input_layout.addWidget(self.run_button)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        input_layout.addWidget(self.text_output)

        splitter.addWidget(img_widget)
        splitter.addWidget(input_widget)

        total_width = self.width()
        splitter.setSizes([int(total_width * 0.7), int(total_width * 0.3)])

        main_layout.addWidget(splitter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "img_label") and self.img_label.pixmap():
            self.update_image(self.image_selector.currentText())

    def run_script(self):
        filename = self.input_field.text()
        if filename:
            result = run_script(filename)
            self.text_output.setText(result)
            self.refresh_image_list() 
        else:
            self.text_output.setText("Please enter a valid filename.")

    def get_image_files(self, folder_path):
        return [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]

    def update_image(self, image_file):
        image_path = os.path.join(self.image_folder, image_file)
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            int(self.scaleFactor * pixmap.width()),
            int(self.scaleFactor * pixmap.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.img_label.setPixmap(scaled_pixmap)
        self.img_label.adjustSize()

    def on_image_select(self, selected_image):
        self.update_image(selected_image)

    def zoom_in(self):
        self.scaleFactor *= 1.1
        self.update_image(self.image_selector.currentText())

    def zoom_out(self):
        self.scaleFactor *= 0.9
        self.update_image(self.image_selector.currentText())

    def refresh_image_list(self):
        self.image_files = self.get_image_files(self.image_folder)
        self.image_selector.clear()
        self.image_selector.addItems(self.image_files)
        if self.image_files:
            self.update_image(self.image_files[0])  # Update to the first image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SplitViewer()
    viewer.show()
    sys.exit(app.exec())