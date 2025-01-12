import sys
import subprocess
import signal
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
    QDialog,
    QProgressBar,
    QMessageBox,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import platform

class Worker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    output = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, command, input_param):
        super().__init__()
        self.command = command
        self.input_param = input_param
        self._is_running = True
        self.process = None

    def run(self):
        try:
            creation_flags = 0
            preexec_fn = None

            # Use `os.setsid` for Unix-like systems or `CREATE_NEW_PROCESS_GROUP` for Windows
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = os.setsid

            if self.command == "aco":
                self.process = subprocess.Popen(
                    ['.venv/Scripts/python', 'aco/bf-aco-clean.py', self.input_param],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=creation_flags,
                    preexec_fn=preexec_fn
                )
            elif self.command == "cplex":
                self.process = subprocess.Popen(
                    ['.venv/Scripts/python', 'run-cplex.py', self.input_param],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=creation_flags,
                    preexec_fn=preexec_fn
                )
            else:
                raise ValueError("Invalid command")

            while self._is_running:
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    self.output.emit(output.strip())

            if self.process.returncode != 0:
                error_output = self.process.stderr.read()
                self.error.emit(f"Error: {error_output}")
            else:
                self.output.emit("Finished successfully!")

        except Exception as e:
            self.error.emit(f"Error: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False
        if self.process:
            try:
                # Terminate the entire process group
                if platform.system() == "Windows":
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception as e:
                self.error.emit(f"Error stopping process: {e}")





class RunningDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Running...")
        self.setModal(True)

        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)


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

        self.run_button1 = QPushButton("Start ACO")
        self.run_button1.clicked.connect(self.run_aco_button)
        input_layout.addWidget(self.run_button1)

        self.run_button2 = QPushButton("Start CPLEX")
        self.run_button2.clicked.connect(self.run_cplex_button)
        input_layout.addWidget(self.run_button2)

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

    def run_aco_button(self):
        filename = self.input_field.text()
        if filename:
            self.run_algorithm("aco", filename)
        else:
            self.text_output.setText("Please enter a valid filename.")

    def run_cplex_button(self):
        filename = self.input_field.text()
        if filename:
            self.run_algorithm("cplex", filename)
        else:
            self.text_output.setText("Please enter a valid filename.")

    def run_algorithm(self, command, filename):
        self.dialog = RunningDialog(self)
        self.worker = Worker(command, filename)
        self.worker.finished.connect(self.on_finished)
        self.worker.output.connect(self.on_output)
        self.worker.error.connect(self.on_error)
        self.dialog.rejected.connect(self.worker.stop)

        self.worker.start()
        self.dialog.exec()

    def on_finished(self):
        self.dialog.close()
        self.refresh_image_list()

    def on_output(self, output):
        self.text_output.append(output)

    def on_error(self, error):
        QMessageBox.critical(self, "Error", error)
        self.dialog.close()

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
            self.update_image(self.image_files[0])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SplitViewer()
    viewer.show()
    sys.exit(app.exec())