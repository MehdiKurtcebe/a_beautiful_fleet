import sys
import subprocess
import signal
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QTextEdit,
    QSplitter,
    QLineEdit,
    QPushButton,
    QComboBox,
    QDialog,
    QProgressBar,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
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
        self.globalCommand = None
        super().__init__()
        self.setWindowTitle("A Beautiful Fleet")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        html_widget = QWidget()
        html_layout = QVBoxLayout(html_widget)

        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl("about:blank"))
        html_layout.addWidget(self.web_view)
        
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True) # For map rendering
        
        profile = QWebEngineProfile.defaultProfile()
        profile.setHttpUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        self.html_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps')
        self.html_files = self.get_html_files(self.html_folder)

        if self.html_files:
            self.update_html(self.html_files[0])

        self.html_selector = QComboBox()
        self.html_selector.addItems(self.html_files)
        self.html_selector.currentTextChanged.connect(self.on_html_select)
        html_layout.addWidget(self.html_selector)

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

        splitter.addWidget(html_widget)
        splitter.addWidget(input_widget)

        total_width = self.width()
        splitter.setSizes([int(total_width * 0.85), int(total_width * 0.15)])

        main_layout.addWidget(splitter)

    def resizeEvent(self, event):
        super().resizeEvent(event)

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
        self.globalCommand = command
        self.dialog = RunningDialog(self)
        self.worker = Worker(command, filename)
        self.worker.finished.connect(self.on_finished)
        #self.worker.output.connect(self.on_output)
        self.worker.error.connect(self.on_error)
        self.dialog.rejected.connect(self.worker.stop)

        self.worker.start()
        self.dialog.exec()

    def on_finished(self):
        self.dialog.close()
        self.refresh_html_list()
        self.writeOutput()

    def on_output(self, output):
        self.text_output.append(output)

    def on_error(self, error):
        QMessageBox.critical(self, "Error", error)
        self.dialog.close()

    def get_html_files(self, folder_path):
        return [f for f in os.listdir(folder_path) if f.lower().endswith('.html')]

    def update_html(self, html_file):
        html_path = os.path.join(os.getcwd(), "maps", html_file) 

        if not html_file:
            return 

        try:
            self.web_view.setUrl(QUrl.fromLocalFile(html_path))

        except FileNotFoundError:
            print(f"Error: HTML file not found at {html_path}")
        except Exception as e:
            print(f"An error occurred: {e}") 


    def on_html_select(self, selected_html):
        self.update_html(selected_html)

    def refresh_html_list(self):
        self.html_files = self.get_html_files(self.html_folder)
        self.html_selector.clear()
        self.html_selector.addItems(self.html_files)
        if self.html_files:
            self.update_html(self.html_files[0])
    
    def writeOutput(self):
        if self.globalCommand == "cplex":
            cplexOutputFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cplex-files', 'results', 'generalOutput.txt')
            with open(cplexOutputFile, 'r') as f:
                cplexOutputContent = f.read()
            self.on_output(cplexOutputContent)
        
        if self.globalCommand == "aco":
            acoOutputFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'aco', 'generalOutput.txt')
            with open(acoOutputFile, 'r') as f:
                acoOutputContent = f.read()
            self.on_output(acoOutputContent)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SplitViewer()
    viewer.show()
    sys.exit(app.exec())
