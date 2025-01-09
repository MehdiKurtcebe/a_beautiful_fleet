import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QTextEdit,
    QSplitter,
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from pandas import read_csv


class SplitViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A Beatiful Fleet")
        self.setGeometry(100, 100, 1200, 800)

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main horizontal layout
        main_layout = QHBoxLayout(central_widget)

        # Create a splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left widget (Image)
        img_widget = QWidget()
        img_layout = QVBoxLayout(img_widget)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Load and display the image
        pixmap = QPixmap("izmit.png")
        scaled_pixmap = pixmap.scaled(
            800,
            780,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.img_label.setPixmap(scaled_pixmap)
        img_layout.addWidget(self.img_label)

        # Right widget (Text Area)
        self.text_area = QTextEdit()
        self.text_area.setFont(QFont("Consolas", 10))
        self.text_area.setReadOnly(True)  # Make it read-only
        self.text_area.setPlaceholderText("Result")

        # Add some sample text
        sample_text = printCSV("result_izmit_6_200.csv")

        self.text_area.setText(sample_text)

        # Add widgets to splitter
        splitter.addWidget(img_widget)
        splitter.addWidget(self.text_area)

        # Set the initial sizes (70% - 30% split)
        total_width = self.width()
        splitter.setSizes([int(total_width * 0.7), int(total_width * 0.3)])

        # Add splitter to main layout
        main_layout.addWidget(splitter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update image scaling when window is resized
        if hasattr(self, "image_label") and self.img_label.pixmap():
            pixmap = QPixmap("izmit.png")
            scaled_pixmap = pixmap.scaled(
                800,
                780,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.img_label.setPixmap(scaled_pixmap)
            
            
def printCSV(filename):
    data = readCSV(filename)
    string = ""
    for d in data.values:
        string += f"Beautificator {d[0]}, in zone {d[1]}, {d[6]}\n"
    return string


def readCSV(filename):
    df = read_csv(filename)
    df = df.sort_values(by=[df.columns[0], df.columns[2]])
    # df = df.groupby(df.columns[0])
    return df


if __name__ == "__main__":
    # data = readCSV('result_izmit_6_200.csv')
    app = QApplication(sys.argv)
    viewer = SplitViewer()
    viewer.show()
    sys.exit(app.exec())
