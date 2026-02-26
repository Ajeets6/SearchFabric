from PySide6.QtWidgets import (
    QMainWindow, QLineEdit, QListWidget, QTextEdit, QWidget, QVBoxLayout
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OmniSearch")

        self.search_bar = QLineEdit()
        self.results = QListWidget()
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.search_bar)
        layout.addWidget(self.results)
        layout.addWidget(self.preview)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)