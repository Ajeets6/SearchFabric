"""
Multimodal Search GUI - Powered by Ollama
PyQt5 version (more Windows-compatible)
Supports: text files, images, PDFs | Live search-as-you-type | Streaming results
"""

import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor, QPalette

from ui.styles import DARK
from ui.main_window import MultimodalSearchApp


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Multimodal Search")
    app.setStyle("Fusion")

    # Apply dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(DARK["bg"]))
    palette.setColor(QPalette.WindowText, QColor(DARK["text"]))
    palette.setColor(QPalette.Base, QColor(DARK["surface"]))
    palette.setColor(QPalette.AlternateBase, QColor(DARK["surface2"]))
    palette.setColor(QPalette.Button, QColor(DARK["surface2"]))
    palette.setColor(QPalette.ButtonText, QColor(DARK["text"]))
    palette.setColor(QPalette.Highlight, QColor(DARK["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = MultimodalSearchApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
