"""
Multimodal Search GUI - Powered by Ollama
PySide6 version (modern Qt with better tooling)
Supports: text files, images, PDFs | Live search-as-you-type | Streaming results
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor, QPalette

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ui.styles import DARK
from ui.main_window import MultimodalSearchApp


def main():
    """Application entry point."""
    # Load optional .env values (e.g., RAM_PLUS_CHECKPOINT) from project root.
    if load_dotenv is not None:
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env")

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
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
