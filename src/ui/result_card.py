"""
Result Card Widget
"""

from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.styles import DARK


class ResultCard(QFrame):
    """Widget displaying a single search result."""

    def __init__(self, result_id, filename, file_type, parent=None):
        super().__init__(parent)
        self.result_id = result_id
        self.filename = filename
        self.file_type = file_type
        self._text_buffer = ""
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("ResultCard")
        self.setStyleSheet(f"""
            QFrame#ResultCard {{
                background: {DARK['result_bg']};
                border: 1px solid {DARK['border']};
                border-radius: 10px;
                margin: 4px 2px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        type_colors = {
            "text": DARK["accent"],
            "image": DARK["accent2"],
            "pdf": DARK["warning"],
            "unknown": DARK["text_dim"]
        }
        type_icons = {
            "text": "📄",
            "image": "🖼️",
            "pdf": "📕",
            "unknown": "❓"
        }

        header = QHBoxLayout()
        icon_lbl = QLabel(type_icons.get(self.file_type, "📄"))
        icon_lbl.setFont(QFont("Segoe UI Emoji", 14))

        fname_lbl = QLabel(self.filename)
        fname_lbl.setFont(QFont("Courier New", 10, QFont.Bold))
        fname_lbl.setStyleSheet(f"color: {type_colors.get(self.file_type, DARK['accent'])};")

        badge = QLabel(self.file_type.upper())
        badge.setFont(QFont("Courier New", 8))
        badge.setStyleSheet(f"""
            color: {DARK['bg']};
            background: {type_colors.get(self.file_type, DARK['accent'])};
            border-radius: 4px;
            padding: 2px 6px;
        """)

        self.status_dot = QLabel("⏳")
        self.status_dot.setFont(QFont("Segoe UI Emoji", 10))

        header.addWidget(icon_lbl)
        header.addWidget(fname_lbl)
        header.addStretch()
        header.addWidget(badge)
        header.addWidget(self.status_dot)
        layout.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {DARK['border']};")
        layout.addWidget(sep)

        self.content_label = QLabel("")
        self.content_label.setWordWrap(True)
        self.content_label.setFont(QFont("Georgia", 10))
        self.content_label.setStyleSheet(f"color: {DARK['text']};")
        self.content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.content_label)

    def append_token(self, token):
        """Append a token to the result content."""
        self._text_buffer += token
        self.content_label.setText(self._text_buffer)

    def mark_done(self):
        """Mark the result as complete."""
        self.status_dot.setText("✅")
        self.setStyleSheet(f"""
            QFrame#ResultCard {{
                background: {DARK['result_bg']};
                border: 1px solid {DARK['accent']};
                border-radius: 10px;
                margin: 4px 2px;
            }}
        """)

    def mark_error(self, err):
        """Mark the result as having an error."""
        self.status_dot.setText("❌")
        self.content_label.setText(f"Error: {err}")
        self.content_label.setStyleSheet(f"color: {DARK['error']};")
