from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from ui.styles import DARK

class DirectResultCard(QFrame):
    """Widget displaying direct search results without LLM processing."""

    analyze_requested = pyqtSignal(str, str, str)  # (result_id, filename, file_type)

    def __init__(self, result_id, filename, file_type, content, score=0.0, parent=None):
        super().__init__(parent)
        self.result_id = result_id
        self.filename = filename
        self.file_type = file_type
        self.content = content
        self.score = score
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

        # Header
        header = QHBoxLayout()
        icon_lbl = QLabel(type_icons.get(self.file_type, "📄"))
        icon_lbl.setFont(QFont("Segoe UI Emoji", 14))

        fname_lbl = QLabel(self.filename)
        fname_lbl.setFont(QFont("Courier New", 10, QFont.Bold))
        fname_lbl.setStyleSheet(f"color: {type_colors.get(self.file_type, DARK['accent'])};")

        # Score indicator
        score_color = DARK["success"] if self.score > 0.7 else (DARK["warning"] if self.score > 0.3 else DARK["text_dim"])
        score_badge = QLabel(f"Relevance: {self.score:.1%}")
        score_badge.setFont(QFont("Courier New", 8))
        score_badge.setStyleSheet(f"""
            color: {score_color};
            background: {DARK['surface2']};
            border: 1px solid {score_color};
            border-radius: 4px;
            padding: 2px 6px;
        """)

        # Type badge
        badge = QLabel(self.file_type.upper())
        badge.setFont(QFont("Courier New", 8))
        badge.setStyleSheet(f"""
            color: {DARK['bg']};
            background: {type_colors.get(self.file_type, DARK['accent'])};
            border-radius: 4px;
            padding: 2px 6px;
        """)

        # Direct result indicator
        direct_icon = QLabel("⚡")
        direct_icon.setFont(QFont("Segoe UI Emoji", 10))
        direct_icon.setToolTip("Direct content match")

        header.addWidget(icon_lbl)
        header.addWidget(fname_lbl)
        header.addStretch()
        header.addWidget(score_badge)
        header.addWidget(badge)
        header.addWidget(direct_icon)
        layout.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {DARK['border']};")
        layout.addWidget(sep)

        # Content display
        self.content_label = QLabel(self.content)
        self.content_label.setWordWrap(True)
        self.content_label.setFont(QFont("Consolas", 9))  # Monospace for better code reading
        self.content_label.setStyleSheet(f"""
            color: {DARK['text']};
            background: {DARK['surface']};
            border: 1px solid {DARK['border']};
            border-radius: 4px;
            padding: 12px;
            line-height: 1.4;
        """)
        self.content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.content_label)

        # Optional AI analysis button
        button_layout = QHBoxLayout()

        self.analyze_btn = QPushButton("🤖 Analyze with AI")
        self.analyze_btn.setFixedHeight(28)
        self.analyze_btn.setFont(QFont("Courier New", 8))
        self.analyze_btn.clicked.connect(self._request_analysis)
        self.analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {DARK['surface2']};
                border: 1px solid {DARK['accent2']};
                border-radius: 4px;
                color: {DARK['accent2']};
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background: {DARK['accent2']};
                color: white;
            }}
        """)

        # Copy content button
        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.setFixedHeight(28)
        self.copy_btn.setFont(QFont("Courier New", 8))
        self.copy_btn.clicked.connect(self._copy_content)
        self.copy_btn.setStyleSheet(f"""
            QPushButton {{
                background: {DARK['surface2']};
                border: 1px solid {DARK['text_dim']};
                border-radius: 4px;
                color: {DARK['text_dim']};
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background: {DARK['text_dim']};
                color: white;
            }}
        """)

        button_layout.addWidget(self.copy_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.analyze_btn)
        layout.addLayout(button_layout)

    def _request_analysis(self):
        """Request AI analysis for this result."""
        self.analyze_requested.emit(self.result_id, self.filename, self.file_type)
        self.analyze_btn.setText("🤖 Analyzing...")
        self.analyze_btn.setEnabled(False)

    def _copy_content(self):
        """Copy content to clipboard."""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.content)

        # Visual feedback
        original_text = self.copy_btn.text()
        self.copy_btn.setText("✅ Copied!")

        # Reset button text after 2 seconds
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.copy_btn.setText(original_text))

    def mark_done(self):
        """Mark the result as complete (for compatibility with ResultCard interface)."""
        # Direct results are already complete, but we can add visual feedback
        self.setStyleSheet(f"""
            QFrame#ResultCard {{
                background: {DARK['result_bg']};
                border: 1px solid {DARK['success']};
                border-radius: 10px;
                margin: 4px 2px;
            }}
        """)

    def append_token(self, token: str):
        """Append token (for compatibility with ResultCard interface - not used in direct search)."""
        # Direct search results are complete, so this is a no-op
        pass

    def mark_error(self, error: str):
        """Mark the result as having an error."""
        self.content_label.setText(f"❌ Error: {error}")
        self.content_label.setStyleSheet(f"""
            color: {DARK['error']};
            background: {DARK['surface']};
            border: 1px solid {DARK['error']};
            border-radius: 4px;
            padding: 12px;
            line-height: 1.4;
        """)