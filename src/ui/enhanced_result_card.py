from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ui.styles import DARK

class EnhancedResultCard(QFrame):
    """Widget displaying fast results with optional LLM enhancement."""

    llm_analysis_requested = Signal(str)  # Request LLM analysis for this result

    def __init__(self, result_id, filename, file_type, score=0.0, parent=None):
        super().__init__(parent)
        self.result_id = result_id
        self.filename = filename
        self.file_type = file_type
        self.score = score
        self._fast_content = ""
        self._llm_content = ""
        self._has_llm_analysis = False
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

        # Header with enhanced info
        header = QHBoxLayout()
        icon_lbl = QLabel(type_icons.get(self.file_type, "📄"))
        icon_lbl.setFont(QFont("Segoe UI Emoji", 14))

        fname_lbl = QLabel(self.filename)
        fname_lbl.setFont(QFont("Consolas, Courier New, monospace", 10, QFont.Bold))
        fname_lbl.setStyleSheet(f"color: {type_colors.get(self.file_type, DARK['accent'])};")

        # Score badge
        score_badge = QLabel(f"{self.score:.2f}")
        score_badge.setFont(QFont("Consolas, Courier New, monospace", 8))
        score_color = DARK["success"] if self.score > 0.7 else (DARK["warning"] if self.score > 0.4 else DARK["text_dim"])
        score_badge.setStyleSheet(f"""
            color: {DARK['bg']};
            background: {score_color};
            border-radius: 4px;
            padding: 2px 6px;
        """)

        # Type badge
        badge = QLabel(self.file_type.upper())
        badge.setFont(QFont("Consolas, Courier New, monospace", 8))
        badge.setStyleSheet(f"""
            color: {DARK['bg']};
            background: {type_colors.get(self.file_type, DARK['accent'])};
            border-radius: 4px;
            padding: 2px 6px;
        """)

        self.status_dot = QLabel("⚡")  # Fast result indicator
        self.status_dot.setFont(QFont("Segoe UI Emoji", 10))

        header.addWidget(icon_lbl)
        header.addWidget(fname_lbl)
        header.addStretch()
        header.addWidget(score_badge)
        header.addWidget(badge)
        header.addWidget(self.status_dot)
        layout.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {DARK['border']};")
        layout.addWidget(sep)

        # Fast content area
        self.fast_content_label = QLabel("")
        self.fast_content_label.setWordWrap(True)
        self.fast_content_label.setFont(QFont("Segoe UI, Arial, sans-serif", 10))
        self.fast_content_label.setStyleSheet(f"color: {DARK['text_dim']};")
        self.fast_content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.fast_content_label)

        # LLM analysis area (initially hidden)
        self.llm_content_label = QLabel("")
        self.llm_content_label.setWordWrap(True)
        self.llm_content_label.setFont(QFont("Segoe UI, Arial, sans-serif", 10))
        self.llm_content_label.setStyleSheet(f"color: {DARK['text']}; margin-top: 8px;")
        self.llm_content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.llm_content_label.hide()
        layout.addWidget(self.llm_content_label)

        # Analyze button (for high-score results without LLM analysis)
        self.analyze_btn = QPushButton("🔍 Analyze with AI")
        self.analyze_btn.setFixedHeight(28)
        self.analyze_btn.setFont(QFont("Consolas, Courier New, monospace", 8))
        self.analyze_btn.clicked.connect(self._request_llm_analysis)
        self.analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {DARK['surface2']};
                border: 1px solid {DARK['accent']};
                border-radius: 4px;
                color: {DARK['accent']};
                padding: 4px 8px;
                margin-top: 4px;
            }}
            QPushButton:hover {{
                background: {DARK['accent']};
                color: white;
            }}
        """)
        self.analyze_btn.hide()
        layout.addWidget(self.analyze_btn)

    def set_fast_content(self, content: str):
        """Set the fast search result content."""
        self._fast_content = content
        self.fast_content_label.setText(f"🔍 Fast match: {content}")

        # Show analyze button for high-score results
        if self.score > 0.5 and not self._has_llm_analysis:
            self.analyze_btn.show()

    def start_llm_analysis(self):
        """Indicate that LLM analysis has started."""
        self._has_llm_analysis = True
        self.analyze_btn.hide()
        self.status_dot.setText("🤔")

        # Show LLM area with loading indicator
        self.llm_content_label.setText("🤖 AI is analyzing...")
        self.llm_content_label.show()

    def append_llm_token(self, token: str):
        """Append a token to the LLM analysis content."""
        if not self._has_llm_analysis:
            self.start_llm_analysis()

        self._llm_content += token
        self.llm_content_label.setText(f"🤖 AI Analysis: {self._llm_content}")

    def mark_llm_done(self):
        """Mark the LLM analysis as complete."""
        self.status_dot.setText("✅")
        self.setStyleSheet(f"""
            QFrame#ResultCard {{
                background: {DARK['result_bg']};
                border: 1px solid {DARK['success']};
                border-radius: 10px;
                margin: 4px 2px;
            }}
        """)

    def mark_error(self, err: str):
        """Mark the result as having an error."""
        self.status_dot.setText("❌")
        if self._has_llm_analysis:
            self.llm_content_label.setText(f"🤖 AI Analysis Error: {err}")
            self.llm_content_label.setStyleSheet(f"color: {DARK['error']};")
        else:
            self.fast_content_label.setText(f"Error: {err}")
            self.fast_content_label.setStyleSheet(f"color: {DARK['error']};")

    def _request_llm_analysis(self):
        """Request LLM analysis for this result."""
        self.llm_analysis_requested.emit(self.result_id)
        self.start_llm_analysis()