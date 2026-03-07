from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QFileDialog,
    QProgressBar, QSplitter, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QSpinBox, QGroupBox, QToolButton, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from ui.styles import DARK, get_main_stylesheet
from ui.result_card import ResultCard
from ui.direct_result_card import DirectResultCard
from models.ollama_client import OllamaClient
from search.search_worker import SearchWorker
from search.direct_search_worker import DirectSearchWorker
from data.file_processor import SUPPORTED_TEXT, SUPPORTED_IMAGE, SUPPORTED_PDF, MAX_FILE_SIZE_MB

# Debounce delay in milliseconds
DEBOUNCE_MS = 400


class MultimodalSearchApp(QMainWindow):
    """Main application window for multimodal search."""

    def __init__(self):
        super().__init__()
        self.ollama = OllamaClient()
        self.files = []
        self.result_cards = {}
        self.search_worker = None
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._trigger_search)

        # Search mode: direct or AI-powered
        self.direct_search_mode = True

        self._setup_window()
        self._setup_ui()
        self._apply_styles()
        self._check_ollama()

    def _setup_window(self):
        self.setWindowTitle("SearchFabric - Multimodal Search")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {DARK['border']}; }}")
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([320, 880])
        root.addWidget(splitter, 1)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background: {DARK['surface']};
                color: {DARK['text_dim']};
                font-family: 'Courier New';
                font-size: 11px;
                border-top: 1px solid {DARK['border']};
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — add files and start searching")

    def _build_header(self):
        header = QWidget()
        header.setFixedHeight(70)
        header.setObjectName("Header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel("⚡ MULTIMODAL SEARCH")
        title.setFont(QFont("Consolas, Courier New, monospace", 14, QFont.Bold))
        title.setStyleSheet(f"color: {DARK['accent']}; letter-spacing: 2px;")

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to search across all files...")
        self.search_input.setFixedHeight(42)
        self.search_input.setMinimumWidth(400)
        self.search_input.setFont(QFont("Segoe UI, Arial, sans-serif", 13))
        self.search_input.textChanged.connect(self._on_search_changed)

        model_label = QLabel("Model:")
        model_label.setStyleSheet(f"color: {DARK['text_dim']}; font-size: 11px;")
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(200)
        self.model_combo.setFixedHeight(36)

        refresh_btn = QToolButton()
        refresh_btn.setText("↻")
        refresh_btn.setFixedSize(36, 36)
        refresh_btn.setFont(QFont("Segoe UI, Arial, sans-serif", 14))
        refresh_btn.clicked.connect(self._refresh_models)
        refresh_btn.setToolTip("Refresh model list")

        # Search mode toggle
        self.mode_toggle = QPushButton("⚡ Direct Search")
        self.mode_toggle.setFixedHeight(36)
        self.mode_toggle.setFont(QFont("Consolas, Courier New, monospace", 9))
        self.mode_toggle.clicked.connect(self._toggle_search_mode)
        self.mode_toggle.setStyleSheet(f"""
            QPushButton {{
                background: {DARK['success']};
                border: 1px solid {DARK['success']};
                border-radius: 6px;
                color: white;
                padding: 0 12px;
            }}
            QPushButton:hover {{
                background: {DARK['accent']};
                border-color: {DARK['accent']};
            }}
        """)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.search_input, 1)
        layout.addStretch()
        layout.addWidget(self.mode_toggle)
        layout.addWidget(model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(refresh_btn)
        return header

    def _build_left_panel(self):
        panel = QWidget()
        panel.setObjectName("LeftPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 8, 16)
        layout.setSpacing(10)

        lbl = QLabel("📁  FILES")
        lbl.setFont(QFont("Consolas, Courier New, monospace", 10, QFont.Bold))
        lbl.setStyleSheet(f"color: {DARK['text_dim']}; letter-spacing: 2px;")
        layout.addWidget(lbl)

        btn_row = QHBoxLayout()
        self.add_files_btn = QPushButton("+ Add Files")
        self.add_folder_btn = QPushButton("+ Add Folder")
        for btn in (self.add_files_btn, self.add_folder_btn):
            btn.setFixedHeight(32)
            btn.setFont(QFont("Consolas, Courier New, monospace", 9))
        self.add_files_btn.clicked.connect(self._add_files)
        self.add_folder_btn.clicked.connect(self._add_folder)
        btn_row.addWidget(self.add_files_btn)
        btn_row.addWidget(self.add_folder_btn)
        layout.addLayout(btn_row)

        filter_group = QGroupBox("File Types")
        filter_group.setFont(QFont("Consolas, Courier New, monospace", 9))
        fg_layout = QVBoxLayout(filter_group)
        fg_layout.setSpacing(4)
        self.cb_text = QCheckBox("📄 Text / Code")
        self.cb_image = QCheckBox("🖼️  Images")
        self.cb_pdf = QCheckBox("📕 PDFs")
        for cb in (self.cb_text, self.cb_image, self.cb_pdf):
            cb.setChecked(True)
            cb.setFont(QFont("Segoe UI, Arial, sans-serif", 10))
            fg_layout.addWidget(cb)
        layout.addWidget(filter_group)

        max_row = QHBoxLayout()
        max_lbl = QLabel("Max files per search:")
        max_lbl.setFont(QFont("Segoe UI, Arial, sans-serif", 9))
        max_lbl.setStyleSheet(f"color: {DARK['text_dim']};")
        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(1, 100)
        self.max_files_spin.setValue(10)
        self.max_files_spin.setFixedWidth(60)
        max_row.addWidget(max_lbl)
        max_row.addStretch()
        max_row.addWidget(self.max_files_spin)
        layout.addLayout(max_row)

        self.file_list = QListWidget()
        self.file_list.setFont(QFont("Consolas, Courier New, monospace", 9))
        self.file_list.setSpacing(2)
        layout.addWidget(self.file_list, 1)

        bottom_row = QHBoxLayout()
        self.file_count_label = QLabel("0 files")
        self.file_count_label.setStyleSheet(f"color: {DARK['text_dim']}; font-size: 10px;")
        clear_btn = QPushButton("Clear All")
        clear_btn.setFixedHeight(26)
        clear_btn.setFont(QFont("Consolas, Courier New, monospace", 8))
        clear_btn.clicked.connect(self._clear_files)
        bottom_row.addWidget(self.file_count_label)
        bottom_row.addStretch()
        bottom_row.addWidget(clear_btn)
        layout.addLayout(bottom_row)
        return panel

    def _build_right_panel(self):
        panel = QWidget()
        panel.setObjectName("RightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 16, 16, 16)
        layout.setSpacing(10)

        results_header = QHBoxLayout()
        self.results_title = QLabel("RESULTS")
        self.results_title.setFont(QFont("Consolas, Courier New, monospace", 10, QFont.Bold))
        self.results_title.setStyleSheet(f"color: {DARK['text_dim']}; letter-spacing: 2px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setVisible(False)

        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setFixedHeight(28)
        self.stop_btn.setFont(QFont("Consolas, Courier New, monospace", 9))
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self._stop_search)

        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.setFixedHeight(28)
        self.clear_results_btn.setFont(QFont("Consolas, Courier New, monospace", 9))
        self.clear_results_btn.clicked.connect(self._clear_results)

        results_header.addWidget(self.results_title)
        results_header.addStretch()
        results_header.addWidget(self.stop_btn)
        results_header.addWidget(self.clear_results_btn)
        layout.addLayout(results_header)
        layout.addWidget(self.progress_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setObjectName("ResultsScroll")

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 8, 0)
        self.results_layout.setSpacing(6)

        self.placeholder = QLabel(
            "🔍  Start typing in the search bar\nto see direct content matches"
        )
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setFont(QFont("Segoe UI, Arial, sans-serif", 13))
        self.placeholder.setStyleSheet(f"color: {DARK['text_dim']};")
        self.results_layout.addWidget(self.placeholder)
        self.results_layout.addStretch()

        scroll.setWidget(self.results_container)
        layout.addWidget(scroll, 1)
        self.scroll_area = scroll
        return panel

    def _apply_styles(self):
        self.setStyleSheet(get_main_stylesheet())

    # ── Ollama ────────────────────────────────
    def _check_ollama(self):
        if self.ollama.is_running():
            self._refresh_models()
            self.status_bar.showMessage("✅  Ollama connected — ready to search")
        else:
            self.status_bar.showMessage("⚠️  Ollama not detected — start Ollama and click ↻")
            self.model_combo.addItem("(Ollama offline)")

    def _refresh_models(self):
        models = self.ollama.list_models()
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            for m in models:
                if any(x in m.lower() for x in ("llava", "vision", "bakllava")):
                    self.model_combo.setCurrentText(m)
                    break
            self.status_bar.showMessage(f"✅  {len(models)} models loaded")
        else:
            self.model_combo.addItem("(no models found)")
            self.status_bar.showMessage("⚠️  No models — run: ollama pull llava")

    def _toggle_search_mode(self):
        """Toggle between direct search and AI-powered search."""
        self.direct_search_mode = not self.direct_search_mode

        if self.direct_search_mode:
            self.mode_toggle.setText("⚡ Direct Search")
            self.mode_toggle.setStyleSheet(f"""
                QPushButton {{
                    background: {DARK['success']};
                    border: 1px solid {DARK['success']};
                    border-radius: 6px;
                    color: white;
                    padding: 0 12px;
                }}
                QPushButton:hover {{
                    background: {DARK['accent']};
                    border-color: {DARK['accent']};
                }}
            """)
            self.placeholder.setText("🔍  Start typing in the search bar\nto see direct content matches")
            self.status_bar.showMessage("⚡ Direct search mode - shows actual content without AI analysis")
        else:
            self.mode_toggle.setText("🤖 AI Search")
            self.mode_toggle.setStyleSheet(f"""
                QPushButton {{
                    background: {DARK['accent2']};
                    border: 1px solid {DARK['accent2']};
                    border-radius: 6px;
                    color: white;
                    padding: 0 12px;
                }}
                QPushButton:hover {{
                    background: {DARK['accent']};
                    border-color: {DARK['accent']};
                }}
            """)
            self.placeholder.setText("🔍  Start typing in the search bar\nto see AI-powered results stream in")
            self.status_bar.showMessage("🤖 AI search mode - uses LLM to analyze and explain content")

        # Clear current results when switching modes
        self._clear_results()

    # ── File Management ───────────────────────
    def _add_files(self):
        exts = []
        if self.cb_text.isChecked():
            exts += list(SUPPORTED_TEXT)
        if self.cb_image.isChecked():
            exts += list(SUPPORTED_IMAGE)
        if self.cb_pdf.isChecked():
            exts += list(SUPPORTED_PDF)
        filter_str = "Supported Files (*" + " *".join(exts) + ");;All Files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", filter_str)
        for p in paths:
            self._add_file(Path(p))

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        all_exts = SUPPORTED_TEXT | SUPPORTED_IMAGE | SUPPORTED_PDF
        for f in Path(folder).rglob("*"):
            if f.suffix.lower() in all_exts and f.stat().st_size < MAX_FILE_SIZE_MB * 1024 * 1024:
                self._add_file(f)

    def _add_file(self, path):
        if path not in self.files:
            self.files.append(path)
            suffix = path.suffix.lower()
            icon = "🖼️" if suffix in SUPPORTED_IMAGE else ("📕" if suffix in SUPPORTED_PDF else "📄")
            item = QListWidgetItem(f"{icon}  {path.name}")
            item.setToolTip(str(path))
            self.file_list.addItem(item)
            self.file_count_label.setText(f"{len(self.files)} files")

    def _clear_files(self):
        self.files.clear()
        self.file_list.clear()
        self.file_count_label.setText("0 files")

    # ── Search ────────────────────────────────
    def _on_search_changed(self, text):
        self.debounce_timer.stop()
        if text.strip():
            self.debounce_timer.start(DEBOUNCE_MS)
        else:
            self._stop_search()
            self._clear_results()

    def _trigger_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
        if not self.files:
            self.status_bar.showMessage("⚠️  Add files first")
            return

        # Get filtered files
        filtered = []
        for f in self.files:
            s = f.suffix.lower()
            if s in SUPPORTED_TEXT and self.cb_text.isChecked():
                filtered.append(f)
            elif s in SUPPORTED_IMAGE and self.cb_image.isChecked():
                filtered.append(f)
            elif s in SUPPORTED_PDF and self.cb_pdf.isChecked():
                filtered.append(f)
        filtered = filtered[:self.max_files_spin.value()]

        if not filtered:
            self.status_bar.showMessage("⚠️  No matching files for enabled types")
            return

        self._stop_search()
        self._clear_results()

        self.placeholder.setVisible(False)
        self.progress_bar.setVisible(True)
        self.stop_btn.setVisible(True)

        if self.direct_search_mode:
            # Direct search mode
            self.results_title.setText(f'⚡ DIRECT RESULTS  ·  "{query}"  ·  {len(filtered)} files')
            self.status_bar.showMessage(f"⚡ Direct search in {len(filtered)} files...")

            self.search_worker = DirectSearchWorker(query, filtered)
            self.search_worker.result_found.connect(self._on_direct_result_found)
            self.search_worker.result_started.connect(self._on_result_started)
            self.search_worker.result_done.connect(self._on_result_done)
            self.search_worker.error_occurred.connect(self._on_result_error)
            self.search_worker.all_done.connect(self._on_all_done)
        else:
            # AI search mode
            model = self.model_combo.currentText()
            if "(no models" in model or "(Ollama" in model:
                self.status_bar.showMessage("⚠️  No valid model selected")
                return

            self.results_title.setText(f'🤖 AI RESULTS  ·  "{query}"  ·  {len(filtered)} files')
            self.status_bar.showMessage(f"🤖 AI analyzing {len(filtered)} files with {model}...")

            self.search_worker = SearchWorker(query, filtered, model, self.ollama)
            self.search_worker.result_started.connect(self._on_result_started)
            self.search_worker.token_received.connect(self._on_token_received)
            self.search_worker.result_done.connect(self._on_result_done)
            self.search_worker.error_occurred.connect(self._on_result_error)
            self.search_worker.all_done.connect(self._on_all_done)

        self.search_worker.start()

    def _stop_search(self):
        if self.search_worker and self.search_worker.isRunning():
            self.search_worker.stop()
            self.search_worker.wait(2000)
        self.progress_bar.setVisible(False)
        self.stop_btn.setVisible(False)

    def _clear_results(self):
        for card in list(self.result_cards.values()):
            self.results_layout.removeWidget(card)
            card.deleteLater()
        self.result_cards.clear()
        self.placeholder.setVisible(True)

        if self.direct_search_mode:
            self.results_title.setText("⚡ DIRECT RESULTS")
        else:
            self.results_title.setText("🤖 AI RESULTS")

    # ── Signal handlers ───────────────────────
    def _on_direct_result_found(self, result_id, filename, file_type, content, score):
        """Handle direct search result found."""
        card = DirectResultCard(result_id, filename, file_type, content, score)
        card.analyze_requested.connect(self._on_analyze_requested)
        self.result_cards[result_id] = card
        count = self.results_layout.count()
        self.results_layout.insertWidget(count - 1, card)

        # Auto-scroll to bottom
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def _on_analyze_requested(self, result_id, filename, file_type):
        """Handle request for AI analysis of a specific result."""
        model = self.model_combo.currentText()
        if "(no models" in model or "(Ollama" in model:
            self.status_bar.showMessage("⚠️  No valid model selected for analysis")
            return

        # Find the file for this result
        target_file = None
        for file_path in self.files:
            if f"{file_path.stem}" in result_id and file_path.name == filename:
                target_file = file_path
                break

        if not target_file:
            self.status_bar.showMessage("⚠️  File not found for analysis")
            return

        # Create a single-file AI analysis worker
        from search.search_worker import SearchWorker
        query = self.search_input.text().strip()

        analysis_worker = SearchWorker(query, [target_file], model, self.ollama)

        # Create new result card for AI analysis
        ai_result_id = f"ai_{result_id}"

        def on_started(rid, fname, ftype):
            if rid.replace(f"result_0_", "") == target_file.stem:
                card = ResultCard(ai_result_id, f"🤖 {filename}", file_type)
                self.result_cards[ai_result_id] = card
                count = self.results_layout.count()
                self.results_layout.insertWidget(count - 1, card)
                self.status_bar.showMessage(f"🤖 Analyzing {filename} with AI...")

        def on_token(rid, token):
            if ai_result_id in self.result_cards:
                self.result_cards[ai_result_id].append_token(token)

        def on_done(rid):
            if ai_result_id in self.result_cards:
                self.result_cards[ai_result_id].mark_done()
                self.status_bar.showMessage(f"✅ AI analysis complete for {filename}")

        def on_error(rid, error):
            if ai_result_id in self.result_cards:
                self.result_cards[ai_result_id].mark_error(error)

        analysis_worker.result_started.connect(on_started)
        analysis_worker.token_received.connect(on_token)
        analysis_worker.result_done.connect(on_done)
        analysis_worker.error_occurred.connect(on_error)
        analysis_worker.start()

    def _on_result_started(self, result_id, filename, file_type):
        if not self.direct_search_mode:
            card = ResultCard(result_id, filename, file_type)
            self.result_cards[result_id] = card
            count = self.results_layout.count()
            self.results_layout.insertWidget(count - 1, card)
            QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ))

    def _on_token_received(self, result_id, token):
        if result_id in self.result_cards:
            self.result_cards[result_id].append_token(token)

    def _on_result_done(self, result_id):
        if result_id in self.result_cards:
            self.result_cards[result_id].mark_done()

    def _on_result_error(self, result_id, error):
        if result_id in self.result_cards:
            self.result_cards[result_id].mark_error(error)

    def _on_all_done(self):
        self.progress_bar.setVisible(False)
        self.stop_btn.setVisible(False)
        self.status_bar.showMessage(f"✅  Done — {len(self.result_cards)} results")
