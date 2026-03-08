from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QFileDialog,
    QProgressBar, QSplitter, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QSpinBox, QGroupBox, QToolButton, QStatusBar
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont

from ui.styles import DARK, get_main_stylesheet
from ui.enhanced_result_card import EnhancedResultCard
from search.hybrid_search import HybridSearchEngine, SearchResult, SearchMode
from data.file_processor import SUPPORTED_TEXT, SUPPORTED_IMAGE, SUPPORTED_PDF, MAX_FILE_SIZE_MB

# Debounce delay in milliseconds
DEBOUNCE_MS = 400


class IndexingWorker(QThread):
    """Background worker for file indexing."""
    progress_update = Signal(int, int)  # completed, total
    indexing_finished = Signal(dict)  # stats dict
    error_occurred = Signal(str)

    def __init__(self, search_engine, files):
        super().__init__()
        self.search_engine = search_engine
        self.files = files

    def run(self):
        try:
            def progress_callback(total, completed):
                self.progress_update.emit(completed, total)

            stats = self.search_engine.index_files(self.files, progress_callback)
            self.indexing_finished.emit(stats)
        except Exception as e:
            self.error_occurred.emit(str(e))


class HybridSearchWorker(QThread):
    """Worker thread for hybrid search operations."""
    search_completed = Signal(list)  # List[SearchResult]
    progress_update = Signal(int, int)  # current, total

    def __init__(self, search_engine, query, files, max_files):
        super().__init__()
        self.search_engine = search_engine
        self.query = query
        self.files = files[:max_files]  # Limit files
        self._stop = False

    def stop_search(self):
        self._stop = True

    def run(self):
        try:
            # Determine file types from the files list
            file_types = set()
            for file_path in self.files:
                suffix = file_path.suffix.lower()
                if suffix in SUPPORTED_TEXT:
                    file_types.add("text")
                elif suffix in SUPPORTED_IMAGE:
                    file_types.add("image")
                elif suffix in SUPPORTED_PDF:
                    file_types.add("pdf")

            # First, ensure files are indexed
            if self.files:
                self.search_engine.index_files(self.files)

            # Perform search
            results = self.search_engine.search(
                query=self.query,
                mode=SearchMode.HYBRID,
                file_types=file_types,
                limit=50
            )

            # Filter results to only include our target files
            file_paths_set = {str(f) for f in self.files}
            filtered_results = [r for r in results if r.file_path in file_paths_set]

            if not self._stop:
                self.search_completed.emit(filtered_results)
        except Exception as e:
            print(f"Search error: {e}")
            if not self._stop:
                self.search_completed.emit([])


class MultimodalSearchApp(QMainWindow):
    """Main application window for fast hybrid search."""

    def __init__(self):
        super().__init__()

        # Initialize search engine with error handling
        try:
            self.search_engine = HybridSearchEngine()
        except Exception as e:
            print(f"Warning: Failed to initialize hybrid search engine: {e}")
            # Create a minimal search engine with just text indexing
            from indexing.text_indexer import TextIndexer
            self.search_engine = type('MinimalSearchEngine', (), {
                'text_indexer': TextIndexer(),
                'semantic_indexer': None,
                'semantic_available': False,
                'index_files': lambda files, callback=None: {"text": 0, "semantic": 0, "errors": len(files), "skipped": 0},
                'search': lambda query, mode=None, file_types=None, limit=50: []
            })()

        self.files = []
        self.result_cards = {}
        self.search_worker = None
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._trigger_search)

        self._setup_window()
        self._setup_ui()
        self._apply_styles()
        self._setup_embedding_models()

        # Check if search engine initialized properly
        if hasattr(self.search_engine, 'semantic_available') and not self.search_engine.semantic_available:
            self.status_bar.showMessage("⚠️ Text search only - install PyTorch for full functionality")
        else:
            self.status_bar.showMessage("Ready — add files and start searching")

    def _setup_window(self):
        self.setWindowTitle("SearchFabric - Fast Hybrid Search")
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

        title = QLabel("⚡ HYBRID SEARCH")
        title.setFont(QFont("Consolas, Courier New, monospace", 14, QFont.Bold))
        title.setStyleSheet(f"color: {DARK['accent']}; letter-spacing: 2px;")

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to search - text results on top, semantic below...")
        self.search_input.setFixedHeight(42)
        self.search_input.setMinimumWidth(400)
        self.search_input.setFont(QFont("Segoe UI, Arial, sans-serif", 13))
        self.search_input.textChanged.connect(self._on_search_changed)

        model_label = QLabel("Embedding Model:")
        model_label.setStyleSheet(f"color: {DARK['text_dim']}; font-size: 11px;")
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(220)
        self.model_combo.setFixedHeight(36)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        refresh_btn = QToolButton()
        refresh_btn.setText("↻")
        refresh_btn.setFixedSize(36, 36)
        refresh_btn.setFont(QFont("Segoe UI, Arial, sans-serif", 14))
        refresh_btn.clicked.connect(self._setup_embedding_models)
        refresh_btn.setToolTip("Refresh embedding models")

        # Index files button
        self.index_btn = QPushButton("📚 Index Files")
        self.index_btn.setFixedHeight(36)
        self.index_btn.setFont(QFont("Consolas, Courier New, monospace", 9))
        self.index_btn.clicked.connect(self._index_all_files)
        self.index_btn.setStyleSheet(f"""
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

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.search_input, 1)
        layout.addStretch()
        layout.addWidget(self.index_btn)
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
            "🔍  Start typing to see hybrid search results\n📄 Direct text matches on top • 🧠 Semantic matches below"
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
    def _setup_embedding_models(self):
        """Setup available embedding models."""
        self.model_combo.clear()

        # Check what models are available based on installed dependencies
        available_models = []

        # Always include basic text models (sentence-transformers)
        try:
            from sentence_transformers import SentenceTransformer
            available_models.extend([
                "all-MiniLM-L6-v2 (Text - Fast)",
                "all-mpnet-base-v2 (Text - Quality)"
            ])
        except ImportError:
            pass

        # Add CLIP model if available
        try:
            import open_clip
            import torch
            available_models.append("clip-ViT-B-32 (Images + Text)")
        except ImportError:
            pass

        if not available_models:
            available_models = ["No models available - check dependencies"]
            self.status_bar.showMessage("⚠️ No embedding models available - install sentence-transformers")
        else:
            self.status_bar.showMessage("📚 Embedding models ready - click 'Index Files' to prepare search")

        self.model_combo.addItems(available_models)
        self.model_combo.setCurrentIndex(0)  # Default to first available

    def _on_model_changed(self, model_text):
        """Handle embedding model change."""
        if not model_text:
            return

        model_name = model_text.split(" (")[0]  # Extract model name

        try:
            # Create new semantic indexer with the selected model
            if self.search_engine.semantic_indexer:
                old_db_path = self.search_engine.semantic_indexer.db_path
                self.search_engine.semantic_indexer = None  # Clean up old instance

                # Create new indexer with new model
                from indexing.semantic_indexer import SemanticIndexer
                self.search_engine.semantic_indexer = SemanticIndexer(old_db_path, model_name)

                self.status_bar.showMessage(f"📚 Switched to {model_name} - re-index files for best results")
        except Exception as e:
            self.status_bar.showMessage(f"⚠️ Model change failed: {e}")
            print(f"Model change error: {e}")

    def _index_all_files(self):
        """Index all loaded files in a background thread."""
        if not self.files:
            self.status_bar.showMessage("⚠️ Add some files first")
            return

        self.index_btn.setEnabled(False)
        self.index_btn.setText("📚 Indexing...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.files))

        self._indexing_worker = IndexingWorker(self.search_engine, list(self.files))
        self._indexing_worker.progress_update.connect(self._on_indexing_progress)
        self._indexing_worker.indexing_finished.connect(self._on_indexing_finished)
        self._indexing_worker.error_occurred.connect(self._on_indexing_error)
        self._indexing_worker.start()

    def _on_indexing_progress(self, completed, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.status_bar.showMessage(f"📚 Indexing files... {completed}/{total}")

    def _on_indexing_finished(self, stats):
        total_indexed = stats.get('text', 0) + stats.get('semantic', 0)
        errors = stats.get('errors', 0)
        skipped = stats.get('skipped', 0)

        msg_parts = [f"✅ Indexed {total_indexed} files"]
        if errors > 0:
            msg_parts.append(f"{errors} errors")
        if skipped > 0:
            msg_parts.append(f"{skipped} skipped")

        self.status_bar.showMessage(", ".join(msg_parts))
        self.index_btn.setEnabled(True)
        self.index_btn.setText("📚 Index Files")
        self.progress_bar.setVisible(False)

    def _on_indexing_error(self, error_msg):
        self.status_bar.showMessage(f"❌ Indexing failed: {error_msg}")
        print(f"Indexing error: {error_msg}")
        self.index_btn.setEnabled(True)
        self.index_btn.setText("📚 Index Files")
        self.progress_bar.setVisible(False)

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

        # Hybrid search mode
        self.results_title.setText(f'🔍 HYBRID RESULTS  ·  "{query}"  ·  {len(filtered)} files')
        self.status_bar.showMessage(f"🔍 Hybrid search in {len(filtered)} files...")

        # Create and start hybrid search worker
        self.search_worker = HybridSearchWorker(
            self.search_engine,
            query,
            filtered,  # Pass the filtered files to the worker
            self.max_files_spin.value()
        )
        self.search_worker.search_completed.connect(self._on_hybrid_results)
        self.search_worker.progress_update.connect(self._on_search_progress)
        self.search_worker.start()

    def _on_hybrid_results(self, results):
        """Handle hybrid search results."""
        self._stop_search()

        if not results:
            self.status_bar.showMessage("No matching results found")
            no_results_label = QLabel("No relevant matches found for your query.\nTry different keywords or add more files.")
            no_results_label.setFont(QFont("Segoe UI, Arial, sans-serif", 11))
            no_results_label.setStyleSheet(f"color: {DARK['text_dim']}; padding: 24px;")
            no_results_label.setAlignment(Qt.AlignCenter)
            self.results_layout.addWidget(no_results_label)
            return

        # Group results by type for display
        text_results = [r for r in results if r.search_type in ('text', 'hybrid')]
        semantic_results = [r for r in results if r.search_type == 'semantic']

        # Display text results first (on top)
        if text_results:
            header_text = QLabel("📄 DIRECT TEXT MATCHES")
            header_text.setFont(QFont("Consolas, Courier New, monospace", 9, QFont.Bold))
            header_text.setStyleSheet(f"color: {DARK['accent']}; padding: 8px 0 4px 0;")
            self.results_layout.addWidget(header_text)

            for result in text_results[:10]:  # Limit text results
                card = self._create_result_card(result)
                self.results_layout.addWidget(card)

        # Display semantic results below
        if semantic_results:
            header_semantic = QLabel("🧠 SEMANTIC MATCHES")
            header_semantic.setFont(QFont("Consolas, Courier New, monospace", 9, QFont.Bold))
            header_semantic.setStyleSheet(f"color: {DARK['accent2']}; padding: 8px 0 4px 0;")
            self.results_layout.addWidget(header_semantic)

            for result in semantic_results[:10]:  # Limit semantic results
                card = self._create_result_card(result)
                self.results_layout.addWidget(card)

        total_results = len(text_results) + len(semantic_results)
        self.status_bar.showMessage(f"✅ Found {total_results} results ({len(text_results)} text, {len(semantic_results)} semantic)")

    def _create_result_card(self, result):
        """Create a result card for hybrid search results."""
        result_id = f"result_{len(self.result_cards)}_{result.file_name}"
        card = EnhancedResultCard(
            result_id=result_id,
            filename=result.file_name,
            file_type=result.file_type,
            score=result.score,
            file_path=result.file_path
        )

        # Set the search result content
        card.set_fast_content(result.snippet)
        self.result_cards[result_id] = card
        return card

    def _on_search_progress(self, current, total):
        """Handle search progress updates."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)

    def _stop_search(self):
        if self.search_worker and self.search_worker.isRunning():
            self.search_worker.stop_search()
            self.search_worker.wait(2000)
        self.progress_bar.setVisible(False)
        self.stop_btn.setVisible(False)

    def _clear_results(self):
        # Remove ALL widgets from results layout (cards, headers, labels) except placeholder
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget and widget is not self.placeholder:
                widget.deleteLater()
        self.result_cards.clear()
        self.results_layout.addWidget(self.placeholder)
        self.placeholder.setVisible(True)
        self.results_title.setText("🔍 HYBRID RESULTS")

    # ── Signal handlers ───────────────────────

