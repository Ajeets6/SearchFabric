"""
Optimized Search Worker with Hybrid Search
"""

from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
from typing import List, Set, Optional
import time

from search.hybrid_search import HybridSearchEngine, SearchMode, SearchResult
from data.file_processor import FileProcessor

class OptimizedSearchWorker(QThread):
    """Background worker thread for hybrid search processing."""

    # Signals for fast results and LLM analysis
    fast_result_ready = pyqtSignal(str, str, str, float, str)  # (result_id, filename, file_type, score, snippet)
    llm_token_received = pyqtSignal(str, str)  # (result_id, token) - for LLM analysis
    result_started = pyqtSignal(str, str, str)  # (result_id, filename, file_type)
    result_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    all_done = pyqtSignal()
    indexing_progress = pyqtSignal(int, int)  # (total, completed)

    def __init__(self, query: str, files: List[Path], model: str, ollama_client,
                 search_mode: SearchMode = SearchMode.HYBRID):
        super().__init__()
        self.query = query
        self.files = files
        self.model = model
        self.client = ollama_client
        self.search_mode = search_mode
        self.search_engine = HybridSearchEngine()
        self._stop = False

        # Track which files need LLM analysis
        self._llm_queue = []

    def stop(self):
        """Signal the worker to stop processing."""
        self._stop = True

    def run(self):
        """Process search with hybrid approach."""
        try:
            # Step 1: Ensure files are indexed
            self._ensure_indexing()

            if self._stop:
                return

            # Step 2: Fast search for immediate results
            fast_results = self._perform_fast_search()

            if self._stop:
                return

            # Step 3: LLM analysis for top results
            self._perform_llm_analysis(fast_results)

        except Exception as e:
            self.error_occurred.emit("search_error", str(e))
        finally:
            self.all_done.emit()

    def _ensure_indexing(self):
        """Ensure all files are indexed for fast search."""
        unindexed_files = []

        for file_path in self.files:
            if file_path.suffix.lower() in {".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".csv", ".xml", ".yaml", ".yml", ".toml", ".rst", ".log"}:
                if self.search_engine.text_indexer.needs_reindex(file_path):
                    unindexed_files.append(file_path)
            else:
                # For media files, check semantic index
                if self.search_engine.semantic_available:
                    file_type, content, _ = FileProcessor.process(file_path)
                    if content and self.search_engine.semantic_indexer.needs_reindex(file_path, content):
                        unindexed_files.append(file_path)

        if unindexed_files:
            # Index in background with progress updates
            def progress_callback(total, completed):
                if not self._stop:
                    self.indexing_progress.emit(total, completed)

            self.search_engine.index_files(unindexed_files, progress_callback)

    def _perform_fast_search(self) -> List[SearchResult]:
        """Perform fast search and emit immediate results."""
        file_types = self._get_enabled_file_types()
        results = self.search_engine.search(
            self.query,
            mode=self.search_mode,
            file_types=file_types,
            limit=20  # Limit for performance
        )

        # Emit fast results immediately
        for i, result in enumerate(results):
            if self._stop:
                break

            result_id = f"result_{i}_{Path(result.file_path).stem}"

            # Emit fast result
            self.fast_result_ready.emit(
                result_id,
                result.file_name,
                result.file_type,
                result.score,
                result.snippet
            )

            # Add to LLM analysis queue if score is high enough
            if result.score > 0.3:  # Threshold for LLM analysis
                self._llm_queue.append((result_id, result))

        return results

    def _perform_llm_analysis(self, search_results: List[SearchResult]):
        """Perform LLM analysis on top search results."""
        for result_id, result in self._llm_queue[:5]:  # Limit to top 5 for LLM
            if self._stop:
                break

            self.result_started.emit(result_id, result.file_name, result.file_type)

            # Get full content for LLM analysis
            file_path = Path(result.file_path)
            file_type, content, images = FileProcessor.process(file_path)

            if not content:
                self.result_done.emit(result_id)
                continue

            # Create enhanced prompt with search context
            if file_type == "image":
                prompt = (
                    f'The user searched for: "{self.query}"\n\n'
                    f'This image was found with relevance score {result.score:.2f}.\n'
                    f'Analyze this image and explain specifically how it relates to "{self.query}". '
                    f'Be precise and mention visual elements. (2-3 sentences)'
                )
            else:
                prompt = (
                    f'The user searched for: "{self.query}"\n\n'
                    f'This file was found with relevance score {result.score:.2f}.\n'
                    f'File: {result.file_name}\nContent:\n{content}\n\n'
                    f'Explain specifically how this content relates to "{self.query}". '
                    f'Quote relevant parts and be concise. (3-4 sentences)'
                )

            # Stream LLM analysis
            def on_chunk(token, rid=result_id):
                if not self._stop:
                    self.llm_token_received.emit(rid, token)

            def on_done(rid=result_id):
                self.result_done.emit(rid)

            def on_error(err, rid=result_id):
                self.error_occurred.emit(rid, err)

            self.client.stream_query(
                model=self.model, prompt=prompt, images=images,
                on_chunk=on_chunk, on_done=on_done, on_error=on_error,
            )

    def _get_enabled_file_types(self) -> Set[str]:
        """Get enabled file types based on available files."""
        types = set()
        for file_path in self.files:
            suffix = file_path.suffix.lower()
            if suffix in {".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".csv", ".xml", ".yaml", ".yml", ".toml", ".rst", ".log"}:
                types.add("text")
            elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                types.add("image")
            elif suffix in {".pdf"}:
                types.add("pdf")
        return types