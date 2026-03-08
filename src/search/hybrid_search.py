"""
Hybrid Search Engine combining text search and semantic search
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time
import threading

from indexing.text_indexer import TextIndexer
from indexing.semantic_indexer import SemanticIndexer
from data.file_processor import FileProcessor, SUPPORTED_TEXT, SUPPORTED_IMAGE, SUPPORTED_PDF

class SearchMode(Enum):
    FAST = "fast"           # Text search only
    SEMANTIC = "semantic"   # Embeddings only
    HYBRID = "hybrid"       # Both text and semantic

@dataclass
class SearchResult:
    """Unified search result."""
    file_path: str
    file_name: str
    file_type: str
    score: float
    snippet: str
    search_type: str  # "text", "semantic", "hybrid"

class HybridSearchEngine:
    """Fast hybrid search combining text indexing and semantic embeddings."""

    def __init__(self, text_db_path: str = "text_index.db",
                 semantic_db_path: str = "semantic_index.db"):
        self.text_indexer = TextIndexer(text_db_path)

        # Initialize semantic indexer with error handling
        try:
            self.semantic_indexer = SemanticIndexer(semantic_db_path)
            self.semantic_available = True
        except ImportError:
            print("Warning: Semantic search unavailable. Install sentence-transformers for full functionality.")
            self.semantic_indexer = None
            self.semantic_available = False

        self._indexing_lock = threading.Lock()

    def index_files(self, file_paths: List[Path], progress_callback=None) -> Dict[str, int]:
        """
        Index multiple files efficiently with batched processing.

        Returns:
            Dict with counts: {"text": 5, "semantic": 3, "errors": 1}
        """
        stats = {"text": 0, "semantic": 0, "errors": 0, "skipped": 0}

        # Separate files by type for different indexing strategies
        text_files = []
        media_files = []  # Images and PDFs that need semantic indexing

        for path in file_paths:
            if not path.exists():
                stats["errors"] += 1
                continue

            suffix = path.suffix.lower()
            if suffix in SUPPORTED_TEXT:
                text_files.append(path)
            elif suffix in SUPPORTED_IMAGE or suffix in SUPPORTED_PDF:
                media_files.append(path)

        # Batch process text files for efficiency
        BATCH_SIZE = 100
        batch_items = []  # (path, content, file_type) tuples for batch insert

        for i, path in enumerate(text_files):
            try:
                if not self.text_indexer.needs_reindex(path):
                    stats["skipped"] += 1
                else:
                    file_type, content, _ = FileProcessor.process(path)
                    if content and not content.startswith("[Error"):
                        batch_items.append((path, content, file_type))
                    else:
                        stats["skipped"] += 1
            except Exception as e:
                print(f"Error reading text file {path}: {e}")
                stats["errors"] += 1

            # Flush batch when full
            if len(batch_items) >= BATCH_SIZE:
                self.text_indexer.index_files_batch(batch_items)
                stats["text"] += len(batch_items)
                batch_items = []

            if progress_callback:
                progress_callback(len(text_files) + len(media_files),
                                i + 1)

        # Flush remaining batch
        if batch_items:
            self.text_indexer.index_files_batch(batch_items)
            stats["text"] += len(batch_items)
            batch_items = []

        # Index media files (requires content extraction, slower)
        for i, path in enumerate(media_files):
            try:
                if self._index_media_file(path):
                    stats["semantic"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                print(f"Error indexing media file {path}: {e}")
                stats["errors"] += 1

            if progress_callback:
                progress_callback(len(text_files) + len(media_files),
                                len(text_files) + i + 1)

        return stats

    def _index_text_file(self, path: Path) -> bool:
        """Index a single text file."""
        if not self.text_indexer.needs_reindex(path):
            return False  # Already indexed

        file_type, content, _ = FileProcessor.process(path)
        if content and not content.startswith("[Error"):
            self.text_indexer.index_file(path, content, file_type)
            return True
        return False

    def _index_media_file(self, path: Path) -> bool:
        """Index a media file (image/PDF) with semantic embeddings."""
        if not self.semantic_available:
            return False

        file_type, content, images = FileProcessor.process(path)

        # For images, we'll need to generate descriptions using Ollama later
        # For now, use filename and basic metadata as description
        if file_type == "image":
            description = f"Image file: {path.name}. {content if content else ''}"
        elif file_type == "pdf":
            description = content if content and not content.startswith("[Error") else f"PDF file: {path.name}"
        else:
            return False

        if not self.semantic_indexer.needs_reindex(path, description):
            return False  # Already indexed

        return self.semantic_indexer.index_content(path, description, file_type)

    def search(self, query: str, mode: SearchMode = SearchMode.HYBRID,
               file_types: Optional[Set[str]] = None, limit: int = 50) -> List[SearchResult]:
        """
        Unified search interface.

        Args:
            query: Search query
            mode: Search mode (fast, semantic, hybrid)
            file_types: Filter by file types {"text", "image", "pdf"}
            limit: Maximum results

        Returns:
            Ranked list of SearchResult objects
        """
        results = []

        # Text search for fast results
        if mode in (SearchMode.FAST, SearchMode.HYBRID):
            if not file_types or "text" in file_types:
                text_results = self.text_indexer.search_text(query, limit)
                for file_path, file_name, snippet, score in text_results:
                    results.append(SearchResult(
                        file_path=file_path,
                        file_name=file_name,
                        file_type="text",
                        score=score,
                        snippet=snippet,
                        search_type="text"
                    ))

        # Semantic search for images and PDFs
        if mode in (SearchMode.SEMANTIC, SearchMode.HYBRID) and self.semantic_available:
            semantic_file_types = []
            if not file_types or "image" in file_types:
                semantic_file_types.append("image")
            if not file_types or "pdf" in file_types:
                semantic_file_types.append("pdf")

            if semantic_file_types:
                semantic_results = self.semantic_indexer.search_semantic(
                    query, semantic_file_types, threshold=0.25, limit=limit
                )
                for file_path, file_name, description, similarity in semantic_results:
                    # Determine file type from description or path
                    file_type = "image" if Path(file_path).suffix.lower() in SUPPORTED_IMAGE else "pdf"

                    results.append(SearchResult(
                        file_path=file_path,
                        file_name=file_name,
                        file_type=file_type,
                        score=similarity,
                        snippet=description[:200] + "..." if len(description) > 200 else description,
                        search_type="semantic"
                    ))

        # Deduplicate and rank results
        seen_paths = set()
        unique_results = []

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        for result in results:
            if result.file_path not in seen_paths:
                unique_results.append(result)
                seen_paths.add(result.file_path)
                if len(unique_results) >= limit:
                    break

        return unique_results

    def get_indexing_stats(self) -> Dict[str, int]:
        """Get statistics about indexed content."""
        text_files = len(self.text_indexer.get_indexed_files())
        semantic_files = len(self.semantic_indexer.get_indexed_files()) if self.semantic_available else 0

        return {
            "text_files": text_files,
            "semantic_files": semantic_files,
            "total": text_files + semantic_files,
            "semantic_available": self.semantic_available
        }

    def should_use_semantic(self, query: str) -> bool:
        """Heuristic to determine if semantic search would be beneficial."""
        # Use semantic search for:
        # - Visual/conceptual terms
        # - Abstract concepts
        # - Non-literal searches
        conceptual_words = {
            "apple", "apples", "two", "three", "four", "five", "red", "blue", "green",
            "happy", "sad", "beautiful", "dark", "light", "concept", "idea", "similar",
            "like", "looks", "appears", "shows", "displays", "contains"
        }

        query_lower = query.lower()
        return any(word in query_lower for word in conceptual_words)