"""
Semantic Embeddings for Images and PDFs
"""

import numpy as np
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import threading
from contextlib import contextmanager
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class SemanticIndexer:
    """Semantic search using sentence transformers embeddings."""

    def __init__(self, db_path: str = "semantic_index.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self._lock = threading.Lock()

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")

        # Use a lightweight, fast model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            # Enable JSON extension if available
            try:
                conn.enable_load_extension(True)
            except AttributeError:
                pass
            try:
                yield conn
            finally:
                conn.close()

    def _init_db(self):
        """Initialize embedding storage tables."""
        with self._get_connection() as conn:
            conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    file_path TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_type TEXT,
                    content_hash TEXT,
                    description TEXT,
                    embedding BLOB,
                    last_modified REAL,
                    indexed_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_file_type ON embeddings(file_type);
                CREATE INDEX IF NOT EXISTS idx_content_hash ON embeddings(content_hash);
            """)
            conn.commit()

    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def index_content(self, file_path: Path, description: str, file_type: str) -> bool:
        """
        Index semantic content (image descriptions, PDF text).

        Args:
            file_path: Path to the file
            description: Text description/content to embed
            file_type: 'image', 'pdf', etc.

        Returns:
            True if indexing was successful
        """
        try:
            content_hash = self._get_content_hash(description)

            # Check if already indexed with same content
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT content_hash FROM embeddings WHERE file_path = ?
                """, (str(file_path),))
                row = cursor.fetchone()

                if row and row[0] == content_hash:
                    return True  # Already indexed with same content

            # Generate embedding
            embedding = self.model.encode(description, convert_to_numpy=True)
            embedding_blob = embedding.tobytes()

            stat = file_path.stat()

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO embeddings
                    (file_path, file_name, file_type, content_hash, description, embedding, last_modified, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(file_path), file_path.name, file_type, content_hash,
                    description, embedding_blob, stat.st_mtime, threading.time.time()
                ))
                conn.commit()

            return True

        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            return False

    def search_semantic(self, query: str, file_types: Optional[List[str]] = None,
                       threshold: float = 0.5, limit: int = 20) -> List[Tuple[str, str, str, float]]:
        """
        Semantic similarity search.

        Args:
            query: Search query
            file_types: Filter by file types (e.g., ['image', 'pdf'])
            threshold: Minimum similarity threshold
            limit: Maximum results

        Returns:
            List of (file_path, file_name, description, similarity_score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query, convert_to_numpy=True)

            # Get all embeddings from database
            with self._get_connection() as conn:
                if file_types:
                    placeholders = ','.join(['?' for _ in file_types])
                    sql = f"""
                        SELECT file_path, file_name, description, embedding, file_type
                        FROM embeddings
                        WHERE file_type IN ({placeholders})
                    """
                    cursor = conn.execute(sql, file_types)
                else:
                    cursor = conn.execute("""
                        SELECT file_path, file_name, description, embedding, file_type
                        FROM embeddings
                    """)

                results = []
                for row in cursor:
                    file_path, file_name, description, embedding_blob, file_type = row

                    # Convert blob back to numpy array
                    stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)

                    if similarity >= threshold:
                        results.append((file_path, file_name, description, similarity))

                # Sort by similarity descending
                results.sort(key=lambda x: x[3], reverse=True)
                return results[:limit]

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_indexed_files(self, file_type: Optional[str] = None) -> List[str]:
        """Get indexed file paths, optionally filtered by type."""
        with self._get_connection() as conn:
            if file_type:
                cursor = conn.execute("""
                    SELECT file_path FROM embeddings WHERE file_type = ?
                """, (file_type,))
            else:
                cursor = conn.execute("SELECT file_path FROM embeddings")

            return [row[0] for row in cursor.fetchall()]

    def needs_reindex(self, file_path: Path, description: str) -> bool:
        """Check if content needs reindexing based on content hash."""
        content_hash = self._get_content_hash(description)

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT content_hash FROM embeddings WHERE file_path = ?
            """, (str(file_path),))
            row = cursor.fetchone()

            return row is None or row[0] != content_hash

    def remove_file(self, file_path: Path):
        """Remove file from semantic index."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM embeddings WHERE file_path = ?", (str(file_path),))
            conn.commit()

# Precomputed embedding models for different use cases
EMBEDDING_MODELS = {
    "fast": "all-MiniLM-L6-v2",           # 384 dim, fast, good for general text
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dim, multiple languages
    "accurate": "all-mpnet-base-v2",       # 768 dim, slower but more accurate
}