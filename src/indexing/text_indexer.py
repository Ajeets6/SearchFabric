"""
Fast Text Indexing and Grep-based Search
"""

import re
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict
import threading
import time
from contextlib import contextmanager

class TextIndexer:
    """Fast text search using grep patterns and SQLite FTS."""

    def __init__(self, db_path: str = "search_index.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()

    def _init_db(self):
        """Initialize SQLite FTS tables."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS text_content USING fts5(
                    file_path, file_name, content, last_modified
                );
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    file_size INTEGER,
                    last_modified REAL,
                    file_type TEXT,
                    indexed_at REAL
                );
            """)
            conn.commit()

    def index_file(self, file_path: Path, content: str, file_type: str):
        """Index a single text file."""
        stat = file_path.stat()
        with self._get_connection() as conn:
            # Remove existing entry
            conn.execute("DELETE FROM text_content WHERE file_path = ?", (str(file_path),))
            conn.execute("DELETE FROM file_metadata WHERE file_path = ?", (str(file_path),))

            # Insert new content
            conn.execute("""
                INSERT INTO text_content (file_path, file_name, content, last_modified)
                VALUES (?, ?, ?, ?)
            """, (str(file_path), file_path.name, content, stat.st_mtime))

            conn.execute("""
                INSERT INTO file_metadata (file_path, file_size, last_modified, file_type, indexed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (str(file_path), stat.st_size, stat.st_mtime, file_type, time.time()))

            conn.commit()

    def index_files_batch(self, files: list):
        """Index multiple text files in a single transaction.

        Args:
            files: List of (file_path, content, file_type) tuples
        """
        with self._get_connection() as conn:
            for file_path, content, file_type in files:
                try:
                    stat = file_path.stat()
                    conn.execute("DELETE FROM text_content WHERE file_path = ?", (str(file_path),))
                    conn.execute("DELETE FROM file_metadata WHERE file_path = ?", (str(file_path),))
                    conn.execute("""
                        INSERT INTO text_content (file_path, file_name, content, last_modified)
                        VALUES (?, ?, ?, ?)
                    """, (str(file_path), file_path.name, content, stat.st_mtime))
                    conn.execute("""
                        INSERT INTO file_metadata (file_path, file_size, last_modified, file_type, indexed_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (str(file_path), stat.st_size, stat.st_mtime, file_type, time.time()))
                except Exception as e:
                    print(f"Batch index error for {file_path}: {e}")
            conn.commit()

    def search_text(self, query: str, limit: int = 50) -> List[Tuple[str, str, float]]:
        """
        Fast text search using SQLite FTS.

        Returns:
            List of (file_path, snippet, score) tuples
        """
        # Prepare FTS query
        fts_query = self._prepare_fts_query(query)

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT file_path, file_name, snippet(text_content, 2, '<mark>', '</mark>', '...', 64) as snippet,
                       rank
                FROM text_content
                WHERE text_content MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, limit))

            return [(row[0], row[1], row[2], -row[3]) for row in cursor.fetchall()]

    def _prepare_fts_query(self, query: str) -> str:
        """Convert natural query to FTS5 syntax."""
        # Handle phrases and individual terms
        words = re.findall(r'"[^"]*"|\S+', query)
        fts_terms = []

        for word in words:
            if word.startswith('"') and word.endswith('"'):
                # Phrase search
                fts_terms.append(word)
            else:
                # Individual word with prefix matching
                fts_terms.append(f'"{word}"*')

        return " OR ".join(fts_terms)

    def needs_reindex(self, file_path: Path) -> bool:
        """Check if file needs reindexing."""
        if not file_path.exists():
            return False

        stat = file_path.stat()
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT last_modified FROM file_metadata WHERE file_path = ?
            """, (str(file_path),))
            row = cursor.fetchone()

            return row is None or row[0] != stat.st_mtime

    def get_indexed_files(self) -> List[str]:
        """Get all indexed file paths."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT file_path FROM file_metadata")
            return [row[0] for row in cursor.fetchall()]