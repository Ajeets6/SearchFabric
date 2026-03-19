"""
RAM++-first semantic indexer.

Images are indexed as RAM++ tags plus optional context text and stored in the
same semantic embedding space as query text. This keeps retrieval low-latency
while improving image keyword precision over CLIP-only indexing.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ram_plus_tagger import RAMPlusTagger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticIndexer:
    """Semantic search using sentence embeddings with RAM++ tag enrichment."""

    def __init__(self, db_path: str = "semantic_index.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self._lock = threading.Lock()

        self.text_model = None
        self.embedding_dim = 0
        self.ram_tagger = RAMPlusTagger()
        self.last_tagging_error: Optional[str] = None

        self._init_models()
        self._init_db()

    def _init_models(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for semantic indexing")

        try:
            self.text_model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.text_model.get_sentence_embedding_dimension()
            print(f"Loaded semantic model: {self.model_name}")
        except Exception as exc:
            raise ImportError(f"Failed to load semantic model '{self.model_name}': {exc}") from exc

    @contextmanager
    def _get_connection(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_def: str) -> None:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    file_path TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_type TEXT,
                    content_hash TEXT,
                    description TEXT,
                    tags_json TEXT,
                    embedding BLOB,
                    last_modified REAL,
                    indexed_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_file_type ON embeddings(file_type);
                CREATE INDEX IF NOT EXISTS idx_content_hash ON embeddings(content_hash);
                """
            )

            self._ensure_column(conn, "embeddings", "tags_json", "TEXT")
            conn.commit()

    def _get_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]

    def _get_source_hash(self, file_path: Path, description: str, file_type: str) -> str:
        checkpoint_marker = ""
        if file_type == "image":
            checkpoint_marker = self.ram_tagger.checkpoint_path or "fallback-tags"
        return self._get_content_hash(f"{file_path}|{file_type}|{description}|{checkpoint_marker}")

    def _normalize_text(self, value: str, max_chars: int = 8000) -> str:
        if not value:
            return ""
        text = " ".join(value.split())
        return text[:max_chars]

    def _fallback_tags(self, file_path: Path) -> List[str]:
        stem_tokens = [part.strip().lower() for part in file_path.stem.replace("_", " ").replace("-", " ").split()]
        ext_token = file_path.suffix.lower().replace(".", "")
        tags = [token for token in stem_tokens if len(token) >= 2]
        if ext_token:
            tags.append(ext_token)
        if "image" not in tags:
            tags.append("image")
        return list(dict.fromkeys(tags))

    def _build_image_descriptor(self, file_path: Path, base_description: str) -> Tuple[str, List[str]]:
        tags, error = self.ram_tagger.generate_tags(file_path)
        if error:
            self.last_tagging_error = error
            tags = self._fallback_tags(file_path)
        else:
            self.last_tagging_error = None

        base = self._normalize_text(base_description, max_chars=1200)
        tag_text = ", ".join(tags)
        descriptor = f"image tags: {tag_text}. filename: {file_path.name}."
        if base:
            descriptor = f"{descriptor} context: {base}"
        return descriptor, tags

    def _generate_embedding(self, content: str) -> Optional[np.ndarray]:
        try:
            vec = self.text_model.encode(content, convert_to_numpy=True)
            return vec.astype(np.float32)
        except Exception as exc:
            print(f"Embedding generation failed: {exc}")
            return None

    def index_content(self, file_path: Path, description: str, file_type: str) -> bool:
        try:
            tags: List[str] = []
            final_description = self._normalize_text(description)

            if file_type == "image":
                final_description, tags = self._build_image_descriptor(file_path, description)

            source_hash = self._get_source_hash(file_path, description, file_type)

            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT content_hash FROM embeddings WHERE file_path = ?",
                    (str(file_path),),
                ).fetchone()
                if row and row[0] == source_hash:
                    return True

            embedding = self._generate_embedding(final_description)
            if embedding is None:
                return False

            stat = file_path.stat()
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (file_path, file_name, file_type, content_hash, description, tags_json, embedding, last_modified, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(file_path),
                        file_path.name,
                        file_type,
                        source_hash,
                        final_description,
                        json.dumps(tags) if tags else None,
                        embedding.tobytes(),
                        stat.st_mtime,
                        time.time(),
                    ),
                )
                conn.commit()
            return True
        except Exception as exc:
            print(f"Error indexing {file_path}: {exc}")
            return False

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def _keyword_overlap_score(self, query_tokens: List[str], tags: List[str]) -> float:
        if not query_tokens or not tags:
            return 0.0
        tag_tokens = set()
        for tag in tags:
            for token in tag.lower().replace("_", " ").split():
                if token:
                    tag_tokens.add(token)
        if not tag_tokens:
            return 0.0
        overlap = sum(1 for token in query_tokens if token in tag_tokens)
        return min(0.25, overlap * 0.08)

    def search_semantic(
        self,
        query: str,
        file_types: Optional[List[str]] = None,
        threshold: float = 0.25,
        limit: int = 20,
    ) -> List[Tuple[str, str, str, float]]:
        try:
            query_embedding = self._generate_embedding(self._normalize_text(query, max_chars=512))
            if query_embedding is None:
                return []

            search_types = file_types or ["image", "pdf"]
            with self._get_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT file_path, file_name, description, tags_json, embedding
                    FROM embeddings
                    WHERE file_type IN ({','.join(['?' for _ in search_types])})
                    """,
                    search_types,
                ).fetchall()

            query_tokens = [token.lower() for token in query.split() if token.strip()]
            results: List[Tuple[str, str, str, float]] = []

            for file_path, file_name, description, tags_json, embedding_blob in rows:
                stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                if len(stored_embedding) != len(query_embedding):
                    continue

                similarity = self._cosine_similarity(query_embedding, stored_embedding)

                if tags_json and query_tokens:
                    try:
                        tags = json.loads(tags_json)
                        if isinstance(tags, list):
                            similarity += self._keyword_overlap_score(query_tokens, tags)
                    except Exception:
                        pass

                if similarity >= threshold:
                    results.append((file_path, file_name, description, similarity))

            deduped: Dict[str, Tuple[str, str, str, float]] = {}
            for row in results:
                if row[0] not in deduped or row[3] > deduped[row[0]][3]:
                    deduped[row[0]] = row

            sorted_results = sorted(deduped.values(), key=lambda item: item[3], reverse=True)
            return sorted_results[:limit]
        except Exception as exc:
            print(f"Semantic search failed: {exc}")
            return []

    def get_indexed_files(self, file_type: Optional[str] = None) -> List[str]:
        with self._get_connection() as conn:
            if file_type:
                rows = conn.execute(
                    "SELECT file_path FROM embeddings WHERE file_type = ?",
                    (file_type,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT file_path FROM embeddings").fetchall()
        return [row[0] for row in rows]

    def needs_reindex(self, file_path: Path, description: str) -> bool:
        suffix = file_path.suffix.lower()
        file_type = "image" if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"} else "pdf"
        expected_hash = self._get_source_hash(file_path, description, file_type)
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT content_hash, last_modified FROM embeddings WHERE file_path = ?",
                (str(file_path),),
            ).fetchone()

        if row is None:
            return True

        content_hash, last_modified = row
        if content_hash != expected_hash:
            return True

        try:
            if file_path.stat().st_mtime == last_modified:
                return False
        except Exception:
            pass
        return True

    def remove_file(self, file_path: Path):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM embeddings WHERE file_path = ?", (str(file_path),))
            conn.commit()

    def get_backend_status(self) -> Dict[str, Optional[str]]:
        """Expose model health for UI status messaging."""
        return {
            "embedding_model": self.model_name,
            "ram_checkpoint": self.ram_tagger.checkpoint_path,
            "ram_ready": self.ram_tagger.is_ready,
            "ram_error": self.ram_tagger.last_error,
        }


EMBEDDING_MODELS = {
    "fast": "all-MiniLM-L6-v2",
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "accurate": "all-mpnet-base-v2",
}