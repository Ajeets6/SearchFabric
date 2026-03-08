"""
Enhanced Semantic Embeddings for Images and PDFs
Supports both sentence-transformers (text) and CLIP (images)
"""

import numpy as np
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import json
import threading
import time
from contextlib import contextmanager
import hashlib
from PIL import Image

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import open_clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

class SemanticIndexer:
    """Enhanced semantic search using both text and vision embeddings."""

    def __init__(self, db_path: str = "semantic_index.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self._lock = threading.Lock()

        # Initialize models based on model_name
        self.text_model = None
        self.clip_model = None
        self.clip_preprocess = None

        # Set device safely
        if CLIP_AVAILABLE:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                self.device = "cpu"
        else:
            self.device = None

        self._init_models()
        self._init_db()

    def _init_models(self):
        """Initialize embedding models based on model_name."""
        # Load CLIP for image support if available
        if CLIP_AVAILABLE:
            try:
                model_variant = "ViT-B-32"
                pretrained = "openai"
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_variant, pretrained=pretrained, device=self.device
                )
                self.embedding_dim = self.clip_model.visual.output_dim
                print(f"Loaded CLIP model: {model_variant} ({pretrained}) on {self.device}")
            except Exception as e:
                print(f"CLIP initialization failed: {e}")
                self.clip_model = None
                self.clip_preprocess = None

        # Load SentenceTransformer for text/PDF embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                text_model_name = self.model_name if "clip" not in self.model_name.lower() else "all-MiniLM-L6-v2"
                device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu") if CLIP_AVAILABLE else None
                self.text_model = SentenceTransformer(text_model_name, device=device_str)
                self.embedding_dim = self.text_model.get_sentence_embedding_dimension()
                print(f"Loaded text model: {text_model_name} on {device_str or 'cpu'}")
            except Exception as e:
                print(f"Text model initialization failed: {e}")
                self.text_model = None

        if self.text_model is None and self.clip_model is None:
            raise ImportError("No embedding models available - check your dependencies")

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
        Index semantic content with appropriate model.

        Args:
            file_path: Path to the file
            description: Text description/content to embed OR path for images with CLIP
            file_type: 'image', 'pdf', etc.

        Returns:
            True if indexing was successful
        """
        try:
            content_hash = self._get_content_hash(str(file_path) + description)

            # Check if already indexed with same content
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT content_hash FROM embeddings WHERE file_path = ?
                """, (str(file_path),))
                row = cursor.fetchone()

                if row and row[0] == content_hash:
                    return True  # Already indexed with same content

            # Generate embedding based on file type and available models
            embedding = self._generate_embedding(file_path, description, file_type)
            if embedding is None:
                return False

            embedding_blob = embedding.tobytes()
            stat = file_path.stat()

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO embeddings
                    (file_path, file_name, file_type, content_hash, description, embedding, last_modified, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(file_path), file_path.name, file_type, content_hash,
                    description, embedding_blob, stat.st_mtime, time.time()
                ))
                conn.commit()

            return True

        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            return False

    def _generate_embedding(self, file_path: Path, content: str, file_type: str) -> Optional[np.ndarray]:
        """Generate embeddings using appropriate model."""
        try:
            # Use CLIP for images if available and it's an image file
            if (file_type == 'image' and self.clip_model is not None and
                file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}):

                # Load and preprocess image
                image = Image.open(file_path).convert('RGB')
                image_preprocessed = self.clip_preprocess(image).unsqueeze(0).to(self.device)

                # Generate image embedding
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_preprocessed)
                    embedding = image_features.cpu().numpy().astype(np.float32).flatten()

                return embedding

            # Use text model for all other content (including image descriptions)
            elif self.text_model is not None:
                embedding = self.text_model.encode(content, convert_to_numpy=True)
                return embedding.astype(np.float32)

            # Fallback: try CLIP text encoding if available
            elif self.clip_model is not None:
                text_tokens = open_clip.tokenize([content]).to(self.device)
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_tokens)
                    embedding = text_features.cpu().numpy().astype(np.float32).flatten()
                return embedding

            else:
                print(f"No suitable model available for {file_type}")
                return None

        except Exception as e:
            print(f"Embedding generation error for {file_path}: {e}")
            return None

    def _generate_text_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate query embedding using SentenceTransformer (for text/PDF space)."""
        try:
            if self.text_model is not None:
                embedding = self.text_model.encode(query, convert_to_numpy=True)
                return embedding.astype(np.float32)
            return None
        except Exception as e:
            print(f"Text query embedding error: {e}")
            return None

    def _generate_clip_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate query embedding using CLIP text encoder (for image space)."""
        try:
            if self.clip_model is not None:
                text_tokens = open_clip.tokenize([query]).to(self.device)
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_tokens)
                    embedding = text_features.cpu().numpy().astype(np.float32).flatten()
                return embedding
            return None
        except Exception as e:
            print(f"CLIP query embedding error: {e}")
            return None

    def search_semantic(self, query: str, file_types: Optional[List[str]] = None,
                       threshold: float = 0.5, limit: int = 20) -> List[Tuple[str, str, str, float]]:
        """
        Semantic similarity search using appropriate embedding model.

        Searches each embedding space separately to avoid dimension mismatches:
        - Images indexed with CLIP are queried with CLIP text encoder
        - PDFs/text indexed with SentenceTransformer are queried with SentenceTransformer
        """
        results = []
        search_types = file_types or ['image', 'pdf']

        # Determine which types use which embedding space
        clip_types = [ft for ft in search_types if ft == 'image' and self.clip_model is not None]
        text_types = [ft for ft in search_types if ft not in clip_types]

        try:
            with self._get_connection() as conn:
                # Search CLIP embedding space (images)
                if clip_types:
                    clip_query = self._generate_clip_query_embedding(query)
                    if clip_query is not None:
                        results.extend(self._search_embedding_space(
                            conn, clip_query, clip_types, threshold
                        ))

                # Search text embedding space (PDFs and fallback images)
                if text_types:
                    text_query = self._generate_text_query_embedding(query)
                    if text_query is not None:
                        results.extend(self._search_embedding_space(
                            conn, text_query, text_types, threshold
                        ))

                # Also search images with text model if CLIP wasn't available during indexing
                if 'image' in clip_types and self.text_model is not None:
                    text_query = self._generate_text_query_embedding(query)
                    if text_query is not None:
                        results.extend(self._search_embedding_space(
                            conn, text_query, ['image'], threshold, expect_dim=self.text_model.get_sentence_embedding_dimension()
                        ))

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

        # Deduplicate by file_path, keep highest score
        seen = {}
        for r in results:
            if r[0] not in seen or r[3] > seen[r[0]][3]:
                seen[r[0]] = r

        results = sorted(seen.values(), key=lambda x: x[3], reverse=True)
        return results[:limit]

    def _search_embedding_space(self, conn, query_embedding: np.ndarray,
                                 file_types: List[str], threshold: float,
                                 expect_dim: Optional[int] = None) -> List[Tuple[str, str, str, float]]:
        """Search embeddings that match the query embedding's dimension."""
        query_dim = expect_dim or len(query_embedding)
        placeholders = ','.join(['?' for _ in file_types])
        sql = f"""
            SELECT file_path, file_name, description, embedding, file_type
            FROM embeddings
            WHERE file_type IN ({placeholders})
        """
        cursor = conn.execute(sql, file_types)

        results = []
        for row in cursor:
            file_path, file_name, description, embedding_blob, file_type = row
            stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

            # Skip if embedding dimensions don't match (different model space)
            if len(stored_embedding) != len(query_embedding):
                continue

            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            if similarity >= threshold:
                results.append((file_path, file_name, description, similarity))

        return results

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
        content_hash = self._get_content_hash(str(file_path) + description)

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