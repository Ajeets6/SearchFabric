"""
File Processing Utilities
"""

import base64

# File size limit in MB
MAX_FILE_SIZE_MB = 20

# Supported file extensions
SUPPORTED_TEXT = {".txt", ".md", ".py", ".js", ".ts", ".html", ".css",
                  ".json", ".csv", ".xml", ".yaml", ".yml", ".toml", ".rst", ".log"}
SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
SUPPORTED_PDF = {".pdf"}


class FileProcessor:
    """Utility class for processing different file types."""

    @staticmethod
    def load_text(path):
        """Load text content from a file."""
        try:
            return path.read_text(errors="replace")[:8000]
        except Exception as e:
            return f"[Error reading file: {e}]"

    @staticmethod
    def load_image_b64(path):
        """Load image as base64 encoded string."""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except:
            return None

    @staticmethod
    def load_pdf_text(path):
        """Extract text from PDF file."""
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n\n".join(
                    p.extract_text() or "" for p in pdf.pages[:10]
                )
            return text[:8000] or "[No text extracted from PDF]"
        except ImportError:
            try:
                import PyPDF2
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(
                        reader.pages[i].extract_text() or ""
                        for i in range(min(10, len(reader.pages)))
                    )
                return text[:8000] or "[No text extracted from PDF]"
            except ImportError:
                return "[PDF support: pip install pdfplumber]"
        except Exception as e:
            return f"[Error reading PDF: {e}]"

    @classmethod
    def process(cls, path):
        """
        Process a file and return its content.

        Args:
            path: Path object to the file

        Returns:
            Tuple of (file_type, content, images_list)
        """
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT:
            return ("text", cls.load_text(path), None)
        elif suffix in SUPPORTED_IMAGE:
            b64 = cls.load_image_b64(path)
            return ("image", f"[Image: {path.name}]", [b64] if b64 else None)
        elif suffix in SUPPORTED_PDF:
            return ("pdf", cls.load_pdf_text(path), None)
        return ("unknown", None, None)
