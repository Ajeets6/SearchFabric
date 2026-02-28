from pathlib import Path
class FileProcessor:
    @staticmethod
    def load_text(path: Path) -> str:
        try:
            return path.read_text(errors="replace")[:8000]
        except Exception as e:
            return f"[Error reading file: {e}]"

    @staticmethod
    def load_image_b64(path: Path) -> str | None:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except:
            return None

    @staticmethod
    def load_pdf_text(path: Path) -> str:
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
                return "[PDF support requires: pip install pdfplumber or PyPDF2]"
        except Exception as e:
            return f"[Error reading PDF: {e}]"

    @classmethod
    def process(cls, path: Path):
        """Returns (content_type, content, images_b64)"""
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT:
            return ("text", cls.load_text(path), None)
        elif suffix in SUPPORTED_IMAGE:
            b64 = cls.load_image_b64(path)
            return ("image", f"[Image: {path.name}]", [b64] if b64 else None)
        elif suffix in SUPPORTED_PDF:
            return ("pdf", cls.load_pdf_text(path), None)
        else:
            return ("unknown", None, None)
