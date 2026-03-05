"""
Direct Search Worker - Shows actual content matches without LLM analysis
"""

from PyQt5.QtCore import QThread, pyqtSignal
import re
from pathlib import Path

from data.file_processor import FileProcessor


class DirectSearchWorker(QThread):
    """Background worker thread that shows direct content matches."""

    result_found = pyqtSignal(str, str, str, str, float)  # (result_id, filename, file_type, content, score)
    result_started = pyqtSignal(str, str, str)  # (result_id, filename, file_type)
    result_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    all_done = pyqtSignal()

    def __init__(self, query, files, model=None, ollama_client=None):
        super().__init__()
        self.query = query.lower()
        self.files = files
        self.model = model  # Not used in direct search
        self.client = ollama_client  # Not used in direct search
        self._stop = False

    def stop(self):
        """Signal the worker to stop processing."""
        self._stop = True

    def run(self):
        """Process each file and show direct matches."""
        for i, fpath in enumerate(self.files):
            if self._stop:
                break

            result_id = f"result_{i}_{fpath.stem}"
            ctype, content, images = FileProcessor.process(fpath)

            if content is None or content.startswith("[Error"):
                self.error_occurred.emit(result_id, content or "Could not read file")
                continue

            self.result_started.emit(result_id, fpath.name, ctype)

            # Calculate relevance score and extract relevant content
            if ctype == "image":
                # For images, show filename and basic info
                display_content = f"📷 Image file: {fpath.name}\nPath: {fpath}"
                score = 0.5  # Default score for images
            else:
                # For text files, find and highlight matches
                display_content, score = self._extract_relevant_content(content, fpath.name)

            # Emit the direct content
            self.result_found.emit(result_id, fpath.name, ctype, display_content, score)
            self.result_done.emit(result_id)

        self.all_done.emit()

    def _extract_relevant_content(self, content: str, filename: str) -> tuple[str, float]:
        """Extract relevant content based on search query."""
        if not self.query.strip():
            return content[:500] + ("..." if len(content) > 500 else ""), 0.1

        content_lower = content.lower()
        filename_lower = filename.lower()
        query_words = self.query.split()

        # Calculate score based on matches
        score = 0.0
        total_matches = 0

        # Check filename matches
        for word in query_words:
            if word in filename_lower:
                score += 0.3
                total_matches += 1

        # Check content matches
        for word in query_words:
            content_matches = content_lower.count(word)
            if content_matches > 0:
                score += min(content_matches * 0.1, 0.5)  # Cap at 0.5 per word
                total_matches += content_matches

        # Normalize score
        if total_matches > 0:
            score = min(score, 1.0)
        else:
            return f"No matches found for '{self.query}'\n\nFile preview:\n{content[:300]}...", 0.0

        # Extract relevant snippets
        snippets = self._extract_matching_snippets(content, query_words)

        if snippets:
            display_content = f"🔍 Found {total_matches} match(es) for '{self.query}':\n\n"
            display_content += "\n\n".join(snippets)

            # Add file info
            lines = content.count('\n') + 1
            chars = len(content)
            display_content += f"\n\n📄 File Info: {lines} lines, {chars} characters"
        else:
            # Fallback to beginning of content
            display_content = f"📄 Content preview:\n\n{content[:800]}"
            if len(content) > 800:
                display_content += "\n\n... [content truncated] ..."

        return display_content, score

    def _extract_matching_snippets(self, content: str, query_words: list) -> list:
        """Extract text snippets around matching words."""
        snippets = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check if line contains any query words
            for word in query_words:
                if word in line_lower:
                    # Extract snippet with context
                    start_line = max(0, i - 1)
                    end_line = min(len(lines), i + 2)

                    snippet_lines = lines[start_line:end_line]
                    snippet = '\n'.join(snippet_lines)

                    # Highlight matching words (simple approach)
                    for highlight_word in query_words:
                        pattern = re.compile(re.escape(highlight_word), re.IGNORECASE)
                        snippet = pattern.sub(f"**{highlight_word.upper()}**", snippet)

                    snippet = f"Line {i+1}: {snippet}"
                    if snippet not in snippets:  # Avoid duplicates
                        snippets.append(snippet)

                    break  # Found match in this line, move to next line

            # Limit to 5 snippets to avoid overwhelming output
            if len(snippets) >= 5:
                break

        return snippets