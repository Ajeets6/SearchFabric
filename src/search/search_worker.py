"""
Search Worker Thread
"""

from PyQt5.QtCore import QThread, pyqtSignal

from data.file_processor import FileProcessor


class SearchWorker(QThread):
    """Background worker thread for processing search queries."""

    token_received = pyqtSignal(str, str)       # (result_id, token)
    result_started = pyqtSignal(str, str, str)  # (result_id, filename, file_type)
    result_done = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    all_done = pyqtSignal()

    def __init__(self, query, files, model, ollama_client):
        super().__init__()
        self.query = query
        self.files = files
        self.model = model
        self.client = ollama_client
        self._stop = False

    def stop(self):
        """Signal the worker to stop processing."""
        self._stop = True

    def run(self):
        """Process each file and stream results."""
        for i, fpath in enumerate(self.files):
            if self._stop:
                break
            result_id = f"result_{i}_{fpath.stem}"
            ctype, content, images = FileProcessor.process(fpath)
            if content is None:
                continue

            self.result_started.emit(result_id, fpath.name, ctype)

            if ctype == "image":
                prompt = (
                    f'The user is searching for: "{self.query}"\n\n'
                    f"Analyze this image and explain how it relates to the search query. "
                    f"Be concise (2-3 sentences). If not relevant, say so briefly."
                )
            else:
                prompt = (
                    f'The user is searching for: "{self.query}"\n\n'
                    f"File: {fpath.name}\nContent (excerpt):\n{content}\n\n"
                    f"Explain how this file relates to the search query. "
                    f"Highlight the most relevant parts. Be concise (3-4 sentences)."
                )

            def on_chunk(token, rid=result_id):
                if not self._stop:
                    self.token_received.emit(rid, token)

            def on_done(rid=result_id):
                self.result_done.emit(rid)

            def on_error(err, rid=result_id):
                self.error_occurred.emit(rid, err)

            self.client.stream_query(
                model=self.model, prompt=prompt, images=images,
                on_chunk=on_chunk, on_done=on_done, on_error=on_error,
            )
        self.all_done.emit()
