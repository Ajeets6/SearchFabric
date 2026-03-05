"""
Ollama API Client
"""

import json
import requests

OLLAMA_BASE = "http://localhost:11434"


class OllamaClient:
    """Client for interacting with the Ollama API."""

    def __init__(self, base_url=OLLAMA_BASE):
        self.base_url = base_url

    def list_models(self):
        """List available models from Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def is_running(self):
        """Check if Ollama server is running."""
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except:
            return False

    def stream_query(self, model, prompt, images=None,
                     on_chunk=None, on_done=None, on_error=None):
        """
        Stream a query to Ollama model.

        Args:
            model: Model name to use
            prompt: The prompt text
            images: Optional list of base64 encoded images
            on_chunk: Callback for each response chunk
            on_done: Callback when streaming is complete
            on_error: Callback for errors
        """
        payload = {"model": model, "prompt": prompt, "stream": True}
        if images:
            payload["images"] = images
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload, stream=True, timeout=120
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if on_chunk and "response" in chunk:
                            on_chunk(chunk["response"])
                        if chunk.get("done"):
                            if on_done:
                                on_done()
                            return
        except Exception as e:
            if on_error:
                on_error(str(e))
