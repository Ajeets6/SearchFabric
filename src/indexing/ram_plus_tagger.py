"""
RAM++ image tagging wrapper.

This module provides a lazy-loading adapter around the recognize-anything RAM++
model so the rest of the app can use tags without handling model setup details.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


class RAMPlusTagger:
    """Generate image tags with RAM++ when the runtime and checkpoint are available."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        image_size: int = 384,
        vit_variant: str = "swin_l",
    ):
        self.image_size = image_size
        self.vit_variant = vit_variant
        self._checkpoint_path = checkpoint_path or os.getenv("RAM_PLUS_CHECKPOINT")

        self._torch = None
        self._model = None
        self._transform = None
        self._inference_fn = None
        self._device = None

        self._init_attempted = False
        self.last_error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def checkpoint_path(self) -> Optional[str]:
        return self._checkpoint_path

    def configure_checkpoint(self, checkpoint_path: str) -> None:
        """Update checkpoint path and reset model state for re-initialization."""
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._transform = None
        self._inference_fn = None
        self._torch = None
        self._device = None
        self._init_attempted = False
        self.last_error = None

    def _initialize_if_needed(self) -> bool:
        if self._model is not None:
            return True
        if self._init_attempted and self.last_error:
            return False

        self._init_attempted = True

        try:
            import torch
            from ram import get_transform, inference_ram  # type: ignore[import-not-found]
            from ram.models import ram_plus  # type: ignore[import-not-found]
        except Exception as exc:
            self.last_error = (
                "RAM++ dependencies are missing. Install recognize-anything and required "
                f"runtime packages. Details: {exc}"
            )
            return False

        if not self._checkpoint_path:
            self.last_error = (
                "RAM++ checkpoint not configured. Set RAM_PLUS_CHECKPOINT env var "
                "or pass checkpoint path when constructing RAMPlusTagger."
            )
            return False

        checkpoint = Path(self._checkpoint_path)
        if not checkpoint.exists():
            self.last_error = f"RAM++ checkpoint not found: {checkpoint}"
            return False

        try:
            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._transform = get_transform(image_size=self.image_size)
            self._model = ram_plus(
                pretrained=str(checkpoint),
                image_size=self.image_size,
                vit=self.vit_variant,
            )
            self._model.eval()
            self._model = self._model.to(self._device)
            self._inference_fn = inference_ram
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"RAM++ model initialization failed: {exc}"
            self._model = None
            self._transform = None
            self._inference_fn = None
            self._torch = None
            self._device = None
            return False

    def generate_tags(self, image_path: Path) -> Tuple[List[str], Optional[str]]:
        """
        Generate normalized English tags for an image.

        Returns:
            (tags, error_message)
        """
        if not self._initialize_if_needed():
            return [], self.last_error

        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self._device)

            with self._torch.no_grad():
                result = self._inference_fn(tensor, self._model)

            tag_string = ""
            if isinstance(result, (list, tuple)) and result:
                tag_string = result[0][0] if isinstance(result[0], list) else result[0]
            elif isinstance(result, str):
                tag_string = result

            tags = [part.strip().lower() for part in tag_string.split("|") if part.strip()]
            deduped = []
            seen = set()
            for tag in tags:
                if tag not in seen:
                    seen.add(tag)
                    deduped.append(tag)

            return deduped, None
        except Exception as exc:
            return [], f"RAM++ inference failed for {image_path.name}: {exc}"
