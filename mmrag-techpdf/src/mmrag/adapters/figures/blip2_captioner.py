from pathlib import Path

from PIL import Image

from ...ports.captioner import Captioner


class BLIPCaptioner(Captioner):
    """
    Image captioner backed by a HuggingFace image-to-text model.

    Default model: ``Salesforce/blip-image-captioning-base``
    For higher quality use ``Salesforce/blip-image-captioning-large``
    or ``Salesforce/blip2-opt-2.7b`` (requires GPU + ~8 GB VRAM).

    The HuggingFace ``transformers`` pipeline is loaded lazily on first call
    so that importing this module does not pull GPU memory up-front.
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.model_name = model_name
        self._pipe = None  # lazy

    def _get_pipe(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline(
                "image-to-text",
                model=self.model_name,
            )
        return self._pipe

    def caption(self, image_path: str) -> str:
        """Generate a textual caption for the image at *image_path*."""
        path = Path(image_path)
        if not path.exists():
            return ""

        image = Image.open(path).convert("RGB")
        pipe = self._get_pipe()
        result = pipe(image, max_new_tokens=128)

        # pipeline returns a list of dicts, e.g. [{"generated_text": "..."}]
        if result and isinstance(result, list):
            text = result[0].get("generated_text", "")
        else:
            text = str(result)

        return text.strip()
