"""Utility functions: image array/PIL/data-URI conversions and remote loading."""

import base64
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image

_REQUEST_TIMEOUT = 20
_USER_AGENT = "Zyra/1.0"


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype("uint8"), "RGB")


def pil_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def load_image_from_url(url: str) -> np.ndarray | None:
    """Download an image and return it as an RGB numpy array, or None on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as response:
            image = Image.open(BytesIO(response.read())).convert("RGB")
        return np.array(image)
    except Exception:
        return None
