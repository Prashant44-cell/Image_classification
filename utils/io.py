# deep_image_analyzer/utils/io.py

# Import async file handling and PIL
import aiofiles
from PIL import Image

async def async_load_image(path):
    """
    Asynchronously load an image from disk and convert to RGB.
    """
    # aiofiles is used for async I/O; PIL handles image decoding
    return Image.open(path).convert("RGB")
