"""image_captioner package.

Registers the HEIC/HEIF Pillow opener at import time so that
``PIL.Image.open()`` transparently supports ``.heic``/``.heif`` files
wherever it is used in this package (hashing, resizing, etc.).
"""
from __future__ import annotations

import pillow_heif

pillow_heif.register_heif_opener()
