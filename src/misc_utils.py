import functools
import os


@functools.cache
def pil_font() -> str:
    path_canidates = [
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",  # Arch
        "/usr/share/fonts/liberation/truetype/LiberationSans-Regular.ttf",  # Debian
    ]
    for path in path_canidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No font found in {path_canidates}")
