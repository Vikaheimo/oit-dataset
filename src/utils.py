import logging
import os
from pathlib import Path
import constants


def get_log_level_from_env() -> int:
    """Get the logging level from the LOG_LEVEL environment variable."""
    log_level_str = os.getenv("LOG_LEVEL", constants.DEFAULT_LOG_LEVEL_STRING).upper()

    # `logging.getLevelName` doesn't convert strings properly,
    # so we use `getattr` to map manually to level constants.
    level = getattr(logging, log_level_str, None)

    if isinstance(level, int):
        return level

    return constants.DEFAULT_LOG_LEVEL


def is_image_file(path_to_file: Path) -> bool:
    return path_to_file.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
    }


def is_video_file(path_to_file: Path) -> bool:
    return path_to_file.suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".flv",
        ".wmv",
    }
