import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from custom_types import VideoData
import constants


def video_data_to_dataframe(video_data: VideoData) -> pd.DataFrame:
    # Extract structured timestamp fields
    frame_indices = video_data.timestamps["frame_index"]
    timestamps = video_data.timestamps["timestamp"]

    df = pd.DataFrame(
        {
            "frame_idx": frame_indices,
            "timestamp": timestamps,
        }
    )

    # Add probability columns (one per class)
    prob_df = pd.DataFrame(
        np.asarray(video_data.probabilities),
        columns=video_data.class_names,
    )

    df = pd.concat([df, prob_df], axis=1)

    # Metadata columns
    df["video_path"] = str(video_data.video_path)
    df["fps"] = video_data.fps
    df["sample_rate"] = video_data.sample_rate

    if video_data.name:
        df["video_name"] = video_data.name

    return df


def smoothen_np_array(array: np.ndarray, window_size: int = 3) -> np.ndarray:
    kernel = np.ones(window_size)

    numerator = np.apply_along_axis(
        lambda r: np.convolve(r, kernel, mode="same"), axis=1, arr=array
    )

    denominator = np.apply_along_axis(
        lambda r: np.convolve(np.ones_like(r), kernel, mode="same"), axis=1, arr=array
    )

    return numerator / denominator


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
