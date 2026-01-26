from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class VideoData:
    """
    Stores metadata and prediction probabilities for a video.

    Attributes:
        video_path (Path): The path to the video file.
        frame_count (int): The total number of frames in the video.
        fps (float): The frames per second of the video.
        sample_rate (int): The frame sampling rate used.
        timestamps (np.ndarray): Array of timestamps for each analyzed frame (shape (n,)).
        probabilities (np.ndarray): 2D array of predicted probabilities for each analyzed frame
            (shape (n, num_classes)).
        class_names (list[str]): List of weather class names corresponding to the probabilities.
        name (Optional[str]): An optional display name for the video, used for visualization.
    """

    video_path: Path
    frame_count: int
    fps: float
    sample_rate: int
    timestamps: np.ndarray
    probabilities: np.ndarray
    class_names: list[str]
    name: Optional[str] = None


@dataclass
class ImageData:
    """
    Stores metadata and prediction probabilities for a single image.

    Attributes:
        image_path (Path): The path to the image file.
        probabilities (list[float]): List of predicted probabilities for each weather class,
            ordered according to the CLASSES list.
    """

    image_path: Path
    probabilities: list[float]
