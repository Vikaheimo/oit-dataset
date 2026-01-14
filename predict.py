"""Utilities to predict weather from images or videos using a ResNet18.

This module provides simple CLI usage and helper functions for running
predictions on single images or videos (key-frame sampling).
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import os
import sys
from collections import Counter
from typing import Tuple, Union, Optional
import cv2
from PIL import Image
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torchvision import models, transforms

import constants
from utils import get_log_level_from_env, is_image_file, is_video_file


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


DEVICE = os.getenv("DEVICE", "cpu")

load_dotenv()

if __name__ == "__main__":
    logging.basicConfig(
        level=get_log_level_from_env(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

logger = logging.getLogger(__name__)


def get_env_int(name: str, default: int, check_function=lambda x: True) -> int:
    """Get an environment variable as an integer, with a default. Additional check function can be provided."""
    value = os.getenv(name)

    if value is None:
        logger.info(f"Environment variable {name} not set. Defaulting to {default}.")
        return default
    try:
        int_value = int(value)
        if check_function(int_value):
            return int_value
        else:
            logger.warning(
                f"Environment variable {name} has value '{value}' which does not pass the check function. Defaulting to {default}."
            )
            return default
    except ValueError:
        logger.warning(
            f"Environment variable {name} has non-integer value '{value}'. Defaulting to {default}."
        )
        return default


def load_model(model_path: str) -> torch.nn.Module:
    logger.info(f"Loading model from path '{model_path}'")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(constants.CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model_path = os.getenv("MODEL_PATH", constants.DEFAULT_MODEL_PATH)
MODEL = load_model(model_path)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

logger.info(f"Model loaded successfully. Using device: {DEVICE}")


def is_valid_frame_skip(value: int) -> bool:
    """Check function to ensure frame skip is a positive integer. Logs warnings for unusual values."""
    if value <= 0:
        logger.warning("FRAME_SKIP must be a positive integer.")
        return False

    if value > constants.HIGH_FRAME_SKIP_THRESHOLD:
        logger.warning(
            f"FRAME_SKIP is unusually high (>{constants.HIGH_FRAME_SKIP_THRESHOLD}); this may lead to very few frames being analyzed."
        )

    if value < constants.LOW_FRAME_SKIP_THRESHOLD:
        logger.warning(
            f"FRAME_SKIP is quite low (<{constants.LOW_FRAME_SKIP_THRESHOLD}); this may lead to high processing times."
        )
    return True


def predict_image(image_path: Path) -> ImageData:
    """Predict weather from an image file.

    Returns an ImageData object with data from the prediction.
    """
    logger.debug(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    probabilities, predicted_index = predict_frame(image)
    confidence = probabilities[predicted_index].item()

    logger.info(
        f"Predicted Weather: {constants.CLASSES[predicted_index]}  (Confidence: {confidence:.2f})"
    )

    return ImageData(image_path, [probability.item() for probability in probabilities])


def predict_frame(image: Image.Image) -> Tuple[torch.Tensor, int]:
    """Run weather prediction on a single video frame or image.

    Returns (probabilities_tensor, predicted_index).
    """
    raw_input = transform(image)
    if not isinstance(raw_input, torch.Tensor):
        # ensure we have a tensor (transform may not include ToTensor)
        raw_input = transforms.ToTensor()(raw_input)

    model_input = raw_input.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(model_input)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())

    return probabilities, predicted_index


def format_timestamp(frame_index: int, fps: float) -> str:
    """Calculate a formatted timestamp for a frame index.

    Returns a string in MM:SS format.
    """
    timestamp_seconds = frame_index / fps if fps else 0
    minutes = int(timestamp_seconds // 60)
    seconds = int(timestamp_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def predict_video(video_path: Path, frame_skip: int) -> VideoData:
    """Predict weather for key frames in a video and show timestamps.

    Returns a VideoData object containing frame prediction results.
    """
    logger.info(f"Processing video: {video_path} with frame skip: {frame_skip}")

    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    logger.info(f"Total frames: {frame_count}, FPS: {fps:.2f}")

    predictions = []
    timestamps = []

    frame_index = 0

    while True:
        has_frame, frame = video_capture.read()
        if not has_frame:
            break

        if frame_index % frame_skip == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probabilities, predicted_index = predict_frame(image)

            formatted_timestamp = format_timestamp(frame_index, fps)
            predictions.append([probability.item() for probability in probabilities])
            timestamps.append((frame_index, formatted_timestamp))

            logger.debug("Processed frame %d at %s", frame_index, formatted_timestamp)

        frame_index += 1

    video_capture.release()

    video_result = VideoData(
        video_path=video_path,
        frame_count=frame_count,
        fps=fps,
        sample_rate=frame_skip,
        class_names=constants.CLASSES,
        timestamps=np.array(
            timestamps, dtype=[("frame_index", int), ("timestamp", "U6")]
        ),
        probabilities=np.array(predictions),
    )

    logger.info(f"Finished processing video: {video_path}")

    if not video_result.probabilities.size:
        raise ValueError("No frames analyzed. Check if the video file is valid.")

    labels = []
    for probs in video_result.probabilities:
        predicted_index = int(np.argmax(probs))
        labels.append(constants.CLASSES[predicted_index])

    most_common = Counter(labels).most_common(1)[0]
    logger.info("--- SUMMARY ---")
    logger.info(f"Total frames analyzed: {len(video_result.probabilities)}")
    logger.info(f"Most frequent weather: {most_common[0]} ({most_common[1]} frames)")
    logger.info("----------------")

    return video_result


def predict_video_or_image(
    input_path: Path,
) -> Union[ImageData, VideoData]:
    """
    Predict labels for a single image or a video file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported, or if no frames are analyzed in a video.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if is_image_file(input_path):
        return predict_image(input_path)
    if is_video_file(input_path):
        frame_skip = get_env_int(
            "FRAME_SKIP", constants.DEFAULT_FRAME_SKIP, is_valid_frame_skip
        )

        return predict_video(input_path, frame_skip)

    raise ValueError("Unsupported file type. Please provide an image or video file.")


def generate_output_path(
    data: Union[ImageData, VideoData], prefix: str = "prediction_plot"
) -> str:
    """Generate an output file path for saving plots."""

    if isinstance(data, ImageData):
        base_name = os.path.splitext(os.path.basename(data.image_path))[0]
    elif isinstance(data, VideoData):
        base_name = data.name or os.path.splitext(os.path.basename(data.video_path))[0]
    else:
        raise ValueError("Unsupported data type for generating output path.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{base_name}_{timestamp}.png"

    output_dir = os.getenv("OUTPUT_DIR", constants.DEFAULT_OUTPUT_DIR)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create output directory '{output_dir}': {exc}"
        ) from exc
    return os.path.join(output_dir, filename)


def visualize_single_image(image: ImageData, width: int = 10, height: int = 6):
    matplotlib.use("Agg")
    plt.figure(figsize=(width, height))
    plt.bar(
        constants.CLASSES,
        image.probabilities,
        color="cornflowerblue",
    )
    plt.title("Prediction Probabilities for Image")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    output_path = generate_output_path(image)
    plt.savefig(output_path, dpi=200)
    plt.close()

    logger.info(f"Saved plot as {output_path}")


def visualize_video_data(data: VideoData):
    matplotlib.use("Agg")
    if not isinstance(data.probabilities, np.ndarray) or data.probabilities.size == 0:
        logger.info("No data to visualize.")
        return

    frame_indices = data.timestamps["frame_index"]
    probs = data.probabilities
    if probs.ndim != 2 or probs.shape[1] != len(constants.CLASSES):
        raise ValueError("Incorrect amount of probabilities in array!")

    plt.figure(figsize=(10, 6))
    for i, cls in enumerate(constants.CLASSES):
        plt.plot(frame_indices, probs[:, i], label=cls, marker="o")

    display_name = data.name or os.path.basename(data.video_path)
    plt.title(
        f"Prediction Probabilities Over Frames ({display_name}) "
        f"(sampled every {data.sample_rate} frames)"
    )

    plt.xlabel("Frame Index")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()

    max_x = frame_indices[-1] if len(frame_indices) > 0 else 0
    plt.xlim(0, max_x)
    plt.margins(x=0)

    output_path = generate_output_path(data)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Saved plot as {output_path}")


def data_visualization(data: Union[ImageData, VideoData]) -> None:
    """Visualize prediction data for a single frame or multiple frames."""
    visualization_enabled = os.getenv("VISUALIZE", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    if not visualization_enabled:
        logger.info("Visualization is disabled. Skipping visualization step.")
        return

    logger.info("Generating visualization for prediction data.")

    if isinstance(data, VideoData):
        visualize_video_data(data)
        window_size = get_env_int(
            "SLIDING_WINDOW_SIZE",
            constants.DEFAULT_SLIDING_WINDOW_SIZE,
            lambda x: x > 0,
        )
        visualize_video_data(create_sliding_windows_with_time(data, window_size))
    else:
        visualize_single_image(data)


def create_sliding_windows_with_time(
    data: VideoData,
    window_size: int,
    step_size: int = 1,
) -> VideoData:
    """
    Runs a sliding window on a VideoData object and creates a new VideoData object from
    the sliding window data.

    Each window aggregates probabilities (e.g., by averaging) and assigns the center
    timestamp to represent the window.

    Args:
        data (VideoData): Input video data.
        window_size (int): Number of frames per sliding window.
        step_size (int): Number of frames to step the window by.

    Returns:
        VideoData: New object with smoothed (windowed) probabilities and reduced timestamps.
    """

    num_frames = len(data.timestamps)
    num_classes = data.probabilities.shape[1]

    num_windows = (num_frames - window_size) // step_size + 1

    windowed_timestamps = []
    windowed_probabilities = np.zeros((num_windows, num_classes))

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size

        window_probs = data.probabilities[start:end]
        # window_times = data.timestamps[start:end]

        windowed_probabilities[i] = np.mean(window_probs, axis=0)

        center_idx = start + window_size // 2
        windowed_timestamps.append(data.timestamps[center_idx])

    windowed_timestamps = np.array(windowed_timestamps)

    return VideoData(
        video_path=data.video_path,
        name=f"sliding_window_{window_size} {data.name or os.path.basename(data.video_path)}",
        frame_count=len(windowed_timestamps),
        fps=data.fps,
        sample_rate=data.sample_rate,
        timestamps=windowed_timestamps,
        probabilities=windowed_probabilities,
        class_names=data.class_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict weather from an image or video file."
    )
    parser.add_argument("input_path", type=Path, help="Path to image or video file")
    args = parser.parse_args()

    try:
        data = predict_video_or_image(args.input_path)
        data_visualization(data)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - unexpected runtime error
        logger.critical(f"Unexpected error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
