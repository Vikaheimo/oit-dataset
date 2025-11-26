"""Utilities to predict weather from images or videos using a ResNet18.

This module provides simple CLI usage and helper functions for running
predictions on single images or videos (key-frame sampling).
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import os
import sys
from collections import Counter
from typing import Tuple, Union
import cv2
from PIL import Image
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torchvision import models, transforms

import constants


@dataclass
class FrameData:
    """
    Stores metadata and prediction probabilities for a single video frame.

    Attributes:
        frame_index (int): The index of the frame in the video.
        formatted_timestamp (str): Human-readable timestamp for the frame.
        probabilities (list[float]): List of predicted probabilities for each weather class,
            ordered according to the CLASSES list.
    """

    frame_index: int
    formatted_timestamp: str
    probabilities: list[float]


@dataclass
class VideoData:
    """
    Stores metadata and prediction probabilities for a video.

    Attributes:
        video_path (str): The path to the video file.
        frame_count (int): The total number of frames in the video.
        fps (float): The frames per second of the video.
        sample_rate (int): The frame sampling rate used.
        predictions (list[FrameData]): A list of FrameData objects containing
            prediction results for sampled frames.
    """

    video_path: str
    frame_count: int
    fps: float
    sample_rate: int
    predictions: list[FrameData]


@dataclass
class ImageData:
    """
    Stores metadata and prediction probabilities for a single image.

    Attributes:
        image_path (str): The path to the image file.
        probabilities (list[float]): List of predicted probabilities for each weather class,
            ordered according to the CLASSES list.
    """

    image_path: str
    probabilities: list[float]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()


def get_log_level_from_env() -> int:
    """Get the logging level from the LOG_LEVEL environment variable."""
    log_level_str = os.getenv("LOG_LEVEL", constants.DEFAULT_LOG_LEVEL_STRING).upper()

    # `logging.getLevelName` doesn't convert strings properly,
    # so we use `getattr` to map manually to level constants.
    level = getattr(logging, log_level_str, None)

    if isinstance(level, int):
        return level

    return constants.DEFAULT_LOG_LEVEL


logging.basicConfig(
    level=get_log_level_from_env(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> torch.nn.Module:
    logger.debug(f"Loading model from path '{model_path}'")
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


def predict_image(image_path: str) -> ImageData:
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


def predict_video(video_path: str, frame_skip: int = 30) -> VideoData:
    """Predict weather for key frames in a video and show timestamps.

    Returns a VideoData object containing frame prediction results.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    logger.debug(f"Processing video: {video_path}")
    logger.debug(f"Total frames: {frame_count}, FPS: {fps:.2f}")

    video_result = VideoData(
        video_path=video_path,
        frame_count=frame_count,
        fps=fps,
        sample_rate=frame_skip,
        predictions=[],
    )
    frame_index = 0

    while True:
        has_frame, frame = video_capture.read()
        if not has_frame:
            break

        if frame_index % frame_skip == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probabilities, predicted_index = predict_frame(image)
            label = constants.CLASSES[predicted_index]
            confidence = probabilities[predicted_index].item()

            formatted_timestamp = format_timestamp(frame_index, fps)
            video_result.predictions.append(
                FrameData(
                    frame_index=frame_index,
                    formatted_timestamp=formatted_timestamp,
                    probabilities=[probability.item() for probability in probabilities],
                )
            )

            logger.debug(
                f"Frame {frame_index} ({formatted_timestamp}): {label} ({confidence:.2f})"
            )

        frame_index += 1

    video_capture.release()

    if not video_result.predictions:
        raise ValueError("No frames analyzed. Check if the video file is valid.")

    labels = []

    for frame in video_result.predictions:
        max_index = frame.probabilities.index(max(frame.probabilities))
        labels.append(constants.CLASSES[max_index])

    most_common = Counter(labels).most_common(1)[0]
    logger.info("--- SUMMARY ---")
    logger.info(f"Most frequent weather: {most_common[0]} ({most_common[1]} frames)")

    return video_result


def generate_output_filename(
    original_path: str, prefix: str = "prediction_plot"
) -> str:
    """
    Generate a filename for the plot that includes a timestamp and original base name.
    """
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{base_name}_{timestamp}.png"


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

    out_filename = generate_output_filename(image.image_path)
    plt.savefig(out_filename, dpi=200)
    plt.close()

    logger.info(f"Saved plot as {out_filename}")


def visualize_video_data(data: VideoData):
    matplotlib.use("Agg")
    if not data.predictions:
        logger.info("No data to visualize.")
        return

    frame_indices: list[int] = []
    probabilities_list = []
    for frame in data.predictions:
        frame_indices.append(frame.frame_index)

        if len(frame.probabilities) != len(constants.CLASSES):
            raise ValueError("Incorrect amount of probabilities in array!")
        probabilities_list.append(frame.probabilities)

    plt.figure(figsize=(10, 6))
    for i, cls in enumerate(constants.CLASSES):
        plt.plot(
            frame_indices, np.array(probabilities_list)[:, i], label=cls, marker="o"
        )

    plt.title(
        f"Prediction Probabilities Over Frames in Video (sampled every {data.sample_rate} frames)"
    )
    plt.xlabel("Frame Index")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Ensure the x-axis starts at frame zero with no left margin
    max_x = frame_indices[-1] if frame_indices else 0
    plt.xlim(0, max_x)
    plt.margins(x=0)

    out_filename = generate_output_filename(data.video_path)
    plt.savefig(out_filename)
    plt.close()

    logger.info(f"Saved plot as {out_filename}")


def predict_video_or_image(
    input_path: str,
) -> Union[ImageData, VideoData]:
    """
    Predict labels for a single image or a video file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported, or if no frames are analyzed in a video.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        return predict_image(input_path)
    if file_extension in [".mp4", ".avi", ".mov", ".mkv"]:
        return predict_video(input_path, frame_skip=30)

    raise ValueError("Unsupported file type. Please provide an image or video file.")


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
    else:
        visualize_single_image(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict weather from an image or video file."
    )
    parser.add_argument("input_path", help="Path to image or video file")
    args = parser.parse_args()

    try:
        data = predict_video_or_image(args.input_path)
        data_visualization(data)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - unexpected runtime error
        logger.critical(f"Unexpected error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
