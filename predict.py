"""Utilities to predict weather from images or videos using a ResNet18.

This module provides simple CLI usage and helper functions for running
predictions on single images or videos (key-frame sampling).
"""

import argparse
import os
import sys
from collections import Counter
from typing import List, Tuple, Union
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


CLASSES = [
    "beautiful_sunrise",
    "beautiful_sunset",
    "boring_cloudy",
    "clear_sky",
    "fog",
    "good_cloudy",
    "storm",
]

MODEL_PATH = "weather_resnet18.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model(MODEL_PATH)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_image(image_path: str) -> Tuple[str, float]:
    """Predict weather from an image file.

    Returns a tuple of (predicted_label, confidence).
    """
    image = Image.open(image_path).convert("RGB")
    probabilities, predicted_index = predict_frame(image)
    confidence = probabilities[predicted_index].item()

    for index, probability in enumerate(probabilities):
        print(f"{CLASSES[index]}: {probability.item():.4f}")

    print(
        f"\nPredicted Weather: {CLASSES[predicted_index]}  (Confidence: {confidence:.2f})"
    )
    return CLASSES[predicted_index], confidence


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


def predict_video(
    video_path: str, frame_skip: int = 30
) -> List[Tuple[int, str, str, float]]:
    """Predict weather for key frames in a video and show timestamps.

    Returns a list of tuples: (frame_index, formatted_timestamp, label, confidence).
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}, FPS: {fps:.2f}\n")

    results: List[Tuple[int, str, str, float]] = []
    frame_index = 0

    while True:
        has_frame, frame = video_capture.read()
        if not has_frame:
            break

        if frame_index % frame_skip == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probabilities, predicted_index = predict_frame(image)
            label = CLASSES[predicted_index]
            confidence = probabilities[predicted_index].item()

            formatted_timestamp = format_timestamp(frame_index, fps)
            results.append((frame_index, formatted_timestamp, label, confidence))
            print(
                f"Frame {frame_index} ({formatted_timestamp}): {label} ({confidence:.2f})"
            )

        frame_index += 1

    video_capture.release()

    # Aggregate results
    labels = [label for _, _, label, _ in results]
    if not labels:
        print("No frames analyzed. Check if the video file is valid.")
        return results

    most_common = Counter(labels).most_common(1)[0]
    print("\n--- SUMMARY ---")
    print(f"Most frequent weather: {most_common[0]} ({most_common[1]} frames)")

    return results


def predict_video_or_image(
    input_path: str,
) -> Union[Tuple[str, float], List[Tuple[int, str, str, float]]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        return predict_image(input_path)
    if file_extension in [".mp4", ".avi", ".mov", ".mkv"]:
        return predict_video(input_path, frame_skip=30)

    raise ValueError("Unsupported file type. Please provide an image or video file.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict weather from an image or video file."
    )
    parser.add_argument("input_path", help="Path to image or video file")
    args = parser.parse_args()

    try:
        predict_video_or_image(args.input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - unexpected runtime error
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
