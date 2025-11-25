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


def predict_image(img_path: str) -> Tuple[str, float]:
    """Predict weather from an image file.

    Returns a tuple of (predicted_label, confidence).
    """
    img = Image.open(img_path).convert("RGB")
    probs, pred_idx = predict_frame(img)
    confidence = probs[pred_idx].item()

    for i, p in enumerate(probs):
        print(f"{CLASSES[i]}: {p.item():.4f}")

    print(f"\nPredicted Weather: {CLASSES[pred_idx]}  (Confidence: {confidence:.2f})")
    return CLASSES[pred_idx], confidence


def predict_frame(img: Image.Image) -> Tuple[torch.Tensor, int]:
    """Run weather prediction on a single video frame or image.

    Returns (probabilities_tensor, predicted_index).
    """
    raw = transform(img)
    if not isinstance(raw, torch.Tensor):
        # ensure we have a tensor (transform may not include ToTensor)
        raw = transforms.ToTensor()(raw)

    inp = raw.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(inp)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    return probs, pred_idx


def format_timestamp(frame_idx: int, fps: float) -> str:
    """Calculate a formatted timestamp for a frame index.

    Returns a string in MM:SS format.
    """
    timestamp_sec = frame_idx / fps if fps else 0
    minutes = int(timestamp_sec // 60)
    seconds = int(timestamp_sec % 60)
    return f"{minutes:02d}:{seconds:02d}"


def predict_video(
    video_path: str, frame_skip: int = 30
) -> List[Tuple[int, str, str, float]]:
    """Predict weather for key frames in a video and show timestamps.

    Returns a list of tuples: (frame_idx, time_str, label, confidence).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}, FPS: {fps:.2f}\n")

    results: List[Tuple[int, str, str, float]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probabilities, predicted_index = predict_frame(img)
            label = CLASSES[predicted_index]
            confidence = probabilities[predicted_index].item()

            time_str = format_timestamp(frame_idx, fps)
            results.append((frame_idx, time_str, label, confidence))
            print(f"Frame {frame_idx} ({time_str}): {label} ({confidence:.2f})")

        frame_idx += 1

    cap.release()

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

    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        return predict_image(input_path)
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
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
