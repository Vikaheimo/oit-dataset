import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np

import constants
from utils import get_log_level_from_env
from custom_types import VideoData, Weights


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFeatures:
    """
    Video-level aesthetic and temporal features derived from weather predictions.
    """

    # Aesthetic composition
    beautiful_ratio: float
    boring_ratio: float

    # Dynamics
    entropy: float
    temporal_change: float
    phase_count: float

    # Events
    storm_peak: float
    fog_ratio: float


def compute_video_features(video: VideoData) -> VideoFeatures:
    probs = video.probabilities
    eps = 1e-8

    class_to_idx = {c: i for i, c in enumerate(video.class_names)}

    mean_probs = probs.mean(axis=0)
    beautiful_classes = [
        "beautiful_sunrise",
        "beautiful_sunset",
        "good_cloudy",
    ]
    boring_classes = ["clear_sky", "boring_cloudy"]

    beautiful_ratio = sum(
        mean_probs[class_to_idx[c]] for c in beautiful_classes if c in class_to_idx
    )

    boring_ratio = sum(
        mean_probs[class_to_idx[c]] for c in boring_classes if c in class_to_idx
    )

    fog_ratio = mean_probs[class_to_idx["fog"]] if "fog" in class_to_idx else 0.0

    storm_peak = (
        probs[:, class_to_idx["storm"]].max() if "storm" in class_to_idx else 0.0
    )

    entropy = -np.sum(mean_probs * np.log(mean_probs + eps))

    diffs = np.abs(np.diff(probs, axis=0))
    temporal_change = diffs.mean() if len(diffs) > 0 else 0.0

    labels = np.argmax(probs, axis=1)
    phase_changes = np.sum(labels[1:] != labels[:-1])
    phase_count = float(phase_changes + 1)

    return VideoFeatures(
        beautiful_ratio=float(beautiful_ratio),
        boring_ratio=float(boring_ratio),
        entropy=float(entropy),
        temporal_change=float(temporal_change),
        phase_count=phase_count,
        storm_peak=float(storm_peak),
        fog_ratio=float(fog_ratio),
    )


def rate_video_from_features(features: VideoFeatures, weights: Weights) -> float:
    score = (
        weights.beautiful * features.beautiful_ratio
        + weights.boring * features.boring_ratio
        + weights.fog * features.fog_ratio
        + weights.storm * features.storm_peak
        + weights.entropy * features.entropy
        + weights.temporal_change * features.temporal_change
        + weights.phases * features.phase_count
    )

    score = np.clip(score * 100.0, 0.0, 100.0)
    return float(score)


def main():
    parser = argparse.ArgumentParser(description="Give video a rating")
    parser.add_argument("input_path", type=Path, help="Path to image or video file")
    args = parser.parse_args()

    video_prediction = predict_video(args.input_path)
    video_features = compute_video_features(video_prediction)
    rating = rate_video_from_features(video_features, constants.DEFAULT_RATING_WEIGHTS)

    logger.info(f"Got rating {rating}, for video {args.input_path}")


if __name__ == "__main__":
    from predict import predict_video

    logging.basicConfig(
        level=get_log_level_from_env(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    main()
