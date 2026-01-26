import logging

from custom_types import Weights


DEFAULT_MODEL_PATH = "weather_resnet18.pth"
CLASSES = [
    "beautiful_sunrise",
    "beautiful_sunset",
    "boring_cloudy",
    "clear_sky",
    "fog",
    "good_cloudy",
    "storm",
]
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_LEVEL_STRING = "INFO"
DEFAULT_FRAME_SKIP = 30
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_SLIDING_WINDOW_SIZE = 10
DEFAULT_RATING_WEIGHTS = Weights(
    beautiful=0.35,
    boring=-0.25,
    fog=-0.15,
    storm=0.20,
    entropy=0.15,
    temporal_change=0.15,
    phases=0.10,
)

LOW_FRAME_SKIP_THRESHOLD = 10
HIGH_FRAME_SKIP_THRESHOLD = 100
