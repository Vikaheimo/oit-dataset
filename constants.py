import logging


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

LOW_FRAME_SKIP_THRESHOLD = 10
HIGH_FRAME_SKIP_THRESHOLD = 100
