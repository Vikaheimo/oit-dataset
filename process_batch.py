import logging
import sys
import os
from pathlib import Path

from predict import ImageData, VideoData, predict_video_or_image
from utils import get_log_level_from_env
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def process_file(file_path: Path) -> pd.DataFrame | None:
    logger.info(f"Processing file {file_path}")

    data = predict_video_or_image(file_path)

    if isinstance(data, ImageData):
        logger.warning("Skipping image file!")
        return None

    return video_data_to_dataframe(data)


def main():
    if len(sys.argv) != 3:
        print("Usage: python process_batch.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    dfs: list[pd.DataFrame] = []

    files = list(Path(input_folder).iterdir())
    logger.info(f"Found {len(files)} files to process")

    for i, file_path in enumerate(files):
        logger.info(f"Processing file {i + 1}/{len(files)}")
        if file_path.is_file():
            try:
                df = process_file(file_path)
                if df is not None:
                    dfs.append(df)
            except Exception:
                logger.exception(f"Failed to process file {file_path}")

    logger.info(f"Processed {len(dfs)} video files")

    if not dfs:
        logger.warning("No video data processed")
        return

    final_df = pd.concat(dfs, ignore_index=True)

    output_path = Path(output_folder) / "results.parquet"
    final_df.to_parquet(output_path, index=False)

    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=get_log_level_from_env(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename="app.log",
    )
    main()
