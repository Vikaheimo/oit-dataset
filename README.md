# OIT Dataset

This repository contains code for my research in OIT. It is a simple neural network, used to detect different weather patterns / events from a video source. The `data` folder contains ~1500 labelled images used to train this neural network.

## Prerequisites for running the code

Make sure that you have `uv` installed. That can be done from [here](https://docs.astral.sh/uv/).

## Running the program

All of the scripts can then be run using `uv run <script>`, e.g. `uv run main.py`.

| Script name     | Purpose                                 |
| --------------- | --------------------------------------- |
| main.py         | Trains the neural network.              |
| dataset_info.py | Checks and reports on the dataset size. |
| predict.py      | Predicts the weather of a single image. |
