import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import Counter

classes = [
    "beautiful_sunrise",
    "beautiful_sunset",
    "boring_cloudy",
    "clear_sky",
    "fog",
    "good_cloudy",
    "storm",
]

model_path = "weather_resnet18.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model(model_path)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def predict_image(img_path: str):
    """Predict weather from an image file."""
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(inp)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    for i, p in enumerate(probs):
        print(f"{classes[i]}: {p.item():.4f}")

    print(f"\nPredicted Weather: {classes[pred_idx]}  (Confidence: {confidence:.2f})")
    return classes[pred_idx], confidence


def predict_frame(frame):
    """Run weather prediction on a single video frame."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inp)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        return classes[pred_idx], probs[pred_idx].item()


def predict_video(video_path, frame_skip=30):
    """Predict weather for key frames in a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}, FPS: {fps:.2f}\n")

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            label, conf = predict_frame(frame)
            results.append((frame_idx, label, conf))
            print(f"Frame {frame_idx}: {label} ({conf:.2f})")

        frame_idx += 1

    cap.release()

    # Aggregate results
    labels = [label for _, label, _ in results]
    if not labels:
        print("No frames analyzed. Check if the video file is valid.")
        return results

    most_common = Counter(labels).most_common(1)[0]
    print("\n--- SUMMARY ---")
    print(f"Most frequent weather: {most_common[0]} ({most_common[1]} frames)")

    return results


def predict_video_or_image(input_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        return predict_image(input_path)
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        return predict_video(input_path, frame_skip=30)
    else:
        raise ValueError(
            "Unsupported file type. Please provide an image or video file."
        )


def main():
    # input_path = "test.jpg"
    input_path = "test.mp4"
    predict_video_or_image(input_path)


if __name__ == "__main__":
    main()
