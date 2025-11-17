import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

classes = ['beautiful_sunrise', 'beautiful_sunset', 'clear_sky', 'cloudy', 'fog']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("weather_resnet18.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

image_path = "test.jpg"
img = Image.open(image_path).convert("RGB")
inp = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(inp)
    _, predicted_idx = torch.max(outputs, 1)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    for i, p in enumerate(probs):
        print(f"{classes[i]}: {p.item():.4f}")
    confidence = probs[predicted_idx.item()].item()

predicted_label = classes[predicted_idx.item()]
print(f"Predicted Weather: {predicted_label}  (Confidence: {confidence:.2f})")