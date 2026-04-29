import json
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor

from model import SwinClassifier

app = FastAPI(title="Skin Cancer ViT Model API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_training_weights.pth"
CLASSES_PATH = "classes.json"

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

if isinstance(classes, dict):
    class_names = list(classes.values())
else:
    class_names = classes

processor = AutoImageProcessor.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
)

model = SwinClassifier(num_classes=len(class_names))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# if saved as checkpoint
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()


@app.get("/")
def home():
    return {
        "message": "Skin Cancer ViT FastAPI Backend Running",
        "classes": class_names
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": round(probabilities[predicted_index].item() * 100, 2),
        "all_probabilities": {
            class_names[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(class_names))
        }
    }