import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from abc import ABC, abstractmethod


# Abstract Base Class declaring polymorphic interface
class Evaluator(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, img_path):
        """Unified method signature for all evaluators"""
        pass


# Concrete Implementation 1: Our Trained Neural Network
class AIEvaluator(Evaluator):
    def __init__(self, model_path):
        super().__init__("Neural Network Classifier")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reconstruct the exact architecture from train.py
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        # Load your trained weights safely
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def evaluate(self, img_path):
        if not os.path.exists(img_path):
            return "Error: Image not found."

        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            score = output.item()

        # Clamp rating strictly between 1.0 and 5.0 limits
        score = max(1.0, min(5.0, score))
        return f"Predicted Rating: {score:.2f} / 5.0"


# Concrete Implementation 2: Simple Pixel Temperature Logic
class ColorTemperatureEvaluator(Evaluator):
    def __init__(self):
        super().__init__("Color Thermal Processor")

    def evaluate(self, img_path):
        if not os.path.exists(img_path):
            return "Error: Image not found."

        # Shrink image to process colors instantly
        image = Image.open(img_path).convert("RGB").resize((32, 32))
        pixels = list(image.get_flattened_data())

        # Count pixels with more red than blue (warm colors)
        warm_pixels = sum(1 for r, g, b in pixels if r > b + 20)
        warm_ratio = warm_pixels / len(pixels)
        score = 1.0 + (warm_ratio * 4.0)

        return f"Color Heat Rating: {score:.2f} / 5.0"