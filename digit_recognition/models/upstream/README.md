---
language:
- en
license: mit
library_name: pytorch
tags:
- computer-vision
- image-classification
- mnist
- handwritten-digits
metrics:
- accuracy
pipeline_tag: image-classification
---

# MNIST Handwritten Digit Classifiers (CNN & MLP PyTorch Models)

This repository contains pre-trained **PyTorch** model weights for classifying handwritten digits (0 to 9) from 28x28 grayscale images:

- **CNN Model (`MNIST_CNNmodel_weights.pth`)**: **99.03%** Test Accuracy.
- **MLP Model (`MNIST_MLPmodel_weights.pth`)**: Baseline Multi-Layer Perceptron.
- **Application**: Interactive Tkinter digit drawing app (`handdrawnDigitClassification.py`).

---

## Model Architectures

### 1. Convolutional Neural Network (CNN) - 99.03% Accuracy
- **Block 1**: `Conv2d(1, 32, kernel_size=3)` -> `BatchNorm` -> `ReLU` -> `MaxPool2d(2)`
- **Block 2**: `Conv2d(32, 64, kernel_size=3)` -> `BatchNorm` -> `ReLU` -> `MaxPool2d(2)` -> `Dropout2d(0.25)`
- **Classifier**: `Flatten` -> `Linear(64*7*7, 128)` -> `ReLU` -> `Dropout(0.5)` -> `Linear(128, 10)`

### 2. Multi-Layer Perceptron (MLP)
- `Flatten` -> `Linear(784, 128)` -> `ReLU` -> `Linear(128, 64)` -> `ReLU` -> `Linear(64, 10)`

---

## Model Usage

```python
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Define CNN Model
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = MNISTCNN()
weights_path = hf_hub_download(repo_id="Vecrist/mnist-pytorch-digit-classifiers", filename="MNIST_CNNmodel_weights.pth")
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()
```
