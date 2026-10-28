"""Architecture verified against the author's mnistMLP.ipynb and checkpoint."""

import torch
from torch import nn

from .download_model import CHECKPOINT, verify_checkpoint


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep the author's names so load_state_dict(strict=True) verifies every tensor.
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, pixels):
        return self.layers(pixels)


def load_model(path=CHECKPOINT):
    verify_checkpoint(path)
    # Restricted tensor loading; no upstream Python or notebook execution.
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model = MNISTMLP()
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    model = load_model()
    print(model)
    for name, value in model.state_dict().items():
        print(f"{name}: {tuple(value.shape)}, {value.dtype}, {value.numel():,} values")
    count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {count:,}; raw FP32 bytes: {count * 4:,}")
    print(f"Checkpoint file bytes: {CHECKPOINT.stat().st_size:,}")

