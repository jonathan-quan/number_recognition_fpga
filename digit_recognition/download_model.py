"""Download a pinned, verified checkpoint. Never execute the upstream notebook."""

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "upstream"
CHECKPOINT = MODEL_DIR / "MNIST_MLPmodel_weights.pth"
HF_REVISION = "574952b057f9e9858efdcddd1404a8c105cb5d7b"
GITHUB_REVISION = "f9f3195f6b7b2a60f434f871f0f8a33b91fb3969"
NOTEBOOK_SHA256 = "ffc3ddb440ec01e5324375f5da4505e95bfc8872218c4045e628ea89445bf9cc"
FILES = {
    "MNIST_MLPmodel_weights.pth": "086f10b4a3e1a9dfb480bc73fdd6eaad76144156a1e72bc9034ade47e2865407",
    "README.md": "9f9b2879b7bc1d1e9447afbd156f658645c431c952ee99b1de5f667c7b346a72",
    "handdrawnDigitClassification.py": "f6a4ffd992d5cd0f91b36049853e73a937272c6243bed2c62506e780cb970629",
}


def verify_checkpoint(path=CHECKPOINT):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != FILES[CHECKPOINT.name]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")


def download():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/Vecrist/mnist-pytorch-digit-classifiers/resolve/{HF_REVISION}/"
    for name, expected in FILES.items():
        target = MODEL_DIR / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(f"Verified {target.name} ({target.stat().st_size:,} bytes)")
            continue
        with urllib.request.urlopen(base + name, timeout=60) as response:
            contents = response.read()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError(f"Downloaded {name} failed SHA-256 verification")
        target.write_bytes(contents)
        print(f"Downloaded {target.name} ({len(contents):,} bytes)")
    # Provenance only: read the class/transform definitions, never run training cells.
    notebook = MODEL_DIR / "mnistMLP.ipynb"
    if not notebook.exists() or hashlib.sha256(notebook.read_bytes()).hexdigest() != NOTEBOOK_SHA256:
        url = ("https://raw.githubusercontent.com/VecetyP/Hand-Drawn-Digit-Inference-Engine/"
               f"{GITHUB_REVISION}/mnistMLP.ipynb")
        with urllib.request.urlopen(url, timeout=60) as response:
            contents = response.read()
        if hashlib.sha256(contents).hexdigest() != NOTEBOOK_SHA256:
            raise ValueError("Upstream notebook SHA-256 mismatch")
        notebook.write_bytes(contents)


if __name__ == "__main__":
    download()

