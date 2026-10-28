"""Read original MNIST IDX files, without torchvision or running training code."""

import gzip
import hashlib
import struct
import urllib.request

import numpy as np

from .download_model import ROOT

DATA = ROOT / "data"
# Original MNIST archive MD5s, also used by torchvision's MNIST downloader.
ARCHIVES = {
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
}


def archive(name):
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / name
    if not target.exists():
        print(f"Downloading MNIST {name}")
        with urllib.request.urlopen("https://storage.googleapis.com/cvdf-datasets/mnist/" + name, timeout=60) as response:
            raw = response.read()
        if hashlib.md5(raw).hexdigest() != ARCHIVES[name]:
            raise ValueError(f"MNIST archive checksum mismatch: {name}")
        target.write_bytes(raw)
    raw = target.read_bytes()
    if hashlib.md5(raw).hexdigest() != ARCHIVES[name]:
        raise ValueError(f"MNIST archive checksum mismatch: {name}")
    return gzip.decompress(raw)


def images(split="t10k", limit=None):
    if split not in ("t10k", "train"):
        raise ValueError("split must be t10k or train")
    raw = archive(f"{split}-images-idx3-ubyte.gz")
    magic, count, rows, cols = struct.unpack_from(">IIII", raw)
    if magic != 2051 or (rows, cols) != (28, 28) or len(raw) != 16 + count * 784:
        raise ValueError("Invalid MNIST image header/length")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 28, 28)[:limit].copy()


def test_set(limit=None):
    pixels = images(limit=limit)
    raw = archive("t10k-labels-idx1-ubyte.gz")
    magic, count = struct.unpack_from(">II", raw)
    if magic != 2049 or count != 10000 or len(raw) != 8 + count:
        raise ValueError("Invalid MNIST label header/length")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)[:limit].copy()
    return pixels, labels

