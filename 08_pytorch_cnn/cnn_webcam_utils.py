"""Carga y visualización de la CNN del notebook 08, sin entrenar ni descargar datos."""
import torch
import base64
import io
import struct
from pathlib import Path

import numpy as np

from PIL import Image, ImageOps
from torch import nn


class MNISTModelV2(nn.Module):
    def __init__(self, hidden_units=10, output_shape=10):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, hidden_units, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden_units * 7 * 7, output_shape),
        )

    def forward(self, x):
        return self.classifier(self.block_2(self.block_1(x)))


def load_cnn(path):
    # CPU es suficiente para una imagen de 28×28 y simplifica la visualización.
    weights = torch.load(Path(path), map_location="cpu", weights_only=True)
    hidden = weights["block_1.0.weight"].shape[0]
    classes = weights["classifier.1.weight"].shape[0]
    model = MNISTModelV2(hidden, classes)
    model.load_state_dict(weights, strict=True)
    return model.eval()


def prepare_image(image, crop_percent=100, invert=False, contrast=False,
                  threshold=None, center=False):
    gray = image.convert("L")
    width, height = gray.size
    side = max(1, round(min(width, height) * crop_percent / 100))
    left, top = (width - side) // 2, (height - side) // 2
    cropped = gray.crop((left, top, left + side, top + side))
    if contrast:
        cropped = ImageOps.autocontrast(cropped)
    if invert:
        cropped = ImageOps.invert(cropped)
    if threshold is not None:
        cropped = cropped.point(lambda pixel: 255 if pixel >= threshold else 0)
    if center:
        # Se espera un dígito claro sobre fondo oscuro después de invertir.
        bbox = cropped.point(lambda pixel: 255 if pixel > 40 else 0).getbbox()
        result = Image.new("L", (28, 28), 0)
        if bbox:
            digit = cropped.crop(bbox)
            digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
            result.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    else:
        result = cropped.resize((28, 28), Image.Resampling.LANCZOS)
    array = np.asarray(result, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return cropped, result, tensor


def extract_activations(model, tensor):
    """Recorre las capas en orden: mismas operaciones que forward, sin hooks."""
    activations = {}
    with torch.inference_mode():
        x = tensor
        for block_name in ("block_1", "block_2"):
            for index, layer in enumerate(getattr(model, block_name)):
                x = layer(x)
                activations[f"{block_name}.{index}"] = x[0].clone()
        logits = model.classifier(x)
        probabilities = logits.softmax(dim=1)[0]
    return activations, probabilities


def mnist_sample(root, index):
    raw = Path(root) / "mnist_numbers_data" / "MNIST" / "raw"
    with (raw / "t10k-images-idx3-ubyte").open("rb") as images:
        magic, count, rows, cols = struct.unpack(">IIII", images.read(16))
        if magic != 2051 or not 0 <= index < count:
            raise ValueError("Archivo MNIST o índice inválido")
        images.seek(16 + index * rows * cols)
        array = np.frombuffer(images.read(rows * cols), dtype=np.uint8).reshape(rows, cols)
    with (raw / "t10k-labels-idx1-ubyte").open("rb") as labels:
        magic, count = struct.unpack(">II", labels.read(8))
        if magic != 2049 or not 0 <= index < count:
            raise ValueError("Archivo de etiquetas MNIST inválido")
        labels.seek(8 + index)
        label = labels.read(1)[0]
    return Image.fromarray(array), label


def image_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def activation_grid(activations, limit=1.0, columns=5, signed=False):
    from matplotlib import colormaps
    cmap = colormaps["coolwarm" if signed else "viridis"]
    tiles = []
    for index, channel in enumerate(activations):
        array = channel.numpy()
        normalized = (array / limit + 1) / 2 if signed else array / limit
        rgb = (cmap(np.clip(normalized, 0, 1))[..., :3] * 255).astype(np.uint8)
        picture = Image.fromarray(rgb).resize((140, 140), Image.Resampling.NEAREST)
        clipped = np.mean(np.abs(array) > limit) if signed else np.mean(array > limit)
        tiles.append(
            '<figure style="margin:0;text-align:center">'
            f'<img src="{image_uri(picture)}" style="width:100%;max-width:140px" />'
            f'<figcaption>Canal {index}<br><small>'
            f'mín {array.min():.2f} · máx {array.max():.2f}<br>'
            f'{clipped:.0%} fuera de escala</small></figcaption></figure>'
        )
    return (
        f'<div style="display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));'
        'gap:12px">' + ''.join(tiles) + '</div>'
    )
