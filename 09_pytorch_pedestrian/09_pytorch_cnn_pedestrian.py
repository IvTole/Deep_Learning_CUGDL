import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # PyTorch
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    from helper_functions import accuracy_fn

    # TorchMetrics
    import torchmetrics, mlxtend
    from torchmetrics import ConfusionMatrix
    from mlxtend.plotting import plot_confusion_matrix

    # Torchvision 
    import torchvision
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    # Matplotlib
    import matplotlib.pyplot as plt

    # Pandas
    import pandas as pd

    # Random
    import random

    # Path
    from pathlib import Path

    # Timer
    from timeit import default_timer as timer

    # Para barra de progreso
    from tqdm.auto import tqdm

    from xml.etree import ElementTree

    # Versiones
    print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
    return Dataset, ElementTree, ToTensor, torch


@app.cell
def _(Dataset, ElementTree, Image, ToTensor, class_names_label, os):
    class PedestrianDataset(Dataset):
        def __init__(self, root_dir, transform=ToTensor()):
            self.root_dir = root_dir
            self.transform = transform

            self.annotations_dir = os.path.join(root_dir, 'Annotations')
            self.images_dir = os.path.join(root_dir, 'JPEGImages')

            self.annotation_files = sorted(os.listdir(self.annotations_dir))
            self.image_files = sorted(os.listdir(self.images_dir))

        def __len__(self):
            return len(self.annotation_files)

        def __getitem__(self, idx):
            annotation_file = self.annotation_files[idx]
            image_file = self.image_files[idx]

            annotation_path = os.path.join(self.annotations_dir, annotation_file)
            image_path = os.path.join(self.images_dir, image_file)

            # Load image
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            # Load label from XML annotation
            dom = ElementTree.parse(annotation_path)
            vb = dom.findall('object')
            label_text = vb[0].find('name').text
            label = class_names_label[label_text]  # Assumes you have defined this dictionary somewhere

            return image, label

    return


@app.cell
def _(torch):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
