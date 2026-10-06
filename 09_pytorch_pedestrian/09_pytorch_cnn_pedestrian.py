import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os

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
    import torchvision.transforms as transforms

    # Matplotlib
    import matplotlib.pyplot as plt

    # Pandas
    import pandas as pd

    # Numpy
    import numpy as np

    # Random
    import random

    # Path
    from pathlib import Path

    from PIL import Image

    # Timer
    from timeit import default_timer as timer

    # Para barra de progreso
    from tqdm.auto import tqdm

    from xml.etree import ElementTree

    # Versiones
    print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
    return (
        DataLoader,
        Dataset,
        ElementTree,
        Image,
        ToTensor,
        nn,
        np,
        os,
        plt,
        torch,
        transforms,
    )


@app.cell
def _(Dataset, ElementTree, Image, ToTensor, os):
    class PedestrianDataset(Dataset):
        def __init__(self, root_dir, class_names_label, transform=ToTensor()):
            self.root_dir = root_dir
            self.transform = transform
            self.class_names_label = class_names_label

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
            label = self.class_names_label[label_text]  # Assumes you have defined this dictionary somewhere

            return image, label

    return (PedestrianDataset,)


@app.cell
def _(torch):
    # Set device

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    return (device,)


@app.cell
def _(DataLoader, PedestrianDataset, transforms):
    # image - label dictionary
    class_names = ['person', 'person-like']
    class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

    data_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=(-5, 5), translate=(0, 0.1), scale=(1.0, 1.25), shear=(-10, 10)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_dir = 'data/'

    train_dir = dataset_dir + 'Train/Train'
    test_dir = dataset_dir + 'Test/Test'
    val_dir = dataset_dir + 'Val/Val'

    # Use the custom PedestrianDataset class
    train_data = PedestrianDataset(train_dir, class_names_label=class_names_label,transform=data_transform)
    test_data = PedestrianDataset(test_dir, class_names_label=class_names_label, transform=data_transform)
    val_data = PedestrianDataset(val_dir, class_names_label=class_names_label, transform=data_transform)

    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return class_names, val_loader


@app.cell
def _(class_names):
    classes = class_names
    print(classes)
    return (classes,)


@app.cell
def _(np, plt):
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    return (imshow,)


@app.cell
def _(classes, imshow, np, plt):
    def previewSomeImages(loader):
        dataiter = iter(loader)
        #dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images = images.numpy() # convert images to numpy for display


        print(images[0].shape)
        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 8))
        # display some images
        images_to_display = 10
        for idx in np.arange(images_to_display):
            ax = fig.add_subplot(2, int(images_to_display/2), idx+1, xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(classes[int(labels[idx])])
        plt.show()

    return (previewSomeImages,)


@app.cell
def _(previewSomeImages, val_loader):
    previewSomeImages(val_loader)
    return


@app.cell
def _(nn, torch):
    class ModelCNN(nn.Module):
        def __init__(self, num_classes: int = 1000, dropout: float = 0.4) -> None:
            super(ModelCNN, self).__init__()

            self.features = nn.Sequential(
               nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),   
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 256, kernel_size=3, padding=1), 
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(160000, 500),  
                nn.Dropout(dropout),
                nn.InstanceNorm1d(500),
                nn.ReLU(inplace=True),
                nn.Linear(500, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x


    return (ModelCNN,)


@app.cell
def _(ModelCNN, device, nn, torch):
    model = ModelCNN(num_classes=2)
    model = model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    return


app._unparsable_cell(
    r"""
    epochs = 20
    steps = 0
    print_every = 20
    running_loss = 0

    train_losses, validation_losses = [], []
    model.to(device)


    #for epoch in range(epochs):
    #    for inputs, labels, in train_loader:
            steps += 1

            # Move inputs and label to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero out the gradients of the optimizer
            optimizer.zero_grad()

            # Get the outputs of the model and compute loss 
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            # Compute the loss gradient using the backward method and have the optimizer take a step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        validation_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
          
                model.train()
            
                train_losses.append(running_loss/len(train_loader))
                validation_losses.append(validation_loss/len(val_loader))

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(val_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(val_loader)))
            
                running_loss = 0

    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
