import torch # no mover
from torch.utils.data import Dataset
import os

from PIL import Image

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from xml.etree import ElementTree

#from config import TRAIN_PATH, TEST_PATH

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

    