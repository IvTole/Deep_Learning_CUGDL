import os 

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(
        train_data,
        test_data,
        val_data,
        batch_size
):
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, val_dataloader
    