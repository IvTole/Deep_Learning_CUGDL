import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from get_data import PedestrianDataset
from data_setup import create_dataloaders
from model_builder import ModelCNN_V0
from engine import train
from config import HIDDEN_UNITS, LEARNING_RATE, EPOCHS, BATCH_SIZE
from config import TRAIN_PATH, TEST_PATH, VAL_PATH

from torchsummary import summary

# Script principal de entrenamiento
def main():
        
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

    train_data = PedestrianDataset(TRAIN_PATH, class_names_label=class_names_label,transform=data_transform)
    test_data = PedestrianDataset(TEST_PATH, class_names_label=class_names_label, transform=data_transform)
    val_data = PedestrianDataset(VAL_PATH, class_names_label=class_names_label, transform=data_transform)

#    train_data = PedestrianDataset(TRAIN_PATH, class_names_label=class_names_label)
#    test_data = PedestrianDataset(TEST_PATH, class_names_label=class_names_label)
#    val_data = PedestrianDataset(VAL_PATH, class_names_label=class_names_label)

    train_loader, test_loader, val_loader = create_dataloaders(
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        batch_size=BATCH_SIZE
    )

    #for i, (images, labels) in enumerate(train_loader):
    #    print(images.shape)

    # Set up
    # model
    model = ModelCNN_V0(input_shape=3,
                        num_classes=2)
    print(summary(model))

    # función de pérdida
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    ## entrenamiento
    train(model=model,
          train_dataloader=train_loader,
          test_dataloader=test_loader,
          optimizer=optimizer,
          loss_fn=loss_fn,
          device="cpu",
          epochs=EPOCHS)


if __name__ == "__main__":
    main()