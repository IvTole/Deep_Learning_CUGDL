# Logica detras del entrenamiento

# Entrenamiento de la red
import torch
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device: torch.device):
    """
    Esta función entrena un modelo de pytorch en una sola época.

    Args:
    model - Modelo de pytorch (red neuronal)
    X - Datos (características)
    y - Datos (target)
    loss - Función de pérdida
    optimizer - método de optimización de la función de pérdida
    device - hardware utilizado

    Return
    loss - pérdida del entrenamiento
    """
    # primer paso (modo de entrenamiento)
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        # Poner los datos en el hardware
        X, y = X.to(device), y.to(device)

        # 1. Forwar propagation
        y_pred = model.forward(X)

        # 2. Calcular perdida
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Se actualizan los pesos
        optimizer.step()

        # accuracy train
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               device: torch.device):
    """
    Esta función prueba un modelo de pytorch en una sola época.

    Args:
    model - Modelo de pytorch (red neuronal)
    X - Datos (características)
    y - Datos (target)
    loss - Función de pérdida
    device - hardware utilizado

    Return
    loss - pérdida del test
    """

    # primer paso (modo de evaluacion)
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):

            # Poner los datos en el hardware
            X, y = X.to(device), y.to(device)

            # 1. Forward propagation
            y_pred = model.forward(X)

            #print(y_pred)

            # 2. Calcular perdida
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # accuracy
            test_pred_labels = y_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

            # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
        
    return test_loss, test_acc


def train(model,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn,
          optimizer,
          device,
          epochs:int=2):
    
    """
    Esta función entrena un modelo de pytorch en una sola época.

    Args:
    model - Modelo de pytorch (red neuronal)
    X - Datos (características)
    y - Datos (target)
    loss_fn - Función de pérdida
    optimizer - método de optimización de la función de pérdida
    device - hardware utilizado

    Return
    loss - pérdida del entrenamiento
    """


    for epoch in range(0,epochs):

        loss, acc = train_step(
            model = model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        print(f"Epoch: {epoch}, loss (train): {loss:.5f}")
        print(f"Epoch: {epoch}, accuracy (train): {acc:.5f}")

        loss, acc = test_step(
            model = model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Epoch: {epoch}, loss (test): {loss:.5f}")
        print(f"Epoch: {epoch}, accuracy (test): {acc:.5f}")

