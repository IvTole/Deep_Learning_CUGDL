# Logica detras del entrenamiento

# Entrenamiento de la red
import torch
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               X,
               y,
               loss:torch.nn.Module,
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

    # Poner los datos en el hardware
    X, y = X.to(device), y.to(device)

    # 1. Forwar propagation
    y_pred = model(X)

    # 2. Calcular perdida
    loss = loss(y_pred, y)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Se actualizan los pesos
    optimizer.step()

    # accuracy train
    metric = MulticlassAccuracy(num_classes=10)
    acc = metric(y_pred, y)


    return loss, acc

def test_step(model: torch.nn.Module,
               X,
               y,
               loss:torch.nn.Module,
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

    with torch.inference_mode():
        # Poner los datos en el hardware
        X, y = X.to(device), y.to(device)

        # 1. Forward propagation
        y_pred = model(X)

        #print(y_pred)

        # 2. Calcular perdida
        loss = loss(y_pred, y)

        # accuracy
        metric = MulticlassAccuracy(num_classes=10)
        acc = metric(y_pred, y)
        
    return loss, acc


def train(model,
          X_train,
          y_train,
          X_test,
          y_test,
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
            X=X_train,
            y=y_train,
            loss=loss_fn,
            optimizer=optimizer,
            device=device
        )

        print(f"Epoch: {epoch}, loss (train): {loss:.5f}")
        print(f"Epoch: {epoch}, accuracy (train): {acc:.5f}")

        loss, acc = test_step(
            model = model,
            X=X_test,
            y=y_test,
            loss=loss_fn,
            device=device
        )

        print(f"Epoch: {epoch}, loss (test): {loss:.5f}")
        print(f"Epoch: {epoch}, accuracy (test): {acc:.5f}")

