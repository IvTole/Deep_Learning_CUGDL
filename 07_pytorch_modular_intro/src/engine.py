# Logica detras del entrenamiento

# Entrenamiento de la red
import torch
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

    return loss

def train(model,
          X,
          y,
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

        loss = train_step(
            model = model,
            X=X,
            y=y,
            loss=loss_fn,
            optimizer=optimizer,
            device=device
        )

        print(f"Epoch: {epoch}, loss: {loss}")

#def test_step:
#    return
