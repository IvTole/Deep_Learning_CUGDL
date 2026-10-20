# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.25.1", "torch", "torchvision", "torchmetrics", "mlxtend", "matplotlib", "pandas", "tqdm", "numpy", "requests"]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="Tema 13 · Redes neuronales, CNN")


@app.cell
def _(mo):
    mo.md(r"""
    <center> <span style="color:indigo">Deep Learning para Visión por Computadora</span> </center>

    <center>
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkuO0toNyrTS137Jgs3SMNqvwJrxMmxo-DQzCtQyFrVQ&s=10" alt="Drawing" style="width: 600px;"/>
    </center>

    <center> <span style="color:DarkBlue">  Tema 08: Redes neuronales, CNN </span>  </center>
    <center> <span style="color:Blue"> M. en C. Iván A. Toledano Juárez </span>  </center>

    # Redes Neuronales Convolucionales (CNN)

    En este notebook exploraremos el funcionamiento de las **redes neuronales convolucionales (CNN)** utilizando PyTorch. El objetivo es comprender su estructura, cómo procesan imágenes y cómo se entrenan para tareas de clasificación supervisada.

    Comenzaremos repasando los componentes fundamentales de una red neuronal clásica, para luego construir y comparar una CNN.
    """)
    return


@app.cell
def _():
    # PyTorch
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from helper_functions import accuracy_fn

    # TorchMetrics
    import torchmetrics, mlxtend
    import marimo as mo
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

    # Versiones
    print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
    return (
        ConfusionMatrix,
        DataLoader,
        Path,
        ToTensor,
        accuracy_fn,
        datasets,
        mo,
        nn,
        pd,
        plot_confusion_matrix,
        plt,
        random,
        timer,
        torch,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Cómo correr este notebook

    Podemos correr todo de jalón: los ejemplos se ejecutan, pero cada entrenamiento espera su botón. Escogemos las épocas y pulsamos **Entrenar**. Cada clic empieza con una semilla fija y un modelo nuevo.
    """)
    return


@app.cell
def _(torch):
    # Se usa GPU si se encuentra disponible
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")
    return (device,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset

    `torchvision.datasets`: aquí se encuentran varios datasets de ejemplo para problemas con imágenes para algunos contextos, como clasificación, detección de objetos, captions de imágenes, clasificación de videos, etc. También contiene una serie de clases baseline para crear datasets personalizados.

    La estructura básica de un tensor de imagen en PyTorch es:

    $$
    \text{Tensor shape} = (C, H, W)
    $$

    donde $C$ es el número de canales, $H$ la altura y $W$ el ancho.
    """)
    return


@app.cell
def _(Path, ToTensor, datasets):
    # Setup datos de entrenamiento
    train_data = datasets.MNIST(
        root=str(Path(__file__).resolve().parent / "mnist_numbers_data"), # dónde bajar los datos
        train=True, # obtener datos de entrenamiento
        download=True, # download si no se encuentra
        transform=ToTensor(), # images vienen en formato PIL, pasar a tensores de Torch
        target_transform=None
    )

    # Setup datos de prueba
    test_data = datasets.MNIST(
        root=str(Path(__file__).resolve().parent / "mnist_numbers_data"),
        train=False, # obtener datos de validación
        download=True,
        transform=ToTensor()
    )
    return test_data, train_data


@app.cell
def _(train_data):
    # forma del primer datos de entrenamiento
    image, _label = train_data[0]
    image, _label
    return (image,)


@app.cell
def _(image):
    image.shape
    # es una imagen blanco y negro con 28x28 pixeles
    return


@app.cell
def _(test_data, train_data):
    # Qué tantas muestras tenemos?
    len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets)
    return


@app.cell
def _(train_data):
    # Clases
    class_names = train_data.classes
    class_names
    return (class_names,)


@app.cell
def _(plt, train_data):
    # Plot de la primera imagen
    _image, _label = train_data[3]
    print(f"Shape: {_image.shape}")
    plt.imshow(_image.squeeze(), cmap='Greys')
    plt.title(_label)
    plt.gcf()
    return


@app.cell
def _(class_names, plt, torch, train_data):
    # Plot de más imágenes
    torch.manual_seed(42)
    _fig = plt.figure(figsize=(9, 9))
    _rows, _cols = 4, 4
    for _i in range(1, _rows * _cols + 1):
        _random_idx = torch.randint(0, len(train_data), size=[1]).item()
        _img, _label = train_data[_random_idx]
        _fig.add_subplot(_rows, _cols, _i)
        plt.imshow(_img.squeeze(), cmap="gray")
        plt.title(class_names[_label])
        plt.axis(False);
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Preparación de los datos

    La clase `torch.utils.data.DataLoader` permite cargar los datos en *batches* para entrenamiento y validación. El tamaño del batch (`batch_size`) define cuántas muestras se procesan simultáneamente en una pasada del modelo.

    Esto ayuda a optimizar el uso de memoria y permite el cálculo de gradientes por lotes:

    $$
    \text{Loss total} = \frac{1}{N} \sum_{i=1}^{N} \text{Loss}_i
    $$


    Normalmente se suelen tratar en potencias de 2, empezando por 32 (64,128, etc.)
    """)
    return


@app.cell
def _(DataLoader, test_data, train_data):
    # Se fija el hiperparámetro batch_size
    BATCH_SIZE = 32

    # Se realiza el método torch.utils.data.DataLoader para hacer los datasets iterables (batches)
    train_dataloader = DataLoader(train_data, # dataset -> iterable
        batch_size=BATCH_SIZE, # cuántas muestras por batch 
        shuffle=True # shuffle para cada época
    )

    test_dataloader = DataLoader(test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Resumen
    print(f"Dataloaders: {train_dataloader, test_dataloader}") 
    print(f"Train dataloader (numero de batches): {len(train_dataloader)} batches of {BATCH_SIZE}")
    print(f"Test dataloader (numero de batches): {len(test_dataloader)} batches of {BATCH_SIZE}")
    return test_dataloader, train_dataloader


@app.cell
def _(train_dataloader):
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    train_features_batch.shape, train_labels_batch.shape
    return train_features_batch, train_labels_batch


@app.cell
def _(class_names, plt, torch, train_features_batch, train_labels_batch):
    # Una muestra de un batch
    torch.manual_seed(42)
    _random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
    _img, _label = train_features_batch[_random_idx], train_labels_batch[_random_idx]
    plt.imshow(_img.squeeze(), cmap="gray")
    plt.title(class_names[_label])
    plt.axis("Off");
    print(f"Image size: {_img.shape}")
    print(f"Label: {_label}, label size: {_label.shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Construcción de un modelo base

    Como punto de partida, implementaremos una red neuronal *fully connected* (sin convoluciones) para comparar su desempeño con una CNN.

    La arquitectura general es:

    $$
    \text{Input} \rightarrow \text{Flatten} \rightarrow \text{Linear layers} \rightarrow \text{Output}
    $$

    Este modelo servirá como referencia para observar cómo las convoluciones mejoran la capacidad de extracción de patrones espaciales.
    """)
    return


@app.cell
def _(nn, train_features_batch):
    # Capa Flatten
    _flatten_model = nn.Flatten() # all nn modules function as a model (can do a forward pass)

    # Una sola muestra
    _x = train_features_batch[0]

    # Se aplana el outupt
    _output = _flatten_model(_x) # forward pass con el modelo basico

    # Resumen (ahora es un solo vector de features)
    print(f"Shape antes de flattening: {_x.shape} -> [color_channels, height, width]")
    print(f"Shape después de flattening: {_output.shape} -> [color_channels, height*width]")
    return


@app.cell
def _(nn):
    # Model 0
    class MNISTModelV0(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Flatten(), # la entrada de la red neuronal va a ser en forma de vector de features, como se tenía en notas pasadas
                nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = numero de features en los datos (784 píxeles)
                nn.Linear(in_features=hidden_units, out_features=output_shape)
            )

        def forward(self, x):
            return self.layer_stack(x)

    return (MNISTModelV0,)


@app.cell
def _(mo):
    mo.md(r"""
    Podemos empezar con los siguientes parámetros:

    - `input_shape`: se trata de un vector con cada uno de los píxeles de la imagen (28X28).

    - `hidden_units`: numero de neuronas en las capas ocultas, normalmente se empieza por algo pequeño (e.g. 10).

    - `output_shape`: es un problema multiclase, entonces necesitamos un numero de neuronas equivalente.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Función de pérdida y optimizador

    Para entrenar el modelo utilizaremos una función de pérdida (*loss function*) y un optimizador.

    Por ejemplo:
    - **CrossEntropyLoss**: usada para clasificación multiclase.
    - **SGD** o **Adam**: ajustan los pesos minimizando la pérdida.

    Durante el entrenamiento, los parámetros se actualizan según:

    $$
    \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta)
    $$

    donde $\eta$ es la tasa de aprendizaje (*learning rate*).
    """)
    return


@app.cell
def _(nn):
    loss_fn = nn.CrossEntropyLoss()
    return (loss_fn,)


@app.cell
def _(torch):
    # Una función para monitorear el tiempo de ejecución

    def print_train_time(start: float, end: float, device: torch.device = None):
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time

    return (print_train_time,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Entrenamiento del modelo

    Definimos el *loop de entrenamiento* que recorre las épocas y actualiza los parámetros del modelo mediante retropropagación.

    En cada iteración:
    1. Se realiza una pasada hacia adelante (*forward pass*).
    2. Se calcula la pérdida (*loss*).
    3. Se ejecuta la retropropagación (*backward pass*).
    4. Se actualizan los pesos usando el optimizador.

    El gradiente descendente se expresa como:

    $$
    \theta_{t+1} = \theta_t - \eta \frac{\partial L(\theta)}{\partial \theta}
    $$
    donde:
    - $\theta$ son los parámetros del modelo,
    - $\eta$ es la tasa de aprendizaje,
    - $L(\theta)$ la función de pérdida.

    ## Loop de entrenamiento

    A diferencia de los notebooks pasados, ahora tenemos que iterar sobre los BATCH, así que tenemos que agregar esto. En general los pasos serían los siguientes:

    1. Loop de épocas
    2. Loop de batch para entrenar (forward, loss/accuracy, zero grad, backpropagation, optimizador), con el cambio de que tenemos que ver la pérdida por batch.
    3. Loop de batch para validación (forward, loss), y se calcula la pérdida por batch.
    4. Le agregamos el timer.
    """)
    return


@app.cell
def _(mo):
    epocas_0 = mo.ui.number(start=1, stop=200, step=1, value=5, label="Épocas")
    entrenar_0 = mo.ui.run_button(label="Entrenar modelo 0")
    mo.hstack([epocas_0, entrenar_0])
    return entrenar_0, epocas_0


@app.cell
def _(
    MNISTModelV0,
    accuracy_fn,
    class_names,
    device,
    entrenar_0,
    epocas_0,
    loss_fn,
    mo,
    print_train_time,
    test_dataloader,
    timer,
    torch,
    tqdm,
    train_dataloader,
):
    mo.stop(not entrenar_0.value, mo.md("Pulsa **Entrenar modelo 0** para empezar."))



    torch.manual_seed(88) # semilla aleatoria

    # Preparamos el modelo
    model_0 = MNISTModelV0(input_shape=784, # uno por cada pixel
        hidden_units=10, # neuronas en las capas ocultas
        output_shape=len(class_names) # una por cada clase
    )
    model_0.to(device) # se manda a gpu
    _optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    # semilla aleatoria y timer
    torch.manual_seed(88)
    _train_time_start_on_cpu = timer()

    # Número de épocas (se empieza con un número pequeño)
    _epochs = epocas_0.value

    # Loop de entrenamiento y validación
    for _epoch in tqdm(range(_epochs)):
        print(f"Epoch: {_epoch}\n-------")
        ### Entrenamiento
        _train_loss = 0

        # Loop de batch
        for _batch, (_X, _y) in enumerate(train_dataloader):

            _X, _y = _X.to(device), _y.to(device) # para ponerlos en el hardware correcto

            model_0.train() 

            # 1. Forward
            _y_pred = model_0(_X)

            # 2. Se calcula la pérdida (por batch)
            _loss = loss_fn(_y_pred, _y)
            _train_loss += _loss.item() # se añade a la pérdida por época 

            # 3. Zero grad para optimizador
            _optimizer.zero_grad()

            # 4. Backpropagation para loss function
            _loss.backward()

            # 5. Optimizador
            _optimizer.step()

            # Se imprime que tantas muestras se han usado
            if _batch % 400 == 0:
                print(f"Looked at {_batch * len(_X)}/{len(train_dataloader.dataset)} samples")

        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        _train_loss /= len(train_dataloader)

        ### Validacion
        # Se definenen contadores para acumular loss y accuracy
        _test_loss, _test_acc = 0, 0 
        model_0.eval()
        with torch.inference_mode():
            for _X, _y in test_dataloader:

                _X, _y = _X.to(device), _y.to(device) # poner los datos en el hardware correcto

                # 1. Forward
                _test_pred = model_0(_X)

                # 2. Se calcula la pérdida (acumuladamente)
                _test_loss += loss_fn(_test_pred, _y).item() # se añade la perdida por epoca

                # 3. Accuracy (variables preds y y_true tienen que coincidir)
                _test_acc += accuracy_fn(y_true=_y, y_pred=_test_pred.argmax(dim=1))

            # Calculos sobre la métrica escogida
            # Los calculos se hacen dividiendo por la longitud del dataloader y por batch
            _test_loss /= len(test_dataloader)
            _test_acc /= len(test_dataloader)

        ## Resumen
        print(f"\nTrain loss: {_train_loss:.5f} | Test loss: {_test_loss:.5f}, Test acc: {_test_acc:.2f}%\n")

    # Se calcula tiempo de entrenamiento      
    _train_time_end_on_cpu = timer()
    total_train_time_model_0 = print_train_time(start=_train_time_start_on_cpu, 
                                               end=_train_time_end_on_cpu,
                                               device=str(next(model_0.parameters()).device))
    return model_0, total_train_time_model_0


@app.cell
def _(mo):
    mo.md(r"""
    Aunque no hemos metido capas de convolucion, al parecer los resultados son bastante buenos. A continuación se crea una función que va guardando en un diccionario los resultados por modelo creado.
    """)
    return


@app.cell
def _(device, torch):
    torch.manual_seed(42)
    def eval_model(model: torch.nn.Module, 
                   data_loader: torch.utils.data.DataLoader, 
                   loss_fn: torch.nn.Module, 
                   accuracy_fn, 
                   device: torch.device = device):
        loss, acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                # Mandar datos a device
                X, y = X.to(device), y.to(device)  ## En esa línea se considera el hardware
                y_pred = model(X)
                loss += loss_fn(y_pred, y).item()
                acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            loss /= len(data_loader)
            acc /= len(data_loader)
        return {"model_name": model.__class__.__name__,
                "model_loss": loss,
                "model_acc": acc}

    return (eval_model,)


@app.cell
def _(accuracy_fn, eval_model, loss_fn, model_0, test_dataloader):
    # Se calcula el modelo 0 a partir del set de validacion y se hacen pasar por la funcion
    model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
        loss_fn=loss_fn, accuracy_fn=accuracy_fn
    )
    model_0_results
    return (model_0_results,)


@app.cell
def _(mo):
    mo.md(r"""
    En estos casos, las pruebas también incluirían la inclusión de GPU's para acelerar el cálculo.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Primer modelo: red neuronal totalmente conectada

    En esta sección construimos un primer modelo (`MNISTModelV1`) basado en capas completamente conectadas (sin convoluciones).

    Su propósito es servir de referencia para comparar después con una red convolucional.

    La arquitectura típica es:
    $$
    \text{Flatten} \rightarrow \text{Linear} \rightarrow \text{ReLU} \rightarrow \text{Linear}
    $$

    A través de las funciones `train_step` y `test_step` se separa la lógica del entrenamiento y la validación, favoreciendo la reproducibilidad y claridad del código.
    """)
    return


@app.cell
def _(nn, torch):
    class MNISTModelV1(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Flatten(), 
                nn.Linear(in_features=input_shape, out_features=hidden_units),
                nn.ReLU(), # función de activación
                nn.Linear(in_features=hidden_units, out_features=output_shape),
                nn.ReLU() # función de activación
            )

        def forward(self, x: torch.Tensor):
            return self.layer_stack(x)

    return (MNISTModelV1,)


@app.cell
def _(mo):
    mo.md(r"""
    Como se suelen utilizar varias pruebas de diferentes modelos, es útil ponerlo todo en términos de funciones.
    """)
    return


@app.cell
def _(device, torch):
    def train_step(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   accuracy_fn,
                   device: torch.device = device):
        train_loss, train_acc = 0, 0
        model.to(device)
        model.train() # se vuelve a modo entrenamiento en cada época
        for batch, (X, y) in enumerate(data_loader):
            # Mandar datos a CPU/GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward
            y_pred = model(X)

            # 2. Loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            train_acc += accuracy_fn(y_true=y,
                                     y_pred=y_pred.argmax(dim=1)) # Logits -> pred labels

            # 3. Zero grad
            optimizer.zero_grad()

            # 4. Backpropagation Loss
            loss.backward()

            # 5. Optimizer
            optimizer.step()

        # Loss/Accuracy por epoca
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def test_step(data_loader: torch.utils.data.DataLoader,
                  model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  accuracy_fn,
                  device: torch.device = device):
        test_loss, test_acc = 0, 0
        model.to(device)
        model.eval() # poner al modelo en modo eval()
        # Inference context manager
        with torch.inference_mode(): 
            for X, y in data_loader:
                # Mandar datos a CPU/GPU
                X, y = X.to(device), y.to(device)

                # 1. Forward
                test_pred = model(X)

                # 2. Loss/Accuracy
                test_loss += loss_fn(test_pred, y).item()
                test_acc += accuracy_fn(y_true=y,
                    y_pred=test_pred.argmax(dim=1) # logits -> pred labels
                )

            # Se ajustan metricas de interés y se imprimen
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    return test_step, train_step


@app.cell
def _(mo):
    epocas_1 = mo.ui.number(start=1, stop=200, step=1, value=5, label="Épocas")
    entrenar_1 = mo.ui.run_button(label="Entrenar modelo 1")
    mo.hstack([epocas_1, entrenar_1])
    return entrenar_1, epocas_1


@app.cell
def _(
    MNISTModelV1,
    accuracy_fn,
    class_names,
    device,
    entrenar_1,
    epocas_1,
    loss_fn,
    mo,
    print_train_time,
    test_dataloader,
    test_step,
    timer,
    torch,
    tqdm,
    train_dataloader,
    train_step,
):
    mo.stop(not entrenar_1.value, mo.md("Pulsa **Entrenar modelo 1** para empezar."))

    # Parámetros del model 1
    torch.manual_seed(88)
    model_1 = MNISTModelV1(input_shape=784,
        hidden_units=10,
        output_shape=len(class_names) 
    ).to(device) # se mandan al gpu (si se encuentra disponible)
    next(model_1.parameters()).device # se imprime el hardware utilizado

    # Loss/accuracy

    _optimizer = torch.optim.SGD(params=model_1.parameters(), 
                                lr=0.1)

    torch.manual_seed(88)

    # Se inicializa tiempo
    _train_time_start_on_gpu = timer()

    _epochs = epocas_1.value
    for _epoch in tqdm(range(_epochs)):
        print(f"Epoch: {_epoch}\n---------")
        train_step(data_loader=train_dataloader, 
            model=model_1, 
            loss_fn=loss_fn,
            optimizer=_optimizer,
            accuracy_fn=accuracy_fn
        )
        test_step(data_loader=test_dataloader,
            model=model_1,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn
        )

    _train_time_end_on_gpu = timer()
    total_train_time_model_1 = print_train_time(start=_train_time_start_on_gpu,
                                                end=_train_time_end_on_gpu,
                                                device=device)
    return model_1, total_train_time_model_1


@app.cell
def _(mo):
    mo.md(r"""
    El hardware también tiene que coincidir a la hora de comparar modelos. La función creada anteriormente para el diccionario ya contemplaba esto, pero lo volvemos a mencionar.

    ## Uso de GPU y rendimiento

    Recordamos que PyTorch permite acelerar el entrenamiento si se dispone de GPU.
    """)
    return


@app.cell
def _(accuracy_fn, device, eval_model, loss_fn, model_1, test_dataloader):
    # Calculo del model 1 con el hardware
    model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
        loss_fn=loss_fn, accuracy_fn=accuracy_fn,
        device=device
    )
    model_1_results
    return (model_1_results,)


@app.cell
def _(model_0_results):
    model_0_results
    return


@app.cell
def _(mo):
    mo.md(r"""
    Ahora podemos comparar qué pasa al añadir la función de activación `ReLU()`.

    En machine learning, el **overfitting** ocurre cuando la precisión de un modelos es alta para los datos de entrenamiento pero baja significativamente para los datos nuevos. Esto puede pasar cuando un modelo es demasiado complejo para los datos de entrenamiento que le estamos dando, causando que "aprenda" demasiado sobre ellos en lugar de ver los patrones más profundos sobre ellos.

    Signos de que un modelo está sobreentrenado:

    * La precisión del modelo es muy alta para el set de entrenamiento.
    * La precisión del modelo cae significativamente con nuevos datos.
    * El score del set de validación es pobre.


    https://medium.com/@frederik.vl/interpreting-training-validation-accuracy-and-loss-cf16f0d5329f

    Algunas maneras de corregir este problema son las siguientes:

    * Usar un modelo menos complejo o totalmente diferente
    * Usar un dataset más grande
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Modelo convolucional (CNN)

    Ahora introducimos una **red neuronal convolucional (CNN)**, que es especialmente útil para el procesamiento de imágenes.

    Su arquitectura típica incluye:
    1. **Capas de convolución (`nn.Conv2d`)**: aplican filtros o kernels que extraen características espaciales locales.
    2. **Capas de activación (`ReLU`)**: introducen no linealidad.
    3. **Capas de *pooling* (`MaxPool2d`)**: reducen la dimensionalidad preservando características relevantes.
    4. **Capas lineales finales**: clasifican las representaciones extraídas.

    La operación de convolución se define como:
    $$
    S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i+m, j+n)K(m,n)
    $$
    donde:
    - $I$ es la imagen de entrada,
    - $K$ el kernel (filtro),
    - $S$ el mapa de activación resultante.
    """)
    return


@app.cell
def _(nn, torch):
    # Convolutional Neural Network
    class MNISTModelV2(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()

            # Primera capa de convolucion
            self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape, 
                          out_channels=hidden_units, 
                          kernel_size=3, # tamaño del kernel/filtro (normalmente de 3)
                          stride=1, # default
                          padding=1),# opciones = "valid" o "same" 
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, 
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, # Max Pooling
                             stride=2) # por default el valor del stride es igual al tamaño del kernel
            )

            # Segunda capa de convolucion
            self.block_2 = nn.Sequential(
                nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), # Hasta este paso es cuando se aplana el tensor
                # Cada capa de la red comprime y cambia el shape de los datos de entrada
                nn.Linear(in_features=hidden_units*7*7, 
                          out_features=output_shape)
            )

        def forward(self, x: torch.Tensor):
            x = self.block_1(x)
            # print(x.shape)
            x = self.block_2(x)
            # print(x.shape)
            x = self.classifier(x)
            # print(x.shape)
            return x

    return (MNISTModelV2,)


@app.cell
def _(torch):
    # Vemos que le pasa a los datos al propagarlos una vez por la convolucion

    torch.manual_seed(88)

    # Se crea un batch ficticio de numeros aleatorios
    _images = torch.randn(size=(32, 3, 64, 64))
    test_image = _images[0] # usamos uno de ellos para prueba
    print(f"Image batch shape: {_images.shape} -> [batch_size, color_channels, height, width]")
    print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
    print(f"Single image pixel values:\n{test_image}")
    return (test_image,)


@app.cell
def _(nn, test_image, torch):
    torch.manual_seed(88)

    conv_layer = nn.Conv2d(in_channels=3,
                           out_channels=10,
                           kernel_size=3,
                           stride=1,
                           padding=0) # "valid" o "same" 

    # Se pasan los datos por esta capa
    conv_layer(test_image).shape
    return (conv_layer,)


@app.cell
def _(nn, test_image, torch):
    torch.manual_seed(88)

    conv_layer_2 = nn.Conv2d(in_channels=3, # color channels
                             out_channels=10,
                             kernel_size=(5, 5), # kernel de tamaño 5,5 como tupla
                             stride=2,
                             padding=0)

    conv_layer_2(test_image.unsqueeze(dim=0)).shape
    return (conv_layer_2,)


@app.cell
def _(conv_layer_2):
    print(conv_layer_2.state_dict())
    # los pesos asignados
    return


@app.cell
def _(conv_layer_2):
    # Shapes and weights
    print(f"conv_layer_2 weight shape: \n{conv_layer_2.weight.shape} -> [out_channels=10, in_channels=3, kernel_size=5, kernel_size=5]")
    print(f"\nconv_layer_2 bias shape: \n{conv_layer_2.bias.shape} -> [out_channels=10]")
    return


@app.cell
def _(conv_layer, nn, test_image):
    # Al hacerlos pasar por una capa nn.MaxPool2D()

    # Shape original
    print(f"Test image original shape: {test_image.shape}")
    print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")

    # Capa de max pooling
    _max_pool_layer = nn.MaxPool2d(kernel_size=2)

    # Se propagan los datos por la capa de convolucion
    _test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
    print(f"Shape after going through conv_layer(): {_test_image_through_conv.shape}")

    # Se propagan los datos por la capa MaxPool2D
    _test_image_through_conv_and_max_pool = _max_pool_layer(_test_image_through_conv)
    print(f"Shape after going through conv_layer() and max_pool_layer(): {_test_image_through_conv_and_max_pool.shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    https://datascience.stackexchange.com/questions/64278/what-is-a-channel-in-a-cnn

    ## Análisis de dimensiones en convoluciones y pooling

    Vamos a analizar cómo cambian las **dimensiones** de una imagen a medida que pasa por capas convolucionales y de *pooling* en pytorch. Estos cálculos permiten entender cómo las CNN transforman los datos visuales en mapas de características cada vez más abstractas.

    #### 1. Imagen inicial

    * Shape del lote: `[batch_size, color_channels, height, width] = [32, 3, 64, 64]`
    * Shape de una sola imagen: `[3,64,64]`

    Cada imagen tiene 3 canales (RGB) y una resolución de 64x64 pixeles.

    #### 2. Primera convolución

    ```python
    conv_layer = nn.Conv2d(in_channels=3,
                           out_channels=10,
                           kernel_size=3,
                           stride=1,
                           padding=0)
    ```

    Esta capa aplica **10 filtros 3x3** sobre la imagen de entrada.

    El tamaño de salida se calcularía con las siguientes expresiones,

    $$
    H_{\rm out} = \frac{H_{\rm in} - K + 2P}{S} + 1, \quad W_{\rm out} = \frac{W_{\rm in} - K + 2P}{S} + 1
    $$

    de esta forma, $H_{\rm out} = W_{\rm out} = 62$

    Entonces la salida sería `[out_channels, height, width] = [10, 62, 62]`

    Esto significa que ahora tenemos **10 mapas de características**, cada uno de 62x62 píxeles, resultado de aplicar diferentes filtros a la imagen original.

    #### 3. Segunda convolución con stride

    ```python
    conv_layer_2 = nn.Conv2d(in_channels=3,
                             out_channels=10,
                             kernel_size=5,
                             stride=2,
                             padding=0)
    ```

    Para esta capa,

    $$
    H_{\rm out} = \frac{64 - 5 + 0}{2} + 1 = 30
    $$

    Por lo tanto, la salida es `[1, 10, 30, 30]`. El parámetro `stride=2` hace que el filtro/kernel se desplace dos píxeles a la vez, reduciendo la resolución aproximadamente a la mitad. Cada filtro produce un mapa más pequeño, pero captura patrones más amplios de la imagen.

    #### 4. Estructura de los pesos y sesgos

    ```python
    conv_layer_2.weight.shape  # [10, 3, 5, 5]
    conv_layer_2.bias.shape    # [10]
    ```

    * Cada filtro/kernel tiene shape `[in_channels, kernel_height, kernel_width] = [3, 5, 5]`.
    * Hay **10 filtros/kernels**, uno por cada mapa de salida
    * Total de parámetros entrenables,

    $$
    (3 \times 5 \times 5 + 1) \times 10 = 760
    $$

    Estos parámetros se optimizan durante el entrenamiento para que los filtros aprendan patrones relevantes como podrían ser bordes, texturas o contornos.

    #### 5. Capa de Max Pooling

    ```python
    max_pool_layer = nn.MaxPool2d(kernel_size=2)
    ```

    Esta operación selecciona el valor máximo en cada bloque $2 \times 2$, reduciendo la resolución y conservando la información más importante. La nueva resolución se calcula como

    $$
    H_{\rm out} = \frac{H_{\rm in} - K}{K} + 1
    $$

    Lo que resulta en $H_{\rm out} = 31$

    Por lo tanto la salida tiene shape `[1,10,31,31]`. Esta primera dimensión corresponde al tamaño de batch.

    El *pooling* mantiene el número de canales, pero reduce la altura y el ancho en un 50% en este caso.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Entrenamiento y validación del modelo convolucional

    Se repite el proceso de entrenamiento con el modelo convolucional, utilizando el mismo conjunto de datos.

    Compararemos las métricas de desempeño (pérdida y exactitud) entre:
    - Modelo base (lineal)
    - Modelo mejorado (MLP)
    - Modelo convolucional (CNN)

    El objetivo es mostrar cómo las CNN capturan mejor las relaciones espaciales en los datos de imágenes.
    """)
    return


@app.cell
def _(mo):
    epocas_2 = mo.ui.number(start=1, stop=200, step=1, value=5, label="Épocas")
    entrenar_2 = mo.ui.run_button(label="Entrenar modelo 2 · CNN")
    mo.hstack([epocas_2, entrenar_2])
    return entrenar_2, epocas_2


@app.cell
def _(
    MNISTModelV2,
    accuracy_fn,
    class_names,
    device,
    entrenar_2,
    epocas_2,
    loss_fn,
    mo,
    print_train_time,
    test_dataloader,
    test_step,
    timer,
    torch,
    tqdm,
    train_dataloader,
    train_step,
):
    mo.stop(not entrenar_2.value, mo.md("Pulsa **Entrenar modelo 2** para empezar."))

    torch.manual_seed(88)
    model_2 = MNISTModelV2(input_shape=1, 
        hidden_units=10, 
        output_shape=len(class_names)).to(device)
    model_2

    # Loss / Optimizer

    _optimizer = torch.optim.SGD(params=model_2.parameters(), 
                                 lr=0.1)

    torch.manual_seed(88)

    # Tiempo
    _train_time_start_model_2 = timer()

    # Training/Test 
    _epochs = epocas_2.value
    for _epoch in tqdm(range(_epochs)):
        print(f"Epoch: {_epoch}\n---------")
        train_step(data_loader=train_dataloader, 
            model=model_2, 
            loss_fn=loss_fn,
            optimizer=_optimizer,
            accuracy_fn=accuracy_fn,
            device=device
        )
        test_step(data_loader=test_dataloader,
            model=model_2,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )

    _train_time_end_model_2 = timer()
    total_train_time_model_2 = print_train_time(start=_train_time_start_model_2,
                                               end=_train_time_end_model_2,
                                               device=device)
    return model_2, total_train_time_model_2


@app.cell
def _(accuracy_fn, eval_model, loss_fn, model_2, test_dataloader):
    # Resultados del modelo 2
    model_2_results = eval_model(
        model=model_2,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )
    model_2_results
    return (model_2_results,)


@app.cell
def _(
    model_0_results,
    model_1_results,
    model_2_results,
    pd,
    total_train_time_model_0,
    total_train_time_model_1,
    total_train_time_model_2,
):
    compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
    compare_results["training_time"] = [total_train_time_model_0,
                                        total_train_time_model_1,
                                        total_train_time_model_2]
    compare_results
    return (compare_results,)


@app.cell
def _(compare_results, plt):
    compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
    plt.xlabel("accuracy (%)")
    plt.ylabel("model")
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Evaluación final y predicciones

    En esta etapa se evalúa el modelo final sobre el conjunto de prueba, generando predicciones para imágenes no vistas.

    El procedimiento general es:
    1. Enviar las imágenes a `device`.
    2. Aplicar `model.eval()` y usar `torch.inference_mode()` para evaluar sin calcular gradientes.
    3. Obtener las probabilidades de salida con `torch.softmax`.
    4. Asignar la clase más probable.

    Esto permite inspeccionar qué tan bien el modelo generaliza fuera del conjunto de entrenamiento.
    """)
    return


@app.cell
def _(device, torch):
    def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
        pred_probs = []
        model.eval()
        with torch.inference_mode():
            for sample in data:
                # Se prepara la muestra
                sample = torch.unsqueeze(sample, dim=0).to(device) # Se añade una dimensión extra (problemas de versiones)

                # Forward (el modelo tiene un output de logits(raw))
                pred_logit = model(sample)

                # (logit -> pred probability)
                # softmax es aplicado a logits, no a los batch.
                pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) 

                # Si estuvieran los datos en GPU, aqui se envían a CPU
                pred_probs.append(pred_prob.cpu())

        # las predicciones se vuelven a transformar en un tensor        
        return torch.stack(pred_probs)

    return (make_predictions,)


@app.cell
def _(class_names, random, test_data):
    random.seed(45)
    test_samples = []
    test_labels = []
    for _idx in random.sample(range(len(test_data)), k=9):
        _sample, _label = test_data[_idx]
        test_samples.append(_sample)
        test_labels.append(_label)

    # test/label para el primer datos de validacion
    print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")
    return test_labels, test_samples


@app.cell
def _(make_predictions, model_2, test_samples):
    # Ahora hacemos las predicciones para todos

    pred_probs= make_predictions(model=model_2, 
                                 data=test_samples)
    pred_probs[:2] # los primeros dos como lista de probabilidades
    return (pred_probs,)


@app.cell
def _(mo):
    mo.md(r"""
    Pasamos de predicciones a etiquetas con ``torch.argmax()``, aplicado al output del softmax.
    """)
    return


@app.cell
def _(pred_probs):
    # pasamos de predicciones a etiquetas

    pred_classes = pred_probs.argmax(dim=1)
    pred_classes
    return (pred_classes,)


@app.cell
def _(pred_classes, test_labels):
    test_labels, pred_classes
    return


@app.cell
def _(mo):
    mo.md(r"""
    Vemos a ahora sí son de la misma forma que las etiquetas que venían en los datos originales. Ya controlando esto, podemos hacer gráficos para ver las predicciones y ver las imágenes.
    """)
    return


@app.cell
def _(class_names, plt, pred_classes, test_labels, test_samples):
    # Grid de imagenes y su prediccion

    plt.figure(figsize=(9, 9))
    _nrows = 3
    _ncols = 3
    for _i, _sample in enumerate(test_samples):
      # Subplot
      plt.subplot(_nrows, _ncols, _i+1)

      # Se grafica la imagen
      plt.imshow(_sample.squeeze(), cmap="gray")

      # Se encuentra la prediccion (en texto)
      _pred_label = class_names[pred_classes[_i]]

      # Se encuentra el valor real (en texto)
      _truth_label = class_names[test_labels[_i]] 

      # El titulo incluye tanto la prediccion con el valor real
      _title_text = f"Pred: {_pred_label} | Truth: {_truth_label}"

      # Se agrega color verde si coinciden y color rojo si no
      if _pred_label == _truth_label:
          plt.title(_title_text, fontsize=10, c="g")
      else:
          plt.title(_title_text, fontsize=10, c="r")
      plt.axis(False)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matriz de confusión

    Podemos realizar una matriz de confusión sobre las predicciones, utilizando métodos de pytorch como ``torchmetrics.ConfusionMatrix``
    """)
    return


@app.cell
def _(device, model_2, test_dataloader, torch, tqdm):
    # Primero hacemos predicciones

    # Barra de progreso


    # 1. Hacemos predicciones con el modelo 2 ya entrenado
    _y_preds = []
    model_2.eval()
    with torch.inference_mode():
      for _X, _y in tqdm(test_dataloader, desc="Making predictions"):
        # Se mandan los datos al hardware
        _X, _y = _X.to(device), _y.to(device)
        # Forward
        _y_logit = model_2(_X)
        # logits -> prediction probabilities -> predictions labels
        _y_pred = torch.softmax(_y_logit, dim=1).argmax(dim=1)
        # Si fuera el caso, se ponene directamente los resultados en cpu
        _y_preds.append(_y_pred.cpu())
    # Se concatena la lista de predicciones en un tensor
    y_pred_tensor = torch.cat(_y_preds)
    return (y_pred_tensor,)


@app.cell
def _(
    ConfusionMatrix,
    class_names,
    plot_confusion_matrix,
    plt,
    test_data,
    y_pred_tensor,
):
    # Se ajusta la matriz de confusion con sus parametros
    _confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    _confmat_tensor = _confmat(preds=y_pred_tensor,
                             target=test_data.targets)

    # Se grafica la matriz de confusion
    _fig, _ax = plot_confusion_matrix(
        conf_mat=_confmat_tensor.numpy(), # tensor a numpy para usar matplotlib
        class_names=class_names,
        figsize=(10, 7)
    )
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Con la matriz de confusión uno podría evaluar ciertos errores del modelo al hacer predicciones. Por ejemplo, que el modelo confunda 2 con 7, etc. Aquí uno toma la decisión de cambiar etiquetas o mejorar el modelo, por ejemplo.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Guardar el modelo
    """)
    return


@app.cell
def _(mo):
    guardar_modelo = mo.ui.run_button(label="Guardar modelo CNN")
    guardar_modelo
    return (guardar_modelo,)


@app.cell
def _(Path, guardar_modelo, mo, model_2, torch):
    mo.stop(not guardar_modelo.value, mo.md("Pulsa **Guardar modelo** para guardar los pesos de la CNN."))
    _MODEL_PATH = Path(__file__).resolve().parent / "PyTorchModels"
    _MODEL_PATH.mkdir(parents=True, # parents directory
                     exist_ok=True # No marcar error si ya existe
    )

    # Path del archivo
    _MODEL_NAME = "pytorch_computer_vision_model_2.pth"
    MODEL_SAVE_PATH = _MODEL_PATH / _MODEL_NAME

    # Guardar el modelo
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_2.state_dict(), # solo se guardarían los pesos del state_dict()
               f=MODEL_SAVE_PATH)
    return (MODEL_SAVE_PATH,)


@app.cell
def _(MNISTModelV2, MODEL_SAVE_PATH, device, torch):
    # Lo cargamos para probarlo

    loaded_model_2 = MNISTModelV2(input_shape=1, 
                                        hidden_units=10,
                                        output_shape=10) 

    # Se carga el state_dict()
    loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=device, weights_only=True))

    # Mandar modelo a hardware
    loaded_model_2 = loaded_model_2.to(device)
    return (loaded_model_2,)


@app.cell
def _(
    accuracy_fn,
    eval_model,
    loaded_model_2,
    loss_fn,
    test_dataloader,
    torch,
):
    # Evaluamos el modelo y vemos que es el mismo
    torch.manual_seed(88)

    loaded_model_2_results = eval_model(
        model=loaded_model_2,
        data_loader=test_dataloader,
        loss_fn=loss_fn, 
        accuracy_fn=accuracy_fn
    )

    loaded_model_2_results
    return (loaded_model_2_results,)


@app.cell
def _(loaded_model_2_results, model_2_results, torch):
    # Por si son cercanos
    torch.isclose(torch.tensor(model_2_results["model_loss"]), 
                  torch.tensor(loaded_model_2_results["model_loss"]),
                  atol=1e-08, # tolerancia absoluta
                  rtol=0.0001) # tolerancia relativa
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
