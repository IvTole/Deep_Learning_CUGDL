import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <center> <span style="color:indigo">Machine Learning e Inferencia Bayesiana</span> </center>

    <center>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Centro_Universitario_del_Guadalajara_Logo.png/640px-Centro_Universitario_del_Guadalajara_Logo.png" alt="Drawing" style="width: 600px;"/>
    </center>

    <center> <span style="color:DarkBlue">  Tema 13: Redes neuronales, Regresión lineal </span>  </center>
    <center> <span style="color:Blue"> M. en C. Iván A. Toledano Juárez </span>  </center>

    # Regresión lineal con redes neuronales

    La **regresión lineal simple** es uno de los modelos fundamentales en aprendizaje automático supervisado. Su objetivo es encontrar una relación lineal entre una **variable independiente** $x$ y una **variable dependiente** $y$, es decir, ajustar una recta a un conjunto de datos:

    \begin{equation}
    \hat{y} = w \cdot x + b
    \end{equation}

    Donde:

    - $\hat{y}$: valor predicho por el modelo
    - $x$: valor de entrada (feature)
    - $w$: peso (pendiente) que define la inclinación de la recta
    - $b$: sesgo (intersección con el eje $y$)

    Aunque esta tarea puede resolverse fácilmente con álgebra lineal o librerías como scikit-learn, construir una regresión lineal como una **red neuronal simple** en PyTorch nos permite:

    - Aprender cómo se estructuran los modelos en PyTorch (`nn.Module`)
    - Comprender el flujo de entrenamiento: forward, loss, backward, optimizer
    - Aplicar las mismas herramientas que se usan en redes neuronales profundas (como `autograd` y `optim`)
    - Visualizar cómo los parámetros del modelo se ajustan con cada época

    Una red neuronal con:

    - **una sola neurona**
    - **activación lineal (sin no linealidades)**
    - y **una sola entrada**

    es equivalente a un modelo de regresión lineal simple.

    ---

    En este notebook implementaremos este modelo paso a paso, entrenándolo sobre un conjunto de datos de automóviles (`mtcars`). Veremos cómo el modelo aprende a aproximar la relación lineal entre $x$ y $y$ usando descenso de gradiente y optimización basada en pérdida.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import copy
    import io
    BASE_DIR = mo.notebook_dir()
    return BASE_DIR, io, mo


@app.cell
def _():
    # Importación de librerías

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    # pytorch
    import torch
    print('Version de PyTorch =',torch.__version__)
    from torch import nn # bloques fundamentales de nn de pytorch

    return nn, np, pd, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Importación de datos
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/mtcars/mtcars.csv")
    df['disp_0_46'] = df['disp']**(-0.46) # agregamos una variable transformada 
    df.head(5)
    return (df,)


@app.cell
def _(df, plt):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.scatter(df['disp_0_46'], df['mpg'])

    ax.set_xlabel(r'disp$^{-0.46}$')
    ax.set_ylabel('mpg')

    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preparación de datos
    """)
    return


@app.cell
def _(df, np):
    # pasamos a arrays y luego a tensores
    mpg_array = df['mpg'].to_numpy(copy=True)
    disp_array_046 = df['disp_0_46'].to_numpy(copy=True)
    total_array = np.concatenate((mpg_array,disp_array_046))
    return disp_array_046, mpg_array


@app.cell
def _(disp_array_046, mpg_array, torch):
    X = torch.from_numpy(disp_array_046).float()
    y = torch.from_numpy(mpg_array).float()
    return X, y


@app.cell
def _(X, y):
    ## Crear set de entrenamiento y validación

    train_split = int(0.8 * len(X)) # 80% de los datos para entrenamiento
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    len(X_train), len(y_train), len(X_test), len(y_test)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, plt, y_test, y_train):
    # Se crea una función para visualizar los datos

    def plot_predictions(train_data=X_train, 
                         train_labels=y_train, 
                         test_data=X_test, 
                         test_labels=y_test, 
                         predictions=None):
      """
      Se grafican los datos de entrenamiento, validación y se comparan predicciones
      """
      plt.figure(figsize=(8, 5))

      # Datos de entrenamiento (azul)
      plt.scatter(train_data, train_labels, c="b", s=12, label="Datos de entrenamiento")

      # Datos de prueba (verde)
      plt.scatter(test_data, test_labels, c="g", s=12, label="Datos de test")

      if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=12, label="Predictions")

      # Labels
      plt.legend(prop={"size": 14});

      plt.xlabel(r"disp$^{-0.46}$")
      plt.ylabel("mpg")
      return plt.gcf()

    return (plot_predictions,)


@app.cell
def _(plot_predictions):
    plot_predictions()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construcción de un modelo de regresión lineal con PyTorch

    En esta sección construiremos nuestro propio modelo de **regresión lineal simple** utilizando PyTorch. Lo haremos mediante la creación de una clase que hereda de `nn.Module`, el bloque fundamental para definir modelos en PyTorch.

    Recordemos que en la **regresión lineal**, el objetivo es aprender una relación de la forma:

    \begin{equation}
    \hat{y} = w \cdot x + b
    \end{equation}

    donde:

    - $x$ es la variable independiente (input),
    - $w$ es el peso o pendiente,
    - $b$ es el sesgo o término independiente (bias),
    - $\hat{y}$ es la predicción generada por el modelo.

    ---

    ### ¿Qué incluye la clase?

    - **`__init__()`**: define los parámetros entrenables del modelo (en este caso, `weights` y `bias`), ambos inicializados aleatoriamente mediante `torch.randn()` y envueltos en `nn.Parameter`, lo que le indica a PyTorch que deben ser optimizados durante el entrenamiento.

    - **`forward()`**: define cómo se calcula la salida del modelo a partir de una entrada `x`. En este caso, implementamos la fórmula de regresión lineal simple: $y = w \cdot x + b$.

    ---

    ### Consideraciones

    - **Herencia de `nn.Module`**: permite que el modelo integre automáticamente mecanismos de PyTorch como el tracking de gradientes, `state_dict()`, y uso de `.to(device)`.

    - **`requires_grad=True`**: permite que estos parámetros participen en el cálculo de gradientes durante el backpropagation.

    - Este diseño es muy flexible: en versiones más complejas se pueden añadir más capas, transformaciones no lineales, etc.

    A continuación, implementamos esta clase:
    """)
    return


@app.cell
def _():
    #class model(nn.Module):

    #    # constructor
    #    def __init__(self):
    #        super().__init__()
    #        self.weight = nn.Parameter(torch.randint(1,dtype=torch.float),requires_grad=True)
    #        self.bias = #nn.Parameter(torch.randint(1,dtype=torch.float),requires_grad=True)

    #    def forward
    return


@app.cell
def _(nn, torch):
    class LinearRegressionModel(nn.Module): # <- La entrada es un objeto nn.Module de PyTorch
        def __init__(self):
            super().__init__() 
            self.weights = nn.Parameter(torch.randn(1, # se comienzan con pesos aleatorios (que luego serán actualizados)
                                                    dtype=torch.float), # el dtype es float32 por default
                                       requires_grad=True) # True, para poder actualizar estos pesos con gradiente descendiente

            self.bias = nn.Parameter(torch.randn(1, # se comienza con un bias aleatorio (que luego será actualizado)
                                                dtype=torch.float),
                                    requires_grad=True) 

        # Forward define la forma de calcular en el modelo
        def forward(self, x: torch.Tensor) -> torch.Tensor: # "x" son los datos de entrada
            return self.weights * x + self.bias # formula de regresión lineal

    return (LinearRegressionModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fundamentos del módulo `torch.nn` en PyTorch

    El submódulo `torch.nn` es el núcleo de PyTorch para construir redes neuronales. Proporciona las herramientas necesarias para definir modelos de manera modular, incluyendo capas, funciones de activación, pérdidas y utilidades de entrenamiento.

    ---

    ### `torch.nn.Module`

    Es la **clase base de todos los modelos en PyTorch**. Al heredar de `nn.Module`, una clase personalizada puede:

    - Registrar automáticamente sus parámetros entrenables.
    - Integrarse con funcionalidades como `.parameters()`, `.state_dict()`, `.eval()`, `.to(device)`, etc.
    - Ser compatible con el sistema de gradientes de PyTorch (`autograd`).

    > Toda subclase de `nn.Module` debe implementar el método `forward()`.

    ---

    ### `torch.nn.Parameter`

    Es una subclase especial de `torch.Tensor` que indica a PyTorch que el tensor es **un parámetro entrenable del modelo**.

    Cuando se asigna un `nn.Parameter` como atributo de un `nn.Module`, este será incluido automáticamente en `.parameters()` y participará en el cálculo de gradientes (si `requires_grad=True`).

    ---

    ###  `def forward(self, x)`

    Este método debe ser implementado en toda subclase de `nn.Module` y define **cómo se calcula la salida a partir de la entrada**. Es la lógica de cómputo del modelo.

    Cuando se llama al modelo como función (`output = model(x)`), internamente se ejecuta el método `forward()`.

    ---

    ### `torch.nn`

    Es el submódulo general que contiene:

    - Clases para construir modelos (`nn.Module`, `nn.Sequential`)
    - Capas (`nn.Linear`, `nn.Conv2d`, etc.)
    - Funciones de activación (`nn.ReLU`, `nn.Sigmoid`, etc.)
    - Funciones de pérdida (`nn.MSELoss`, `nn.CrossEntropyLoss`, etc.)

    ---
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(BASE_DIR, mo):
    mo.image(BASE_DIR / "imagenes/pytorch_class.png")
    return


@app.cell
def _(LinearRegressionModel, torch):
    # Semilla aleatoria
    torch.manual_seed(88)

    # Creamos una primer instancia del modelo
    model_0 = LinearRegressionModel()
    #model_0 = model_0.to("mps")

    # Checamos los parámetros nn.Parameter(s) dentro de la subclase nn.Module que se creo
    print(list(model_0.parameters()))

    weight = model_0.state_dict()['weights'].clone()
    bias = model_0.state_dict()['bias'].clone()
    print(model_0.state_dict())
    return bias, model_0, weight


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Los tensores para los pesos y bias de entrada fueron generados aleatoriamente con `torch.randn`.
    """)
    return


@app.cell
def _(X_test, model_0, torch):
    # Predicciones
    # Ponemos como input el X_test para ver como predice el y_test
    # Estos datos pasarán por el método forward definido anteriormente forward() y se produce un resultado utilizando el cálculo que nosotros dijimos.

    with torch.inference_mode(): # Se utiliza este método para hacer inferencias (no entrenamiento)
        y_preds = model_0(X_test)

    # Checamos las predicciones
    print(f"Número de muestra (test): {len(X_test)}") 
    print(f"Número de predicciones: {len(y_preds)}")
    print(f"Valores predichos:\n{y_preds}")
    return (y_preds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No todas las inferencias tienen que ser un mapeo 1 a 1. En algunos casos podría ser que 100 valores X se encuentren mapeados a 1, 3 a 10, etc.
    """)
    return


@app.cell
def _(plot_predictions, y_preds):
    # Hacemos un plot de las predicciones
    plot_predictions(predictions=y_preds)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Vemos que estas predicciones son malísimas. Esto es de esperar pues empezamos con unos pesos totalmente aleatorios, que tienen que ser ajustados.
    """)
    return


@app.cell
def _(y_preds, y_test):
    # Checamos las desviaciones por valor
    y_test - y_preds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento del modelo

    Se crea una **loss function** a optimizar, además de un **optimizer**, que le dice al modelo como actualizar sus parámetros internos para mejor dicha función de pérdida. Valores típicos podrían ser,

    * **Loss function** ``torch.nn.L1Loss()``(MAE)
    * **Optimizer** ``torch.optim.SGD()``(Stochastic gradient descent), ``torch.optim.Adam()``, etc.

    En particular, utilizando ``torch.optim.SGD(params,lr)``, se tienen los siguientes parámetros.

    * **params**, son los parámetros target del modelo que se buscan optimizar (en este caso serían los pesos y bias que fueron aleatorios).
    * **lr**, es la taza de aprendizaje (**learning rate**) para el optimizador. Valores típicos pueden ser 0.01, 0.001, 0.0001, aunque pueden ser ajustados posteriormente.
    """)
    return


@app.cell
def _(nn):
    # Se crea la función de pérdida
    ##loss_fn = nn.L1Loss() # equivalente a Mean Absolute Error (MAE)
    loss_fn = nn.MSELoss() # equivalente e Mean Squared Error (MSE)

    # El optimizador se crea junto al modelo que vamos a entrenar.
    learning_rate = 0.03
    return learning_rate, loss_fn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Posteriormente, lo que se tiene que hacer es crear un loop de optimización. Esto significa que el modelo va a ir revisando los datos de entrenamiento y "aprendiendo" las relaciones entre los **features** y **labels**
    """)
    return


@app.cell
def _(BASE_DIR, mo):
    mo.image(BASE_DIR / "imagenes/pytorch_song.png")
    #https://www.youtube.com/watch?v=Nutpusq_AFw
    ##
    return


@app.cell
def _(
    X_test,
    X_train,
    learning_rate,
    loss_fn,
    model_0,
    torch,
    y_test,
    y_train,
):
    # Copiamos el modelo inicial para conservar las predicciones antes de entrenar.
    #model_entrenado = copy.deepcopy(model_0)
    model_entrenado = model_0
    optimizer = torch.optim.SGD(model_entrenado.parameters(), lr=learning_rate)
    # Training Loop

    # Número de época a entrenar (que tantas pasadas el modelo da sobre los datos de entrenamiento y se actualizan los pesos).
    epochs = 50000

    # Valores a comparar después
    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    # Comienza Training Loop (con un número de épocas)
    for epoch in range(epochs):

        # Poner el modelo en modo de entrenamiento (por default está así)
        model_entrenado.train()

        # 1. Propagación hacia adelante
        y_pred = model_entrenado(X_train)

        # 2. Calcular Loss Function
        loss = loss_fn(y_pred,y_train)

        # 3. Zero grad para el optimizador (se empieza en cero y se acumula en cada época)
        optimizer.zero_grad()

        # 4. Propagación hacia atrás (Backpropagation)
        loss.backward()

        # Progreso del optimizador
        optimizer.step()

        # Loop de testing

        # Poner el modelo en modo evaluation
        model_entrenado.eval()

        with torch.inference_mode():

            # 1. Propagación hacia adelant
            test_pred = model_entrenado(X_test)

            # 2. Se calcula la pérdida en los datos de test
            test_loss = loss_fn(test_pred, y_test.type(torch.float))

            # Prints
            if epoch % 1000 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MSE Train Loss: {loss} | MSE Test Loss: {test_loss} ")
    return epoch_count, model_entrenado, test_loss_values, train_loss_values


@app.cell
def _(epoch_count, plt, test_loss_values, train_loss_values):
    plt.figure(figsize=(8, 5))
    # Plot
    # Curva de pérdida

    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend();

    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Loss curves** Nos ayudan a ver cómo cambia la pérdida con las épocas. Esperamos que la pérdida de entrenamiento disminuya, aunque puede oscilar. La pérdida de validación puede aumentar si el modelo empieza a sobreajustarse.
    """)
    return


@app.cell
def _(bias, model_entrenado, weight):
    # Imprimimos los parámetros actuales del modelo
    print("El modelo tiene los siguientes valores para pesos y bias:")
    print(model_entrenado.state_dict())
    print("\nLos valores originales fueron:")
    print(f"weights: {weight}, bias: {bias}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicciones con el modelo entrenado de PyTorch
    """)
    return


@app.cell
def _(X_test, model_entrenado):
    print(X_test)
    print("Ejemplo con los parámetros aprendidos:")
    print(model_entrenado.weights.detach() * X_test[0] + model_entrenado.bias.detach())
    return


@app.cell
def _(X_test, model_entrenado, torch):
    # modo de prediccion
    model_entrenado.eval()

    with torch.inference_mode():
      y_preds_2 = model_entrenado(X_test)
    y_preds_2
    return (y_preds_2,)


@app.cell
def _(plot_predictions, y_preds_2):
    plot_predictions(predictions=y_preds_2)
    return


@app.cell
def _(y_preds_2):
    y_preds_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Guardar/Cargar modelo de PyTorch
    """)
    return


@app.cell
def _(io, mo, model_entrenado, torch):
    # Guardamos los parámetros en memoria y ofrecemos el archivo para descargar.
    # Esto evita sobrescribir un modelo cada vez que se ejecuta una celda.
    modelo_bytes = io.BytesIO()
    torch.save(model_entrenado.state_dict(), modelo_bytes)
    mo.download(modelo_bytes.getvalue(), filename="regresion_marimo.pth", label="Descargar modelo entrenado")
    return (modelo_bytes,)


@app.cell
def _(modelo_bytes):
    print(f"Tamaño del modelo: {len(modelo_bytes.getvalue())} bytes")
    return


@app.cell
def _(LinearRegressionModel, io, modelo_bytes, torch):
    # Cargar modelo

    # Se inicializa un modelo nuevo (con parámetros aleatorios)
    loaded_model_0 = LinearRegressionModel()

    # Se carga el archivo state_dict del modelo guardado, que actualiza los parámetros que ya venían entrenados.
    loaded_model_0.load_state_dict(torch.load(io.BytesIO(modelo_bytes.getvalue()), weights_only=True))
    return (loaded_model_0,)


@app.cell
def _(X_test, loaded_model_0, torch, y_preds_2):
    loaded_model_0.eval()
    with torch.inference_mode():
        loaded_model_preds = loaded_model_0(X_test)
    print("¿Coinciden las predicciones?", torch.allclose(y_preds_2, loaded_model_preds))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explorando los pesos y el bias

    Ya vimos cómo el optimizador ajusta los parámetros. Ahora intentemos hacerlo a mano.

    Movemos el peso $w$ y el bias $b$ para construir $\hat{y}=wx+b$. Observamos la recta y el MSE calculado **solo con entrenamiento**.

    **Checar:** ¿qué parámetro cambia la inclinación? ¿Cuál mueve toda la recta hacia arriba o abajo? Intenta bajar el MSE y luego compara con el modelo entrenado.
    """)
    return


@app.cell
def _(mo):
    peso_manual = mo.ui.slider(-100, 500, step=1, value=100, label="Peso (w)", show_value=True)
    bias_manual = mo.ui.slider(-30, 40, step=0.5, value=0, label="Bias (b)", show_value=True)
    mo.hstack([peso_manual, bias_manual])
    return bias_manual, peso_manual


@app.cell
def _(
    X,
    X_train,
    bias_manual,
    model_entrenado,
    peso_manual,
    plt,
    torch,
    y_train,
):
    pred_manual = peso_manual.value * X_train + bias_manual.value
    mse_manual = ((pred_manual - y_train) ** 2).mean().item()
    _fig_manual, _ax_manual = plt.subplots(figsize=(8, 5))
    _ax_manual.scatter(X_train, y_train, label="Entrenamiento")
    _x_recta = torch.linspace(float(X.min()), float(X.max()), 100)
    _ax_manual.plot(_x_recta, peso_manual.value * _x_recta + bias_manual.value, color="red", label="Recta manual")
    with torch.inference_mode():
        _ax_manual.plot(_x_recta, model_entrenado(_x_recta), color="green", label="Modelo entrenado")
    _ax_manual.set(xlabel=r"disp$^{-0.46}$", ylabel="mpg", title=f"MSE de entrenamiento (recta manual): {mse_manual:.2f}")
    _ax_manual.legend()
    _fig_manual
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
