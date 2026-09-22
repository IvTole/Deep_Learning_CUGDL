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

    <center> <span style="color:DarkBlue">  Tema 13: Redes neuronales, clasificacion </span>  </center>
    <center> <span style="color:Blue"> M. en C. Iván A. Toledano Juárez </span>  </center>

    # Clasificación con PyTorch

    Este notebook está basado en las notas de **MRDBourke** y utiliza datos del famoso dataset del **[Titanic](https://www.kaggle.com/competitions/titanic/overview)** disponible en Kaggle.
    El objetivo es construir un modelo de **clasificación binaria** con **PyTorch**, que prediga la probabilidad de supervivencia de los pasajeros.

    ## Objetivo
    Entrenar una red neuronal simple que aprenda a clasificar a los pasajeros del Titanic según las características disponibles, estimando si **sobrevivieron (1)** o **no sobrevivieron (0)**.

    ## Variables del dataset

    | Variable | Descripción | Valores posibles |
    |-----------|--------------|------------------|
    | `survival` | Supervivencia | 0 = No, 1 = Sí |
    | `pclass` | Clase del boleto | 1 = 1ª, 2 = 2ª, 3 = 3ª |
    | `sex` | Sexo | — |
    | `age` | Edad (en años) | — |
    | `sibsp` | Nº de hermanos / cónyuges a bordo | — |
    | `parch` | Nº de padres / hijos a bordo | — |
    | `ticket` | Número de boleto | — |
    | `fare` | Tarifa pagada | — |
    | `cabin` | Número de cabina | — |
    | `embarked` | Puerto de embarque | C = Cherbourg, Q = Queenstown, S = Southampton |

    ---

    A lo largo del notebook se realizará el **preprocesamiento de datos**, la **construcción del modelo**, y la **evaluación de su desempeño** mediante métricas de clasificación como **exactitud** y **matriz de confusión**.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import copy
    import io
    BASE_DIR = mo.notebook_dir()
    return copy, mo


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split

    import torch
    from torch import nn

    return nn, pd, torch, train_test_split


@app.cell
def _(pd):
    df_titanic_train = pd.read_csv("titanic_data/train.csv")
    df_titanic_test = pd.read_csv("titanic_data/test.csv")

    # Algunas variables no son necesarias

    variables = ['Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked','Survived']
    variables_2 = variables.copy()
    variables_2.remove('Survived')

    df_titanic_train = df_titanic_train[variables]
    df_titanic_test = df_titanic_test[variables_2]


    print(df_titanic_train.head(5))

    print('Shape(Train)',df_titanic_train.shape)
    print('Shape(test)',df_titanic_test.shape)
    return df_titanic_test, df_titanic_train


@app.cell
def _(df_titanic_train):
    df_titanic_train.info()
    return


@app.cell
def _(df_titanic_train):
    # Valores nulos
    df_titanic_train.isna().sum().to_frame(name="Valores nulos").head(83)
    return


@app.cell
def _(df_titanic_test):
    # Valores nulos
    df_titanic_test.isna().sum().to_frame(name="Valores nulos").head(83)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Podemos ignorar la variable Cabin. Para la variable de edad, (... podríamos rellenarlo con la media .... o no).
    """)
    return


@app.cell
def _(df_titanic_test, df_titanic_train, train_test_split):
    # Quitamos Cabin y separamos antes de calcular estadísticas.
    columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']
    df2_titanic_train, df_validacion = train_test_split(
        df_titanic_train[columns], test_size=0.2, random_state=42,
        stratify=df_titanic_train['Survived'])
    df2_titanic_train = df2_titanic_train.copy()
    df_validacion = df_validacion.copy()
    df2_titanic_test = df_titanic_test.drop(columns='Cabin').copy()
    # Las medias y la moda se aprenden SOLO con entrenamiento.
    for _col in ['Age', 'Fare']:
        _media = df2_titanic_train[_col].mean()
        for _datos in [df2_titanic_train, df_validacion, df2_titanic_test]:
            _datos[_col] = _datos[_col].fillna(_media)
    _moda = df2_titanic_train['Embarked'].mode()[0]
    for _datos in [df2_titanic_train, df_validacion, df2_titanic_test]:
        _datos['Embarked'] = _datos['Embarked'].fillna(_moda)
    return df2_titanic_test, df2_titanic_train, df_validacion


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **NOTA:** Primero separamos entrenamiento y validación. Si calculamos la media usando también validación, estaríamos usando información que el modelo no debería conocer (fuga de información).

    Aquí `X_test` y `y_test` serán nuestro set de **validación**. El archivo `test.csv` de Kaggle no trae la columna `Survived`, así que no lo usamos para calcular accuracy. Al comparar modelos o ajustar el umbral usamos validación; para reportar el desempeño final necesitamos un set de prueba independiente.
    """)
    return


@app.cell
def _(df2_titanic_train):
    # Variables categoricas y numéricas

    categorical = df2_titanic_train.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    numerical = df2_titanic_train.select_dtypes(include='number').columns.tolist()
    return (categorical,)


@app.cell
def _(categorical, df2_titanic_test, df2_titanic_train, df_validacion, pd):
    # Variables dummy con pandas, conservando las mismas columnas.
    df3_titanic_train = pd.get_dummies(df2_titanic_train, columns=categorical, dtype=int)
    df3_validacion = pd.get_dummies(df_validacion, columns=categorical, dtype=int).reindex(columns=df3_titanic_train.columns, fill_value=0)
    to_keep = df3_titanic_train.columns.tolist()
    df3_titanic_test = pd.get_dummies(df2_titanic_test, columns=categorical, dtype=int).reindex(columns=[_c for _c in to_keep if _c != 'Survived'], fill_value=0)
    return df3_titanic_train, df3_validacion, to_keep


@app.cell
def _(df3_titanic_train):
    df3_titanic_train
    return


@app.cell
def _(to_keep):
    # Features y class
    X_list = to_keep.copy()
    X_list.remove('Survived')
    Y_list = 'Survived'
    return X_list, Y_list


@app.cell
def _(X_list, Y_list, df3_titanic_train):
    # Dataframes a array

    X_array = df3_titanic_train[X_list].to_numpy(copy=True)
    Y_array = df3_titanic_train[Y_list].to_numpy(copy=True)
    return X_array, Y_array


@app.cell
def _(X_array):
    X_array
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arquitectura de una red neuronal para clasificación

    | Hiperparámetro | Clasificación binaria | Clasificación multiclase |
    | --- | --- | --- |
    | Capa de entrada | El mismo que el número de variables de entrada | Igual que clasificación binaria|
    | Capas ocultas | Depende del problema. Teóricamente puede ir de 1 a infinito | Igual que clasificación binaria|
    | Neuronas por capa oculta | Depende del probleme. Usualmente entre 10 y 512 | Igual que clasificación binaria|
    | Capas de salida | 1 (una por clase) | Una por cada clase|
    | Función de activación de capas ocultas | Típica: ReLU, pero puede ser cualquiera. | Igual que clasificación binaria|
    | Función de activación de capa de salida | Típica: Sigmoid | [Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) |
    | Loss Function | [Binary crossentropy](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) | [Crossentropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)|
    | Optimizador | SGD, [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) | Igual que clasificación binaria|
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arrays a tensores, y sets de entrenamiento y validación
    """)
    return


@app.cell
def _(X_array, Y_array, torch):
    X = torch.from_numpy(X_array).type(torch.float)
    y = torch.from_numpy(Y_array).type(torch.float)
    return X, y


@app.cell
def _(X, X_list, Y_list, df3_validacion, torch, y):
    # El split ya se hizo antes de imputar los valores faltantes.
    # Estandarizamos con las estadísticas del set de entrenamiento.
    media_X = X.mean(dim=0)
    std_X = X.std(dim=0).clamp_min(1e-6)
    X_train = (X - media_X) / std_X
    y_train = y
    X_test = (torch.tensor(df3_validacion[X_list].to_numpy(copy=True), dtype=torch.float32) - media_X) / std_X
    y_test = torch.tensor(df3_validacion[Y_list].to_numpy(copy=True), dtype=torch.float32)
    len(X_train), len(X_test), len(y_train), len(y_test)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creando el modelo

    Ya tenemos nuestros datos listos, así que es momento de **construir un modelo** de clasificación utilizando **PyTorch**.
    El proceso lo dividiremos en varios pasos clave:

    1. Configurar código **agnóstico al dispositivo** (CPU o GPU).
    2. Construir un modelo **subclasificando `nn.Module`**.
    3. Definir una **función de pérdida** y un **optimizador**.
    4. Crear un **bucle de entrenamiento**.

    La buena noticia es que ya hemos seguido estos pasos antes (en el notebook anterior), solo que ahora los **ajustaremos para un problema de clasificación**.

    ---

    ## Configuración del dispositivo

    Comenzamos importando las librerías necesarias y preparando el entorno para que el modelo pueda ejecutarse en **CPU o GPU**, según disponibilidad. Si tu equipo tiene acceso a una GPU compatible, pytorch la utilizará automáticamente. Esto permite que todo --datos, modelos y tensores -- se gestionesn en el dispositivo adecuado.
    """)
    return


@app.cell
def _(torch):
    # Fijar el tipo de hardware
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device
    return (device,)


@app.cell
def _(X_train):
    # Shapes de tensores

    X_train.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creación del modelo

    Queremos un modelo que reciba nuestros datos de entrada `X`(features) y produzca una predicción `y`, es decir, un tipo de problema supervisado. Para ello, definiremos una clase en python que,

    * Herede de `nn.module` (como todos los modelos de pytorch)
    * Cree dos capas lineales `nn.linear` en el constructor, con las dimensiones de entrada y salida adecuadas para nuestros datos.
    * Implemente un método `forward()` que defina la propagación hacia adelante del modelo
    * Instanciamos el modelo y lo envíamos al dispositivo configurado
    """)
    return


@app.cell
def _(device, nn, torch):
    torch.manual_seed(88) # reproducibilidad antes de crear los pesos
    # 1. Construimos la clase del modelo con la subclase nn.Module
    class ModelV0(nn.Module):
        def __init__(self):
            super().__init__()
            # 2. Creamos las capas de entrada (lineales) capaces de manejar los features de entrada y clase de salida
            self.layer_1 = nn.Linear(in_features=10, out_features=20) # toma 10 features (X), produce 20 features
            self.layer_2 = nn.Linear(in_features=20, out_features=1) # toma 20 features, produce 1 feature (y)

        # 3. Definimos un método para la propagación (forward)
        def forward(self, x):
            # Regresa la capa de salida de layer_2, un solo features, con el mismo shape que y
            # El calculo pasa sobre layer_1 y luego su output es el input de layer_2
            return self.layer_2(self.layer_1(x)) 

    # 4. Creamos una instancia con el modelo y se manda al hardware
    model_0 = ModelV0().to(device)
    model_0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La primera capa (`layer_1`) recibe 10 características de entrada (`in_features=10`) y produce 20 salidas (`out_features=20`). Estas 20 salidas se conocen como unidades ocultas, y forman una representación intermedia de los datos.

    La segunda capa (`layer_2`) toma esas 20 características y las transforma en una única salida (`out_features=1`), que corresponden a la predicción del modelo.

    **NOTA**: El número de unidades ocultas las elige uno. Más unidades podrían capturar patrones más complejos, pero también pueden provocar sobreajuste y entrenamiento más lento.

    ## `nn.Sequential`

    El método `nn.Sequential()` ejecuta la propagación hacia adelante en el orden en que aparecen las capas, simplificando la sintaxis cuando no se requieren pasos intermedios personalizados.

    **NOTA:** Si solo apilamos capas lineales, el resultado sigue siendo una transformación afín. Para aprender relaciones no lineales necesitamos una activación entre capas, como ReLU.
    """)
    return


@app.cell
def _(device, nn, torch):
    torch.manual_seed(88) # reproducibilidad antes de crear los pesos
    # Se replica el modelV0
    model_0_2 = nn.Sequential(
        nn.Linear(in_features=10, out_features=20),
        nn.Linear(in_features=20, out_features=1)
    ).to(device)

    model_0_2
    return (model_0_2,)


@app.cell
def _(X_test, device, model_0_2, y_test):
    # Hacemos predicciones con el modelo
    untrained_preds = model_0_2(X_test.to(device))
    print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
    print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
    print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
    print(f"\nFirst 10 test labels:\n{y_test[:10]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuración de la función de pérdida y el optimizador

    Ya hemos configurado modelos, así que ahora toca definir **cómo aprenderá**, a través de una **función de pérdida** (*loss function*) y un **optimizador**.

    En el notebook previo ya usamos estos conceptos, pero es importante notar que **diferentes tipos de problemas requieren distintas funciones de pérdida.**

    ---

    ## Función de pérdida

    La función de pérdida (también llamada *cost function*) mide **qué tan equivocadas son las predicciones del modelo**. Mientras más alto sea su valor, peor está aprendiendo el modelo.  El entrenamiento consiste en **minimizar esta pérdida**.

    Ejemplos comunes:

    | Función / Optimizador | Tipo de problema | Código en PyTorch |
    |------------------------|------------------|-------------------|
    | Stochastic Gradient Descent (SGD) | Clasificación, regresión, muchos otros | `torch.optim.SGD()` |
    | Adam Optimizer | Clasificación, regresión, muchos otros | `torch.optim.Adam()` |
    | Binary Cross Entropy (BCE) | Clasificación binaria | `torch.nn.BCELoss()` o `torch.nn.BCEWithLogitsLoss()` |
    | Cross Entropy | Clasificación multiclase | `torch.nn.CrossEntropyLoss()` |
    | Mean Absolute Error (MAE) / L1 Loss | Regresión | `torch.nn.L1Loss()` |
    | Mean Squared Error (MSE) / L2 Loss | Regresión | `torch.nn.MSELoss()` |

    ---

    ## Elección para nuestro caso

    Como estamos trabajando con un **problema de clasificación binaria**, la opción más adecuada es usar una **pérdida de entropía cruzada binaria** (*binary cross entropy loss*).

    PyTorch ofrece dos versiones:

    - `torch.nn.BCELoss()`
      Calcula la entropía cruzada binaria entre las predicciones y las etiquetas.

    - `torch.nn.BCEWithLogitsLoss()`
      Hace lo mismo, pero **integra internamente una función sigmoide** (`nn.Sigmoid`).
      Esto la hace **más estable numéricamente** y generalmente se recomienda sobre la anterior.

    > 💡 **Recomendación:**
    > Usa `torch.nn.BCEWithLogitsLoss()` en la mayoría de los casos de clasificación binaria.
    > Evita aplicar manualmente un `Sigmoid` si utilizas esta versión.

    ---

    ## Optimizador

    El optimizador es el algoritmo que **ajusta los pesos del modelo** para minimizar la pérdida.  Podemos usar el clásico **descenso de gradiente estocástico (SGD)** o el más moderno **Adam**. Ambos funcionan bien, pero empezaremos con **SGD** para mayor claridad.
    """)
    return


@app.cell
def _(nn):
    # Loss Function
    loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in
    return (loss_fn,)


@app.cell
def _(torch):
    # Métrica de evaluación (accuracy)
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calcula si dos tensores son iguales
        acc = (correct / len(y_pred)) * 100 
        return acc

    return (accuracy_fn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento del modelo

    Antes de realizar el loop de entrenamiento, veamos que sale del modelo al realizar una propagación hacia adelante, usando los datos de validacion.
    """)
    return


@app.cell
def _(X_test, device, model_0_2):
    # Los 5 primeros outputs
    y_logits = model_0_2(X_test.to(device))[:5]
    y_logits
    return (y_logits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Como el modelo todavía **no ha sido entrenado**, sus salidas son esencialmente **valores aleatorios**.  Durante la **propagación hacia adelante**, los datos pasan a través de las dos capas lineales definidas, las cuales aplican internamente la siguiente ecuación:

    \begin{equation}
    y = x \cdot w^T + \mathrm{bias}
    \end{equation}

    Los valores resultantes $y$ de esta operación, así como los que produce el modelo, se conocen como **_logits_**.  En un modelo puramente lineal, estos logits representarían simplemente **valores numéricos** sin restricción en su rango (pueden ser negativos o positivos).

    Si aplicamos una **función de activación sigmoide** sobre ellos, podemos convertir dichos valores en **probabilidades** dentro del intervalo $(0, 1)$, lo que nos permite interpretar el resultado como:

    \begin{equation}
    \text{probabilidad de clase positiva} = \sigma(y) = \frac{1}{1 + e^{-y}}
    \end{equation}

    De este modo, al establecer un **umbral (threshold)** —por ejemplo, 0.5— podemos transformar las probabilidades en una **clasificación binaria**:

    - Si $\sigma(y) \ge 0.5$ → clase **1 (positivo)**
    - Si $\sigma(y) < 0.5$ → clase **0 (negativo)**

    > 💡 **Nota:** En PyTorch, cuando se utiliza `nn.BCEWithLogitsLoss`, la función sigmoide ya está incorporada dentro de la función de pérdida, por lo que **no es necesario aplicarla manualmente** en la salida del modelo.
    """)
    return


@app.cell
def _(torch, y_logits):
    # Sigmoid
    y_pred_probs = torch.sigmoid(y_logits)
    y_pred_probs
    return (y_pred_probs,)


@app.cell
def _(X_test, device, model_0_2, torch, y_pred_probs):
    # Redondeamos para obtener una clasificación (threshold 0.5)
    #y_preds = torch.round(y_pred_probs)
    threshold = 0.5
    y_preds = (y_pred_probs >= threshold).float()

    y_pred_labels = torch.round(torch.sigmoid(model_0_2(X_test.to(device))[:5]))

    # Checamos igualdad
    print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

    # Quitamos la dimensión extra
    y_preds.squeeze()
    return


@app.cell
def _(y_test):
    y_test[:5]
    # Vemos que ahora si tenemos las etiquetas que queremos
    return


@app.cell
def _(
    X_test,
    X_train,
    accuracy_fn,
    copy,
    device,
    loss_fn,
    model_0_2,
    torch,
    y_test,
    y_train,
):
    X_test_2 = X_test
    X_train_2 = X_train
    y_test_2 = y_test
    y_train_2 = y_train
    modelo_lineal = copy.deepcopy(model_0_2)
    optimizer = torch.optim.SGD(modelo_lineal.parameters(), lr=0.01)
    torch.manual_seed(88) # semilla aleatoria

    # Número de epocas
    epochs = 4000

    # Poner los datos en el hardware target
    X_train_2, y_train_2 = X_train_2.to(device), y_train_2.to(device)
    X_test_2, y_test_2 = X_test_2.to(device), y_test_2.to(device)

    # Loop de training y eval
    for epoch in range(epochs):
        ### Training
        modelo_lineal.train()

        # 1. Forward propagation (el modelo regresa logits)
        y_logits_2 = modelo_lineal(X_train_2).squeeze() # squeeze para remover `1` dimension extra
        y_pred = torch.round(torch.sigmoid(y_logits_2)) # logits -> pred probs -> pred labls

        # 2. Se calcula loss/accuracy
        # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
        #                y_train) 
        loss = loss_fn(y_logits_2, # nn.BCEWithLogitsLoss acepta los logits de salida
                       y_train_2) 
        acc = accuracy_fn(y_true=y_train_2, 
                          y_pred=y_pred) 

        # 3. Zero grad para el optimizador
        optimizer.zero_grad()

        # 4. Back propagation
        loss.backward()

        # 5. Optimizador
        optimizer.step()

        ### Evaluacion
        modelo_lineal.eval()
        with torch.inference_mode():
            # 1. Forward 
            test_logits = modelo_lineal(X_test_2).squeeze() 
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 2. loss/accuracy
            test_loss = loss_fn(test_logits,
                                y_test_2)
            test_acc = accuracy_fn(y_true=y_test_2,
                                   y_pred=test_pred)

        # Print cada 10 epocas
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    return X_test_2, X_train_2, modelo_lineal, y_test_2, y_train_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mejorando el modelo

    Una vez que el modelo básico está funcionando, existen diversas estrategias para **mejorar su desempeño**.  Cada una de las siguientes técnicas busca aumentar la capacidad del modelo para **aprender patrones más complejos** o **ajustarse mejor a los datos**.

    ---

    ### 1. Añadir más capas
    Si incluimos activaciones no lineales entre capas, cada capa adicional puede incrementar la **capacidad de representación** del modelo, permitiéndole aprender **patrones más abstractos y no lineales**. Agregar más capas hace que la red sea más **profunda**, lo que da origen al término *deep learning*.

    ---

    ### 2. Añadir más neuronas ocultas
    De forma similar, aumentar el número de **neuronas (unidades ocultas)** dentro de una capa puede mejorar la capacidad del modelo para capturar relaciones complejas entre las variables. Sin embargo, demasiadas neuronas pueden llevar al **sobreajuste (overfitting)**.

    ---

    ### 3. Entrenar por más épocas
    Dar al modelo más **épocas** (iteraciones completas sobre los datos) permite que los pesos se actualicen más veces, lo que puede mejorar el rendimiento si el modelo aún no ha convergido. Pero un número excesivo de épocas también puede causar **sobreajuste**.

    ---

    ### 4. Cambiar la función de activación
    Los datos reales rara vez son lineales. Usar funciones de activación **no lineales** (como ReLU, tanh o sigmoid) permite que el modelo aprenda relaciones más complejas.
    Por ejemplo:
    - `nn.ReLU()` → común en redes profundas.
    - `nn.Sigmoid()` → útil en clasificación binaria.
    - `nn.Tanh()` → centrada en 0, útil para ciertos tipos de datos.

    ---

    ### 5. Ajustar la tasa de aprendizaje
    La **tasa de aprendizaje** (`learning_rate`) controla qué tanto se ajustan los parámetros en cada actualización.
    - Si es **demasiado alta**, el modelo puede **oscilar o divergir**.
    - Si es **demasiado baja**, el aprendizaje será **muy lento** o se quedará estancado en un mínimo local.

    Encontrar un valor adecuado requiere **experimentación o búsqueda sistemática (grid/random search)**.

    ---

    ### 6. Cambiar la función de pérdida
    Cada tipo de problema (clasificación binaria, multiclase, regresión, etc.) requiere una **función de pérdida diferente**.
    Probar distintas opciones puede mejorar la estabilidad o precisión del aprendizaje.

    Ejemplo:
    - Clasificación binaria → `nn.BCEWithLogitsLoss()`
    - Clasificación multiclase → `nn.CrossEntropyLoss()`
    - Regresión → `nn.MSELoss()` o `nn.L1Loss()`

    ---

    ### 7. Transfer learning
    En lugar de entrenar un modelo desde cero, se puede **aprovechar un modelo preentrenado** en un problema similar y **ajustarlo (fine-tuning)** a los nuevos datos.
    Esta técnica es muy útil cuando se dispone de **pocos datos** o se trabaja con **dominios complejos**, como imágenes o texto.

    ---

    > **NOTA:** No existe una receta única para mejorar el modelo.
    > La práctica más común es **ajustar un hiperparámetro a la vez**, observar su impacto en la pérdida y en las métricas de validación, y repetir el proceso hasta encontrar un equilibrio entre **precisión y generalización**.
    """)
    return


@app.cell
def _(device, nn, torch):
    torch.manual_seed(88) # reproducibilidad antes de crear los pesos
    class ModelV1(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=10, out_features=20)
            self.layer_2 = nn.Linear(in_features=20, out_features=20) # capa extra
            self.layer_3 = nn.Linear(in_features=20, out_features=1)

        def forward(self, x): 
            # z = self.layer_1(x)
            # z = self.layer_2(z)
            # z = self.layer_3(z)
            # return z
            return self.layer_3(self.layer_2(self.layer_1(x)))

    model_1 = ModelV1().to(device)
    model_1
    return (model_1,)


@app.cell
def _(nn):
    loss_fn_2 = nn.BCEWithLogitsLoss()
    return (loss_fn_2,)


@app.cell
def _(
    X_test_2,
    X_train_2,
    accuracy_fn,
    copy,
    device,
    loss_fn_2,
    model_1,
    torch,
    y_test_2,
    y_train_2,
):
    X_test_3 = X_test_2
    X_train_3 = X_train_2
    y_test_3 = y_test_2
    y_train_3 = y_train_2
    modelo_profundo = copy.deepcopy(model_1)
    optimizer_2 = torch.optim.SGD(modelo_profundo.parameters(), lr=0.01)
    torch.manual_seed(88) # semilla aleatoria

    # Número de epocas
    epochs_2 = 5000

    # Poner los datos en el hardware target
    X_train_3, y_train_3 = X_train_3.to(device), y_train_3.to(device)
    X_test_3, y_test_3 = X_test_3.to(device), y_test_3.to(device)

    # Loop de training y eval
    for epoch_2 in range(epochs_2):
        ### Training
        modelo_profundo.train()

        # 1. Forward propagation (el modelo regresa logits)
        y_logits_3 = modelo_profundo(X_train_3).squeeze() # squeeze para remover `1` dimension extra
        y_pred_2 = torch.round(torch.sigmoid(y_logits_3)) # logits -> pred probs -> pred labls

        # 2. Se calcula loss/accuracy
        # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
        #                y_train) 
        loss_2 = loss_fn_2(y_logits_3, # nn.BCEWithLogitsLoss acepta los logits de salida
                       y_train_3) 
        acc_2 = accuracy_fn(y_true=y_train_3, 
                          y_pred=y_pred_2) 

        # 3. Zero grad para el optimizador
        optimizer_2.zero_grad()

        # 4. Back propagation
        loss_2.backward()

        # 5. Optimizador
        optimizer_2.step()

        ### Evaluacion
        modelo_profundo.eval()
        with torch.inference_mode():
            # 1. Forward 
            test_logits_2 = modelo_profundo(X_test_3).squeeze() 
            test_pred_2 = torch.round(torch.sigmoid(test_logits_2))
            # 2. loss/accuracy
            test_loss_2 = loss_fn_2(test_logits_2,
                                y_test_3)
            test_acc_2 = accuracy_fn(y_true=y_test_3,
                                   y_pred=test_pred_2)

        # Print cada 10 epocas
        if epoch_2 % 100 == 0:
            print(f"Epoch: {epoch_2} | Loss: {loss_2:.5f}, Accuracy: {acc_2:.2f}% | Test loss: {test_loss_2:.5f}, Test acc: {test_acc_2:.2f}%")
    return X_test_3, X_train_3, modelo_profundo, y_test_3, y_train_3


@app.cell
def _(device, nn, torch):
    torch.manual_seed(88) # reproducibilidad antes de crear los pesos
    class ModelV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=10, out_features=20)
            self.layer_2 = nn.Linear(in_features=20, out_features=20)
            self.layer_3 = nn.Linear(in_features=20, out_features=1)
            self.relu = nn.ReLU() # <- Se añade función de activación ReLU
            # Sigmoid también puede usarse entre capas; la salida conserva logits. 
            # self.sigmoid = nn.Sigmoid()

        def forward(self, x):
          # ReLU se aplica entre capas
           return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

    model_3 = ModelV2().to(device)
    print(model_3)
    return (model_3,)


@app.cell
def _(
    X_test_3,
    X_train_3,
    accuracy_fn,
    copy,
    device,
    loss_fn_2,
    model_3,
    torch,
    y_test_3,
    y_train_3,
):
    X_test_4 = X_test_3
    X_train_4 = X_train_3
    y_test_4 = y_test_3
    y_train_4 = y_train_3
    modelo_relu = copy.deepcopy(model_3)
    optimizer_3 = torch.optim.SGD(modelo_relu.parameters(), lr=0.01)
    torch.manual_seed(88) # semilla aleatoria

    # Número de epocas
    epochs_3 = 5000

    # Poner los datos en el hardware target
    X_train_4, y_train_4 = X_train_4.to(device), y_train_4.to(device)
    X_test_4, y_test_4 = X_test_4.to(device), y_test_4.to(device)

    # Loop de training y eval
    for epoch_3 in range(epochs_3):
        ### Training
        modelo_relu.train()

        # 1. Forward propagation (el modelo regresa logits)
        y_logits_4 = modelo_relu(X_train_4).squeeze() # squeeze para remover `1` dimension extra
        y_pred_3 = torch.round(torch.sigmoid(y_logits_4)) # logits -> pred probs -> pred labls

        # 2. Se calcula loss/accuracy
        # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
        #                y_train) 
        loss_3 = loss_fn_2(y_logits_4, # nn.BCEWithLogitsLoss acepta los logits de salida
                       y_train_4) 
        acc_3 = accuracy_fn(y_true=y_train_4, 
                          y_pred=y_pred_3) 

        # 3. Zero grad para el optimizador
        optimizer_3.zero_grad()

        # 4. Back propagation
        loss_3.backward()

        # 5. Optimizador
        optimizer_3.step()

        ### Evaluacion
        modelo_relu.eval()
        with torch.inference_mode():
            # 1. Forward 
            test_logits_3 = modelo_relu(X_test_4).squeeze() 
            test_pred_3 = torch.round(torch.sigmoid(test_logits_3))
            # 2. loss/accuracy
            test_loss_3 = loss_fn_2(test_logits_3,
                                y_test_4)
            test_acc_3 = accuracy_fn(y_true=y_test_4,
                                   y_pred=test_pred_3)

        # Print cada 10 epocas
        if epoch_3 % 100 == 0:
            print(f"Epoch: {epoch_3} | Loss: {loss_3:.5f}, Accuracy: {acc_3:.2f}% | Test loss: {test_loss_3:.5f}, Test acc: {test_acc_3:.2f}%")
    return X_test_4, modelo_relu, y_test_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Del logit a la decisión: explorando el umbral

    El modelo produce logits; `torch.sigmoid` los transforma en probabilidades. El **umbral** decide a partir de qué probabilidad asignamos la clase 1.

    Cambiamos el umbral sin volver a entrenar. También podemos comparar las tres arquitecturas anteriores sobre el mismo set de validación.

    * **Falso positivo:** predecimos supervivencia, pero la etiqueta es 0.
    * **Falso negativo:** predecimos que no sobrevivió, pero la etiqueta es 1.
    * **Precision:** de los que clasificamos como supervivientes, ¿qué proporción realmente sobrevivió?
    * **Recall:** de quienes sobrevivieron, ¿qué proporción encontramos?

    **Checar:** al bajar el umbral, ¿aumentan o disminuyen los falsos negativos? ¿Un accuracy mayor siempre implica un recall mayor?

    **NOTA:** La comparación de arquitecturas también depende de las épocas y del optimizador. Los ejemplos anteriores usan 400 épocas para el primer modelo y 500 para los otros dos; no es una comparación controlada del efecto de añadir capas.
    """)
    return


@app.cell
def _(mo):
    modelo_seleccionado = mo.ui.dropdown(["Dos capas lineales", "Tres capas lineales", "Tres capas con ReLU"], value="Tres capas con ReLU", label="Modelo")
    umbral = mo.ui.slider(0, 1, step=0.05, value=0.5, label="Umbral", show_value=True)
    mo.hstack([modelo_seleccionado, umbral])
    return modelo_seleccionado, umbral


@app.cell
def _(
    X_test_4,
    device,
    modelo_lineal,
    modelo_profundo,
    modelo_relu,
    modelo_seleccionado,
    torch,
):
    _modelos = {"Dos capas lineales": modelo_lineal, "Tres capas lineales": modelo_profundo, "Tres capas con ReLU": modelo_relu}
    with torch.inference_mode():
        probabilidades_validacion = torch.sigmoid(_modelos[modelo_seleccionado.value](X_test_4.to(device)).squeeze()).cpu()
    return (probabilidades_validacion,)


@app.cell
def _(mo, pd, probabilidades_validacion, umbral, y_test_4):
    etiquetas_umbral = (probabilidades_validacion >= umbral.value).int()
    _reales = y_test_4.cpu().int()
    _tp = int(((etiquetas_umbral == 1) & (_reales == 1)).sum())
    _tn = int(((etiquetas_umbral == 0) & (_reales == 0)).sum())
    _fp = int(((etiquetas_umbral == 1) & (_reales == 0)).sum())
    _fn = int(((etiquetas_umbral == 0) & (_reales == 1)).sum())
    accuracy_umbral = (_tp + _tn) / len(_reales)
    _precision = f"{_tp / (_tp + _fp):.3f}" if _tp + _fp else "No definida (sin positivos predichos)"
    _recall = f"{_tp / (_tp + _fn):.3f}" if _tp + _fn else "No definido (sin positivos reales)"
    matriz_confusion = pd.DataFrame([[_tn, _fp], [_fn, _tp]], index=["Real: no sobrevivió", "Real: sobrevivió"], columns=["Predicción: no sobrevivió", "Predicción: sobrevivió"])
    mo.vstack([mo.md(f"**Accuracy:** {accuracy_umbral:.1%} · **Precision:** {_precision} · **Recall:** {_recall}"), mo.ui.table(matriz_confusion.reset_index(names="Clase real"), selection=None)])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
