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

    <center> <span style="color:DarkBlue">  Tema 13: Redes neuronales, fundamentos de pytorch </span>  </center>
    <center> <span style="color:Blue"> M. en C. Iván A. Toledano Juárez </span>  </center>

    # Fundamentos de PyTorch

    Basado en notas de **mrdbourke**. Ejemplo es corrido en Google Colab, utilizando GPU y TPU's.

    PyTorch es una de las bibliotecas más populares y poderosas para el desarrollo de modelos de **aprendizaje automático** y especialmente de **aprendizaje profundo (deep learning)**. Desarrollada por **Facebook AI Research (FAIR)** y lanzada en 2016, PyTorch se ha convertido en una herramienta esencial tanto en la investigación como en aplicaciones industriales, gracias a su flexibilidad, facilidad de uso y compatibilidad con aceleradores como **GPU** y **TPU**.

    En este notebook aprenderás los fundamentos prácticos de PyTorch, centrados en la manipulación de su estructura de datos central: el **tensor**. Un tensor es una generalización de matrices y vectores, capaz de representar datos en múltiples dimensiones, y es el bloque fundamental sobre el cual se construyen todos los modelos en PyTorch.

    ---

    ## ¿Qué vas a aprender?

    Al finalizar esta sección, podrás:

    - Comprender qué es un **tensor** y cómo se diferencia de arrays tradicionales como los de NumPy.
    - Crear y manipular tensores en CPU y GPU.
    - Realizar operaciones matemáticas básicas con PyTorch.
    - Cambiar la forma, tipo de datos y dispositivo de los tensores.
    - Explorar las primeras nociones de **diferenciación automática**, esenciales para entrenamiento de redes neuronales.

    ## Referencias

    - Curso base: [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)
    - Documentación oficial: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)
    - Tutoriales oficiales: [PyTorch Tutorials](https://pytorch.org/tutorials/)
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import copy
    import io
    BASE_DIR = mo.notebook_dir()
    return BASE_DIR, mo


@app.cell
def _():
    # Importación de librerías

    # Estandar
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    # PyTorch
    import torch
    print('Version de PyTorch =',torch.__version__)

    #!nvidia-smi
    return np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tensores [(Documentación)](https://pytorch.org/docs/stable/tensors.html)

    En PyTorch, los **tensores** (`torch.Tensor`) son la **estructura de datos fundamental**, y constituyen el bloque básico sobre el que se construyen todos los modelos de machine learning y deep learning.

    Un **tensor** es una **generalización de los vectores y matrices** a un número arbitrario de dimensiones:

    | Tipo de dato | Ejemplo | Dimensión | Descripción |
    |--------------|---------|-----------|-------------|
    | Escalar      | `5`     | 0D        | Un solo número |
    | Vector       | `[1, 2, 3]` | 1D    | Lista de números |
    | Matriz       | `[[1, 2], [3, 4]]` | 2D | Tabla de números |
    | Tensor       | `torch.rand(3, 3, 3)` | 3D o más | Cubo o arreglo multidimensional |

    Los tensores son **similares a los arrays de NumPy**, pero con capacidades adicionales como:

    - Computación acelerada en **GPU**.
    - Soporte automático de **gradientes** (para backpropagation).
    - Compatibilidad con modelos de deep learning y grafos computacionales.

    ---
    """)
    return


@app.cell
def _(torch):
    # Crear tensor 

    # Escalar
    scalar = torch.tensor(7)
    print(scalar)
    print(type(scalar))
    print("\n")

    # Vector
    vector = torch.tensor([7,7])
    print(vector)
    print(type(vector))
    print("\n")

    # Matriz
    matrix = torch.tensor([[7,8],
                          [9,10]])
    print(matrix)
    print(type(matrix))
    print("\n")

    # Tensor (más dimensiones)
    tensor = torch.tensor([[[1,2,3],
                            [3,4,5],
                            [7,8,9]],
                           [[1,2,3],
                            [4,5,6],
                            [4,5,5]]])
    print(tensor)
    print(type(tensor))
    return matrix, scalar, tensor, vector


@app.cell
def _(matrix, scalar, tensor, vector):
    # Atributos

    # Dimension
    print('Rank (escalar) = ', scalar.ndim)
    print('Rank (vector) = ', vector.ndim)
    print('Rank (Matrix) = ', matrix.ndim)
    print('Rank () = ', tensor.ndim)

    # Elementos como valor numérico
    print(scalar.item())

    # Shape
    print(scalar.shape)
    print(vector.shape)
    print(matrix.shape)
    print(tensor.shape) # Checar número de brackets en el tensor
    return


@app.cell
def _(matrix):
    matrix[0]
    return


@app.cell
def _(matrix):
    matrix[1]
    return


@app.cell
def _(vector):
    vector[0]
    return


@app.cell
def _(vector):
    vector[1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Tensores aleatorios
    Por la forma en la cual muchas redes neuronales empiezan, los tensores llenos de números aleatorios son importantes. Después éstos números aleatorios son actualizados para representar mejor a los datos.

    ---

    ### Funciones principales

    PyTorch ofrece varias funciones para generar tensores con contenido aleatorio:

    | Función | Descripción |
    |--------|-------------|
    | `torch.rand(size)` | Valores aleatorios uniformes en $[0, 1)$ |
    | `torch.randn(size)` | Valores aleatorios de una distribución **normal estándar** (media 0, varianza 1) |
    | `torch.randint(low, high, size)` | Valores enteros aleatorios en el intervalo $[low, high)$ |
    | `torch.rand_like(tensor)` | Tensor aleatorio con la **misma forma** que otro tensor dado |

    ---
    """)
    return


@app.cell
def _(torch):
    # Definimos el tamaño del tensor
    size = (3,4)
    #size = (10,10,10)

    random_tensor = torch.rand(size)

    print(random_tensor)
    print(f"shape: {random_tensor.shape}")
    print(f"ndim: {random_tensor.ndim}")
    return


@app.cell
def _(torch):
    # Tensores aleatorios con shape similar al de un tensor de imagen
    size_2 = (224,224,3) # información (height, width, color channels(r,g,b))

    random_image_size_tensor = torch.rand(size=size_2)
    random_image_size_tensor
    return


@app.cell
def _(BASE_DIR, mo):
    mo.image(BASE_DIR / "imagenes/pytorch_rgb.png")
    return


@app.cell
def _(torch):
    # Tensores con 1's o 0's
    # Tensor de ceros puede servir como un mask en una imagen

    size_3 = (3,4)

    zeros = torch.zeros(size=size_3)
    print(zeros)

    ones = torch.ones(size=size_3)
    print(ones)
    return (ones,)


@app.cell
def _(ones):
    # Tipos de objetos dentro de los tensores
    ones.dtype
    return


@app.cell
def _(torch):
    torch.arange(0, 12, dtype=torch.int8) # arange no incluye el límite superior
    return


@app.cell
def _(torch):
    tensor_range = torch.arange(1, 12, 2, dtype=torch.float32)
    tensor_range
    return (tensor_range,)


@app.cell
def _(tensor_range, torch):
    # Crear un tensor de 0 y 1, con el mismo shape de otro tensor anterior

    new_tensor = torch.zeros_like(input=tensor_range)
    new_tensor
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tipo de objetos dentro de un tensor

    Cada tensor en PyTorch contiene **elementos de un tipo de dato específico**, como enteros, flotantes o booleanos. Este tipo se representa con el atributo `.dtype` del tensor. Para datos flotantes, el dtype por default es ``float32``; si damos enteros a `torch.tensor`, normalmente se infiere `int64`.

    Elegir el tipo adecuado es importante porque:

    - Afecta el **uso de memoria**.
    - Determina la **precisión numérica**.
    - Algunas operaciones solo están disponibles para ciertos tipos.

    ---

    #### Tipos comunes de datos (`torch.dtype`)

    | Tipo de dato | Descripción | Equivalente NumPy |
    |--------------|-------------|-------------------|
    | `torch.float32` | Número flotante de 32 bits (por defecto) | `np.float32` |
    | `torch.float64` | Número flotante de 64 bits (doble precisión) | `np.float64` |
    | `torch.int32`   | Entero de 32 bits | `np.int32` |
    | `torch.int64`   | Entero de 64 bits (común para índices) | `np.int64` |
    | `torch.bool`    | Booleano (True/False) | `np.bool_` |
    """)
    return


@app.cell
def _(torch):
    float32_tensor = torch.tensor([3.0,6.0,9.0],
                                  dtype=None, # dtype
                                  device="mps", # GPU,CPU,TPU
                                  requires_grad=False) # Si es True, las operaciones hechas sobre el tensor son registradas
    print(float32_tensor.shape, float32_tensor.dtype, float32_tensor.device)
    return


@app.cell
def _(torch):
    torch.backends.mps
    return


@app.cell
def _(torch):
    float16_tensor = torch.tensor([3.0,6.0,9.0],
                                  dtype=torch.float16, # dtype
                                  device=None, # GPU,CPU,TPU
                                  requires_grad=False) # Si es True, las operaciones hechas sobre el tensor son registradas
    print(float16_tensor.shape, float16_tensor.dtype, float16_tensor.device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Operaciones sobre tensores
    """)
    return


@app.cell
def _(torch):
    tensor_2 = torch.tensor([1,2,3])
    tensor_2
    return (tensor_2,)


@app.cell
def _(tensor_2):
    tensor_2+10
    return


@app.cell
def _(tensor_2):
    tensor_2*10
    return


@app.cell
def _(tensor_2):
    tensor_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para guardar las operaciones sobre los tensores, tenemos que reasignarlos en una variable.
    """)
    return


@app.cell
def _(torch):
    tensor_3 = torch.tensor([1,2,3])
    tensor_3 = tensor_3+10
    tensor_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multiplicación de matrices en PyTorch

    La **multiplicación de matrices** es una operación fundamental tanto en **álgebra lineal** como en **machine learning (ML)** y **deep learning (DL)**. Por ejemplo, en una red neuronal, las entradas son multiplicadas por una matriz de pesos en cada capa. Esta operación se repite millones de veces durante el entrenamiento y la inferencia.

    ---

    ### ¿Qué es la multiplicación de matrices?

    Dado un par de matrices:

    - $A \in \mathbb{R}^{m \times n}$
    - $B \in \mathbb{R}^{n \times p}$

    Entonces su producto $C = A \times B$ está definido **siempre que las dimensiones internas coincidan** (en este caso, $n$), y el resultado es una matriz de tamaño $m \times p$.

    \begin{equation}
    \text{Si } A: (m \times n) \text{ y } B: (n \times p), \quad \Rightarrow \quad A @ B: (m \times p)
    \end{equation}

    Por ejemplo:

    - $A$ tiene dimensión `(3 × 4)`
    - $B$ tiene dimensión `(4 × 5)`
    - Entonces: `A @ B` tendrá dimensión `(3 × 5)`
    """)
    return


@app.cell
def _(torch):
    tensor_4 = torch.tensor([1, 2, 3])
    tensor_4.shape
    return (tensor_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Una confusión común en álgebra lineal y en el uso de PyTorch es la diferencia entre:

    - **Producto punto (dot product)**: también llamado producto escalar. Da como resultado un número (escalar).
    - **Producto elemento a elemento (Hadamard product)**: opera posición por posición. El resultado es un nuevo vector (o tensor) del mismo tamaño.

    Aqui tenemos el vector (1,2,3), al cual podemos multiplicarle su propia expresión de dos formas:

    (1,2,3) * (1,2,3) = (1$\times$1) + (2$\times$2) + (3$\times$3) # producto punto / multiplicación de matrices

    (1,2,3) * (1,2,3) = (1$\times$1,2$\times$2,3$\times$3) # elemento por elemento
    """)
    return


@app.cell
def _(tensor_4):
    tensor_4 * tensor_4
    return


@app.cell
def _(tensor_4, torch):
    torch.matmul(tensor_4,tensor_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El método ``torch.matmul()``es más eficiente y veloz para este tipo de operaciones, así que es el recomendado a utilizar.
    """)
    return


@app.cell
def _(np):
    array_1 = np.array([[1,2],[2,2]])
    array_2 = np.array([[3,2],[4,5]])
    np.matmul(array_1,array_2)
    #array_1*array_2 # producto de hadamard
    #array_1 @ array_2 # multiplicacion de matriz
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Errores comunes en Deep Learning: `shape`, `dtype`, `device`

    Por lo mismo que hay mucha manipulación de tensores en DL, muchos de los errores técnicos vienen ligados a problemas de shape,dtype,device.

    - Formas incompatibles (`shape`)
    - Tipos de datos distintos (`dtype`)
    - Tensores en diferentes dispositivos (`device`)

    ---
    """)
    return


@app.cell
def _(torch):
    tensor_A = torch.tensor([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=torch.float32)

    tensor_B = torch.tensor([[7, 10],
                             [8, 11], 
                             [9, 12]], dtype=torch.float32)

    try:
        torch.matmul(tensor_A, tensor_B)
    except RuntimeError as _error:
        print("Error esperado: las dimensiones internas no coinciden.")
        print(_error)
    return tensor_A, tensor_B


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Si fuera el problema, aquí tendríamos que hacer que las dimensiones internas coincidan.
    """)
    return


@app.cell
def _(tensor_A, tensor_B):
    # Utilizamos la transpuesta de este vector
    print(tensor_A)
    print(tensor_B.T)
    return


@app.cell
def _(tensor_A, tensor_B, torch):
    torch.matmul(tensor_A, tensor_B.T)
    return


@app.cell
def _(tensor_A, tensor_B, torch):
    # Una sintaxis de matmul más común es mm

    torch.mm(tensor_A, tensor_B.T)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Capas lineales en redes neuronales: `torch.nn.Linear`

    Recordando lo que ya vimos: **dentro de cada neurona** en una red completamente conectada (fully connected), ocurre una operación muy parecida a una **regresión lineal**. El cálculo de salida (output) de una neurona se puede expresar como:

    \begin{equation}
    y = x \cdot W^\top + b
    \end{equation}

    Donde:

    - $x$: vector de entrada (input)
    - $W$: matriz de pesos (weight matrix)
    - $b$: vector de bias o sesgo
    - $y$: vector de salida (output)

    Esta operación se aplica en **todas las capas densas (fully-connected)** de una red neuronal, y se conoce como **transformación lineal afín** (lineal + desplazamiento).

    ---

    ### 🧱 Módulo `torch.nn.Linear`

    PyTorch implementa esta operación con el módulo `torch.nn.Linear`:
    """)
    return


@app.cell
def _(torch):
    # Semilla aleatoria
    torch.manual_seed(88)

    # Neural Network (NN)
    linear = torch.nn.Linear(in_features=2, # coincide con las dimensiones interiores del input
                             out_features=6) # valor de salida

    tensor_A_2 = torch.tensor([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=torch.float32)
    x = tensor_A_2 # tensor de input
    output = linear(x) # output

    print(f"Input shape: {x.shape}\n")
    print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Análisis de dimensiones
    * `x`: tensor de entrada de forma `(3,2)`
    * `linear.weight`: matriz de pesos de forma `(6,2)` (6 neuronas $\times$ 2 pesos)
    * `linear.bias`: vector de bias de forma `(6,)`
    * Resultado final: `output` de forma `(3,6)`
    """)
    return


@app.cell
def _(torch):
    try:
        # Semilla aleatoria
        torch.manual_seed(88)

        # Neural Network (NN)
        linear_2 = torch.nn.Linear(in_features=1, # no coincide con las dimensiones interiores del input
                                 out_features=6) # valor de salida

        tensor_A_3 = torch.tensor([[1, 2],
                                 [3, 4],
                                 [5, 6]], dtype=torch.float32)
        x_2 = tensor_A_3 # tensor de input
        output_2 = linear_2(x_2) # output

        print(f"Input shape: {x_2.shape}\n")
        print(f"Output:\n{output_2}\n\nOutput shape: {output_2.shape}")
    except RuntimeError as _error:
        print("Error esperado: in_features debe coincidir con las columnas del input.")
        print(_error)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Operaciones estadísticas básicas en tensores

    PyTorch ofrece varias funciones para calcular **estadísticas elementales** sobre tensores. Estas funciones son muy útiles para:

    - Inspeccionar datos
    - Calcular métricas
    - Hacer reducciones (reducing dimensions)

    | Función | Descripción |
    |--------|-------------|
    | `torch.sum(t)` | Suma total de los elementos |
    | `torch.mean(t)` | Promedio |
    | `torch.min(t)` / `torch.max(t)` | Mínimo / máximo global |
    | `torch.argmin(t)` / `torch.argmax(t)` | Índice del valor mínimo / máximo |
    | `torch.std(t)` | Desviación estándar |
    | `torch.var(t)` | Varianza |

    ---
    """)
    return


@app.cell
def _(torch):
    x_3 = torch.arange(0, 100, 10)
    x_3
    return (x_3,)


@app.cell
def _(torch, x_3):
    print(f"Min: {x_3.min()}")
    print(f"Max: {x_3.max()}")
    print(f"Mean: {x_3.type(torch.float32).mean()}") # la media no funciona sin datatype tipo float, se tiene que espeficicar
    print(f"Sum: {x_3.sum()}")
    return


@app.cell
def _(torch, x_3):
    # O directamente con métodos Torch

    torch.max(x_3), torch.min(x_3), torch.mean(x_3.type(torch.float32)), torch.sum(x_3)
    return


@app.cell
def _(torch):
    # Posición de min y max

    tensor_5 = torch.arange(10, 100, 10)
    print(f"Tensor: {tensor_5}")

    # Regresa el índice donde cae el min/max
    print(f"Indice donde ocurre max: {tensor_5.argmax()}")
    print(f"Indice donde ocurre min: {tensor_5.argmin()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cambio de dtype

    Cambiar el tipo de un tensor es útil para:

    - Aumentar o reducir precisión (por ejemplo, `float64` → `float32`)
    - Preparar datos para operaciones matemáticas compatibles
    - Ahorrar memoria (por ejemplo, usar `float16` en GPU)
    - Convertir tensores booleanos o enteros a flotantes (o viceversa)
    """)
    return


@app.cell
def _(torch):
    tensor_6 = torch.arange(10., 100., 10.)
    tensor_6.dtype
    return (tensor_6,)


@app.cell
def _(tensor_6, torch):
    tensor_float16 = tensor_6.type(torch.float16)
    tensor_float16
    return


@app.cell
def _(tensor_6, torch):
    tensor_int8 = tensor_6.type(torch.int8)
    tensor_int8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Otros métodos

    - **`reshape()`**
      Cambia la forma del tensor, devolviendo un nuevo tensor con los mismos datos reorganizados. Requiere que el número total de elementos se conserve.

    - **`view()`**
      Similar a `reshape`, pero devuelve una vista del tensor original.

    - **`stack()`**
      Combina una secuencia de tensores (de igual tamaño) en una nueva dimensión. Es útil para agrupar múltiples tensores en un solo tensor de mayor dimensionalidad.

    - **`squeeze()`**
      Elimina todas las dimensiones de tamaño 1 (por ejemplo, convierte `(1, 3, 1, 4)` en `(3, 4)`). Ideal para limpiar tensores innecesariamente inflados.

    - **`unsqueeze()`**
      Agrega una dimensión de tamaño 1 en la posición indicada. Comúnmente usado para añadir dimensión de "batch".

    - **`permute()`**
      Cambia el orden de las dimensiones del tensor según los índices especificados. Es una generalización de la transposición para tensores de más de 2 dimensiones.

    Estas funciones son esenciales para adaptar tensores a las formas requeridas por capas de redes neuronales, funciones de pérdida o procesos de entrenamiento.
    """)
    return


@app.cell
def _(torch):
    x_4 = torch.arange(1., 8.)
    x_4, x_4.shape
    return (x_4,)


@app.cell
def _(x_4):
    # Añadir dimension extra
    x_reshaped = x_4.reshape(1, 7)
    x_reshaped, x_reshaped.shape
    return (x_reshaped,)


@app.cell
def _(x_4):
    x_5 = x_4
    #https://stackoverflow.com/a/54507446/7900723
    x_5 = x_5.clone() # Copia para aislar este ejemplo de las otras celdas
    z = x_5.view(1, 7)
    z, z.shape
    # La modificación se hace aquí para observar la memoria compartida.
    z[:, 0] = 5
    print("Después de cambiar z:", z, x_5)
    return x_5, z


@app.cell
def _(x_5, z):
    # Si cambiamos z también cambiamos x (visto en la celda anterior)
    z, x_5
    return


@app.cell
def _(torch, x_5):
    x_stacked = torch.stack([x_5, x_5, x_5, x_5], dim=0) # intentar a cambiar dim=1
    x_stacked
    return


@app.cell
def _(x_reshaped):
    #squeeze para remover todas las dimensiones 1 de un tensor

    print(f"Tensor anterior: {x_reshaped}")
    print(f"Shape anterior: {x_reshaped.shape}")

    # Remover dimensión extra de este reshape
    x_squeezed = x_reshaped.squeeze()
    print(f"\nNuevo tensor: {x_squeezed}")
    print(f"Nuevo shape: {x_squeezed.shape}")
    return (x_squeezed,)


@app.cell
def _(x_squeezed):
    # Unsqueeze las regresa

    print(f"Tensor anterior: {x_squeezed}")
    print(f"Shape anterior: {x_squeezed.shape}")

    # Añadir dimension extra con este unsqueeze
    x_unsqueezed = x_squeezed.unsqueeze(dim=0) # donde se va a agregar esta dimension
    print(f"\nNuevo tensor: {x_unsqueezed}")
    print(f"Nuevo shape: {x_unsqueezed.shape}")
    return


@app.cell
def _(torch):
    # Permutar las dimensiones del tensor

    # Crear tensor con cierto shape
    x_original = torch.rand(size=(224, 224, 3))

    # Se permuta el tensor original para reasignar el orden de los ejes
    x_permuted = x_original.permute(2, 0, 1) # axis 0->1, 1->2, 2->0

    print(f"Shape anterior: {x_original.shape}")
    print(f"Shape posterior: {x_permuted.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Indexing
    """)
    return


@app.cell
def _(torch):
    x_6 = torch.arange(1, 10).reshape(1, 3, 3)
    x_6, x_6.shape
    return (x_6,)


@app.cell
def _(x_6):
    print(f"First square bracket:\n{x_6[0]}") 
    print(f"Second square bracket: {x_6[0][0]}") 
    print(f"Third square bracket: {x_6[0][0][0]}")
    return


@app.cell
def _(x_6):
    x_6[:, 0] # : para especificar "todos los valores en la dimension" y luego de la comma la otra dimension
    return


@app.cell
def _(x_6):
    x_6[:, :, 1] # tomar todos los valores de la dimension 0 y 1, pero el índice 1 de la dimension 2
    return


@app.cell
def _(x_6):
    x_6[0, 0, :] # equivalente a x[0][0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### PyTorch a Numpy

    PyTorch y NumPy son altamente compatibles. Puedes **convertir fácilmente tensores de PyTorch a arrays de NumPy** (y viceversa), lo cual es muy útil cuando deseas:

    - Usar funciones o librerías de Python que trabajan con NumPy
    - Visualizar o exportar datos
    - Realizar análisis fuera de PyTorch
    """)
    return


@app.cell
def _(np, torch):
    array = np.arange(1.0, 8.0)
    tensor_7 = torch.from_numpy(array)
    array, tensor_7

    # Observamos que se heredan los dtypes de numpy a los dtype de pytorch
    return array, tensor_7


@app.cell
def _(array, tensor_7):
    array_2 = array
    array_2 = array_2 + 1
    array_2, tensor_7
    # Esta reasignación crea otro array; no modifica la memoria del anterior.
    # Una modificación in-place del array original sí se refleja en el tensor.
    return


@app.cell
def _(torch):
    # Tensor to NumPy array
    tensor_8 = torch.ones(7) 
    numpy_tensor = tensor_8.numpy()
    tensor_8, numpy_tensor
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Uso de GPU en PyTorch

    Una de las grandes ventajas de PyTorch es su capacidad de ejecutar operaciones en **GPU**, lo cual puede acelerar enormemente el entrenamiento de modelos de Machine Learning y Deep Learning.

    #### ¿Cómo saber si hay GPU disponible?

    PyTorch detecta automáticamente si tienes acceso a una GPU compatible con CUDA. Los equipos Mac modernos con chip **Apple Silicon** (como M1, M2, M3) no usan CUDA ni tarjetas NVIDIA. En su lugar, PyTorch ofrece soporte para la GPU de Apple mediante una tecnología llamada **MPS (Metal Performance Shaders)**, que permite acelerar modelos en la GPU integrada.
    """)
    return


@app.cell
def _(torch):
    torch.cuda.is_available()
    return


@app.cell
def _(torch):
    torch.mps.is_available()
    return


@app.cell
def _(torch):
    if torch.cuda.is_available():
        device = "cuda" # Use NVIDIA GPU (if available)
    elif torch.backends.mps.is_available():
        device = "mps" # Use Apple Silicon GPU (if available)
    else:
        device = "cpu" # Default to CPU if no GPU is available
    device
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explorando el shape de una multiplicación

    Ya vimos que las dimensiones internas deben coincidir. Ahora podemos cambiarlas y observar qué ocurre.

    **Antes de mover los controles:** si $A$ tiene shape $(2,3)$ y $B$ tiene shape $(3,4)$, ¿cuál será el shape de la salida? ¿Qué pasa si cambiamos únicamente las filas de $B$?
    """)
    return


@app.cell
def _(mo):
    filas_A = mo.ui.slider(1, 5, value=2, label="Filas de A", show_value=True)
    columnas_A = mo.ui.slider(1, 5, value=3, label="Columnas de A", show_value=True)
    filas_B = mo.ui.slider(1, 5, value=3, label="Filas de B", show_value=True)
    columnas_B = mo.ui.slider(1, 5, value=4, label="Columnas de B", show_value=True)
    mo.hstack([filas_A, columnas_A, filas_B, columnas_B])
    return columnas_A, columnas_B, filas_A, filas_B


@app.cell
def _(columnas_A, columnas_B, filas_A, filas_B, torch):
    A_interactiva = torch.arange(1, filas_A.value * columnas_A.value + 1).reshape(filas_A.value, columnas_A.value)
    B_interactiva = torch.arange(1, filas_B.value * columnas_B.value + 1).reshape(filas_B.value, columnas_B.value)
    print("A =", A_interactiva, "\nB =", B_interactiva)
    if columnas_A.value == filas_B.value:
        producto_interactivo = A_interactiva @ B_interactiva
        print("A @ B =", producto_interactivo)
        print("Shape de salida:", producto_interactivo.shape)
    else:
        print("No se puede multiplicar: las columnas de A deben coincidir con las filas de B.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Primer vistazo a autograd

    PyTorch puede registrar operaciones y calcular derivadas automáticamente. Por ejemplo, si $y=x^2$, entonces $dy/dx=2x$.

    Cambiamos el valor de $x$, calculamos `y` y llamamos a `backward()`. La derivada queda guardada en `x.grad`. En el siguiente notebook usaremos esta misma idea para actualizar los pesos del modelo.

    **Checar:** ¿qué signo tiene el gradiente cuando $x$ es negativo? ¿Y cuando vale cero?
    """)
    return


@app.cell
def _(mo):
    valor_x = mo.ui.slider(-5, 5, step=0.5, value=2, label="Valor de x", show_value=True)
    valor_x
    return (valor_x,)


@app.cell
def _(torch, valor_x):
    x_grad = torch.tensor(float(valor_x.value), requires_grad=True)
    y_grad = x_grad ** 2
    y_grad.backward()
    print(f"x = {x_grad.item()}, y = {y_grad.item()}")
    print(f"Gradiente calculado: {x_grad.grad.item()}; derivada 2x: {2 * x_grad.item()}")
    return


if __name__ == "__main__":
    app.run()
