# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.24.0", "matplotlib>=3.8", "numpy>=1.26", "scikit-learn>=1.4"]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from sklearn.datasets import make_moons

    return ListedColormap, make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Red neuronal con una capa oculta

    En el script anterior habíamos observado que un solo perceptrón siempre produce una frontera lineal de decisión (clasificación). En este notebook ahora incluiremos una capa oculta con $n_h$ neuronas, es decir


    \[
    2\;\text{entradas}\quad\longrightarrow\quad
    n_h\;\text{neuronas ocultas}\quad\longrightarrow\quad
    1\;\text{neurona de salida}
    \]

    Cada neurona oculta construye una combinación lineal diferente (o sea, la recta). La función de activación utilizada transforma estas combinaciones y permite que la salida final tenga una frontera de decisión no lineal.
    """)
    return


@app.cell
def _(mo):
    # widgets de marimo

    neuronas = mo.ui.slider(1, 12, value=4, step=1, label="Neuronas ocultas $n_h$")
    activacion = mo.ui.dropdown(
        options={"Tangente hiperbólica": "tanh", "ReLU": "relu"},
        value="Tangente hiperbólica",
        label="Activación oculta",
    )
    ruido = mo.ui.slider(0.02, 0.35, value=0.18, step=0.01, label="Ruido")
    learning_rate = mo.ui.slider(0.01, 1.5, value=0.8, step=0.01, label="Learning rate")
    iteraciones = mo.ui.slider(200, 10000, value=3000, step=200, label="Iteraciones")

    mo.vstack([
        mo.md("### Arquitectura y entrenamiento"),
        mo.hstack([neuronas, activacion]),
        mo.hstack([ruido, learning_rate]),
        iteraciones,
    ])
    return activacion, iteraciones, learning_rate, neuronas, ruido


@app.cell
def _(make_moons, ruido):
    muestras, etiquetas = make_moons(
        n_samples=250,
        noise=ruido.value,
        random_state=42,
    )
    X = muestras.T
    y = etiquetas.reshape(1, -1)
    return X, y


@app.cell(hide_code=True)
def _(mo, neuronas):
    mo.md(rf"""
    ## Propagación hacia adelante (*forward propagation*)

    La primera capa calcula:

    \[
    Z^{{[1]}}=W^{{[1]}}X+b^{{[1]}},\qquad
    A^{{[1]}}=g(Z^{{[1]}})
    \]

    La capa de salida calcula:

    \[
    Z^{{[2]}}=W^{{[2]}}A^{{[1]}}+b^{{[2]}},\qquad
    A^{{[2]}}=\sigma(Z^{{[2]}})
    \]

    Con \(n_h={neuronas.value}\), las dimensiones son:

    | Parámetro o activación | Dimensión |
    |---|---:|
    | \(X\) | \((2,m)\) |
    | \(W^{{[1]}}\) | \(({neuronas.value},2)\) |
    | \(b^{{[1]}}\) | \(({neuronas.value},1)\) |
    | \(A^{{[1]}}\) | \(({neuronas.value},m)\) |
    | \(W^{{[2]}}\) | \((1,{neuronas.value})\) |
    | \(b^{{[2]}}\) | \((1,1)\) |
    | \(A^{{[2]}}\) | \((1,m)\) |
    """)
    return


@app.cell
def _(np):
    # Funciones utilizadas

    def sigmoid(Z):
        """
        Función de activación sigmoide
        """
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))

    def activation(Z, nombre):
        """
        Aplica la activación elegida en la capa oculta.
        """

        if nombre == "tanh":
            return np.tanh(Z) 
        else:
            return np.maximum(0,Z) # ReLU

    def activation_derivative(Z, A, nombre):
        """
        Calcula la derivada de la activación oculta.
        """

        if nombre == "tanh":
            return 1 - A**2
        else:
            return (Z > 0).astype(float)


    def initialize_parameters(n_x, n_h, n_y=1, nombre="tanh", semilla=42):
        """
        Inicializa los parámetros del modelo.

        Para tanh utilizamos una escala tipo Xavier; para ReLU, una escala tipo He.
        """
        generador = np.random.default_rng(semilla)
    
        escala_1 = np.sqrt((2 if nombre == "relu" else 1) / n_x)

        W1 = generador.normal(0, escala_1, (n_h, n_x))
        b1 = np.zeros((n_h, 1))
        W2 = generador.normal(0, np.sqrt(1 / n_h), (n_y, n_h))
        b2 = np.zeros((n_y, 1))
    
        parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

        return parameters

    def forward_propagation(X, parameters, nombre):
        """
        Implementa la propagación hacia adelante de las dos capas.
        """

        # parametros, pesos y sesgos
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # capas y sus transformaciones
        Z1 = np.matmul(W1,X) + b1
        A1 = activation(Z1, nombre)
        Z2 = np.matmul(W2,A1) + b2
        A2 = sigmoid(Z2) # salida

        # cache
        cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2":Z2,
            "A2":A2
        }
    
        return A2, cache

    def loss(A2, y):
        """
        Función de pérdida (entropía cruzada binaria)
        """
        A2_segura = np.clip(A2, 1e-10, 1 - 1e-10)

        loss = -np.mean(y * np.log(A2_segura) + (1 - y) * np.log(1 - A2_segura))
    
        return loss

    def backward_propagation(X, y, parameters, cache, nombre):
        """
        Implementa backpropagation aplicando la regla de la cadena.
        """
    
        m = X.shape[1]
        A1, A2, Z1 = cache["A1"], cache["A2"], cache["Z1"]

        dZ2 = A2 - y
        dW2 = np.matmul(dZ2,A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.matmul(parameters["W2"].T, dZ2)
        dZ1 = dA1 * activation_derivative(Z1, A1, nombre)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {
            "dW1":dW1,
            "db1":db1,
            "dW2":dW2,
            "db2":db2
        }
    
        return grads

    def update_parameters(parameters, grads, learning_rate):
        """
        Actualiza los parámetros utilizando descenso del gradiente.
        """
    
        for indice in (1, 2):
            parameters[f"W{indice}"] -= learning_rate * grads[f"dW{indice}"]
            parameters[f"b{indice}"] -= learning_rate * grads[f"db{indice}"]
        
        return parameters

    def nn_model(X, y, n_h=4, nombre="tanh", iterations=3000, learning_rate=0.8):
        """
        Construye y entrena una red neuronal con una capa oculta.
        """
        parameters = initialize_parameters(n_x=X.shape[0],
                                           n_h=n_h,
                                           n_y=y.shape[0],
                                           nombre=nombre)
        costs = []

        # ciclo de entrenamiento
        for i in range(iterations):
            A2, cache = forward_propagation(X=X,
                                            parameters=parameters,
                                            nombre=nombre)
            cost = loss(A2=A2,
                        y=y)
            grads = backward_propagation(X=X,
                                         y=y,
                                         parameters=parameters,
                                         cache=cache, 
                                         nombre=nombre)
            parameters = update_parameters(parameters=parameters,
                                           grads=grads, 
                                           learning_rate=learning_rate)
            if i % 20 == 0:
                costs.append((i, cost))

        return parameters, costs

    return forward_propagation, nn_model


@app.cell
def _(
    X,
    activacion,
    forward_propagation,
    iteraciones,
    learning_rate,
    neuronas,
    nn_model,
    np,
    y,
):
    # Se corre el entrenamiento de la red neuronal

    parameters, costs = nn_model(
        X=X,
        y=y,
        n_h=neuronas.value,
        nombre=activacion.value,
        iterations=iteraciones.value,
        learning_rate=learning_rate.value,
    )
    A2, cache = forward_propagation(X=X, 
                                    parameters=parameters, 
                                    nombre=activacion.value)
    accuracy = float(np.mean((A2 > 0.5) == y))
    return accuracy, cache, costs, parameters


@app.cell
def _(
    ListedColormap,
    X,
    activacion,
    costs,
    forward_propagation,
    np,
    parameters,
    plt,
    y,
):
    eje_x1 = np.linspace(X[0].min() - 0.5, X[0].max() + 0.5, 220)
    eje_x2 = np.linspace(X[1].min() - 0.5, X[1].max() + 0.5, 220)
    malla_x1, malla_x2 = np.meshgrid(eje_x1, eje_x2)
    X_malla = np.vstack((malla_x1.ravel(), malla_x2.ravel()))
    A_malla, _cache_malla = forward_propagation(
        X_malla, parameters, activacion.value
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].contourf(
        malla_x1, malla_x2,
        A_malla.reshape(malla_x1.shape),
        levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.6,
    )
    axes[0].contour(
        malla_x1, malla_x2,
        A_malla.reshape(malla_x1.shape),
        levels=[0.5], colors="black", linewidths=2,
    )
    axes[0].scatter(
        X[0], X[1], c=y.ravel(),
        cmap=ListedColormap(["blue", "red"]), edgecolor="white", s=32,
    )
    axes[0].set(title="Frontera no lineal", xlabel="$x_1$", ylabel="$x_2$")

    pasos, valores = zip(*costs)
    axes[1].plot(pasos, valores, color="purple")
    axes[1].set(title="Función de costo", xlabel="Iteración", ylabel="Costo")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(accuracy, activacion, cache, mo, parameters):
    mo.callout(
        mo.md(
            f"""
            **Exactitud: {accuracy:.1%}**

            La activación oculta utilizada es `{activacion.value}`. Observa que
            ahora `A1.shape = {cache['A1'].shape}`: cada fila contiene la salida
            de una neurona oculta para todas las observaciones.

            Los parámetros aprendidos tienen formas
            `W1={parameters['W1'].shape}` y `W2={parameters['W2'].shape}`.
            """
        ),
        kind="success" if accuracy >= 0.85 else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Por qué necesitamos una función de activación?

    Si reemplazamos \(A^{[1]}=g(Z^{[1]})\) por \(A^{[1]}=Z^{[1]}\), entonces:

    \[
    Z^{[2]}=W^{[2]}W^{[1]}X+W^{[2]}b^{[1]}+b^{[2]}
    \]

    Podemos agrupar los términos anteriores en una nueva matriz \(W'\) y un nuevo
    sesgo \(b'\). Es decir, varias capas lineales siguen siendo equivalentes a una
    sola capa lineal. La activación es la que permite construir una representación
    no lineal.

    ### Preguntas

    1. Entrena la red con una sola neurona oculta y después con cuatro.
    2. Compara `tanh` y `ReLU` manteniendo los demás controles constantes.
    3. Identifica qué configuraciones corresponden a falta de capacidad y cuáles a
       dificultades de optimización.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
