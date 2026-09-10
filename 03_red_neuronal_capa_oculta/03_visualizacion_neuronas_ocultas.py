# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.24.0", "matplotlib>=3.8", "numpy>=1.26", "scikit-learn>=1.4"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_moons

    return make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ¿Qué aprende cada neurona oculta?

    En el tema anterior, se observó la frontera producida por toda la red. Sin embargo, la capa oculta contiene varias neuronas y cada una aprende parámetros diferentes.

    En el notebook anterior observamos la frontera producida por toda la red. Sin
    embargo, la capa oculta contiene varias neuronas y cada una aprende parámetros
    diferentes:

    \[
    z_j^{[1]}=w_{j1}^{[1]}x_1+w_{j2}^{[1]}x_2+b_j^{[1]}
    \]

    donde \(z_j^{[1]}=0\) describe una recta. La activación indica en qué lado de esta recta se encuentra dicha observación.
    """)
    return


@app.cell
def _(mo):
    neuronas = mo.ui.slider(2, 8, value=4, step=1, label="Número de neuronas")
    ruido = mo.ui.slider(0.02, 0.30, value=0.14, step=0.01, label="Ruido")
    iteraciones = mo.ui.slider(500, 8000, value=4000, step=250, label="Iteraciones")
    learning_rate = mo.ui.slider(0.05, 1.5, value=0.8, step=0.05, label="Learning rate")

    mo.vstack([
        mo.md("### Controles del modelo"),
        mo.hstack([neuronas, ruido]),
        mo.hstack([iteraciones, learning_rate]),
    ])
    return iteraciones, learning_rate, neuronas, ruido


@app.cell
def _(make_moons, ruido):
    # Se generan datos ficticios
    muestras, etiquetas = make_moons(
        n_samples=250, noise=ruido.value, random_state=12
    )
    X = muestras.T
    Y = etiquetas.reshape(1, -1)
    return X, Y


@app.cell
def _(np):
    def sigmoid(Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    def initialize_parameters(n_x, n_h, semilla=5):
        generador = np.random.default_rng(semilla)

        W1 = generador.normal(0, np.sqrt(1 / n_x), (n_h, n_x))
        b1 = np.zeros((n_h, 1))
        W2 = generador.normal(0, np.sqrt(1 / n_h), (1, n_h))
        b2 = np.zeros((1, 1))

        parameters = {
            "W1":W1,
            "b1":b1,
            "W2":W2,
            "b2":b2
        }
    
        return parameters

    def forward_propagation(X, parameters):

        # parametros, pesos y sesgos
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.matmul(W1,X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.matmul(W2,A1) + b2
        A2 = sigmoid(Z2) # salida

        cache = {
            "Z1":Z1,
            "A1":A1,
            "Z2":Z2,
            "A2":A2
        }
    
        return A2, cache

    def nn_model(X, Y, n_h, num_iterations, learning_rate):
        """Entrena la red de dos capas utilizando descenso del gradiente."""
        m = X.shape[1]
        parameters = initialize_parameters(X.shape[0], n_h)
        costs = []

        for i in range(num_iterations):
            A2, cache = forward_propagation(X, parameters)
            A2_segura = np.clip(A2, 1e-10, 1 - 1e-10)
            cost = -np.mean(Y * np.log(A2_segura) + (1 - Y) * np.log(1 - A2_segura))

            dZ2 = A2 - Y
            dW2 = np.matmul(dZ2,cache["A1"].T) / m
            db2 = np.sum(dZ2, axis=1, keepdims=True) / m
            dA1 = np.matmul(parameters["W2"].T,dZ2)
            dZ1 = dA1 * (1 - cache["A1"] ** 2)
            dW1 = np.matmul(dZ1, X.T) / m
            db1 = np.sum(dZ1, axis=1, keepdims=True) / m

            parameters["W1"] -= learning_rate * dW1
            parameters["b1"] -= learning_rate * db1
            parameters["W2"] -= learning_rate * dW2
            parameters["b2"] -= learning_rate * db2

            if i % 20 == 0:
                costs.append((i, cost))

        return parameters, costs

    return forward_propagation, nn_model


@app.cell
def _(
    X,
    Y,
    forward_propagation,
    iteraciones,
    learning_rate,
    neuronas,
    nn_model,
    np,
):
    parameters, costs = nn_model(
        X, Y,
        n_h=neuronas.value,
        num_iterations=iteraciones.value,
        learning_rate=learning_rate.value,
    )
    A2, cache = forward_propagation(X, parameters)
    accuracy = float(np.mean((A2 > 0.5) == Y))
    return accuracy, cache, parameters


@app.cell
def _(mo, neuronas):
    neurona = mo.ui.slider(
        1, neuronas.value, value=1, step=1,
        label="Neurona oculta que queremos inspeccionar",
    )
    neurona
    return (neurona,)


@app.cell(hide_code=True)
def _(mo, neurona, parameters):
    indice = neurona.value - 1
    pesos_neurona = parameters["W1"][indice]
    bias_neurona = parameters["b1"][indice, 0]
    peso_salida = parameters["W2"][0, indice]
    mo.md(
        rf"""
        ## Neurona oculta {neurona.value}

        La combinación lineal aprendida por esta neurona es:

        \[
        z_{{{neurona.value}}}^{{[1]}}
        =({pesos_neurona[0]:.3f})x_1
        +({pesos_neurona[1]:.3f})x_2
        +({bias_neurona:.3f})
        \]

        Su peso en la neurona de salida es
        \(w_{{1,{neurona.value}}}^{{[2]}}={peso_salida:.3f}\).
        El signo y la magnitud de este valor indican cómo utiliza la salida final la
        característica construida por esta neurona.

        La combinación final utiliza las activaciones de **todas** las neuronas:

        \[
        z^{{[2]}}=\sum_j w_{{1,j}}^{{[2]}}\tanh(z_j^{{[1]}})+b^{{[2]}},
        \qquad \hat y=\sigma(z^{{[2]}}).
        \]

        La frontera negra corresponde a \(\hat y=0.5\), equivalente a
        \(z^{{[2]}}=0\). El selector cambia la neurona inspeccionada;
        la combinación final siempre utiliza toda la capa oculta.
        """
    )
    return (indice,)


@app.cell
def _(X, Y, forward_propagation, indice, np, parameters, plt):
    eje_x1 = np.linspace(X[0].min() - 0.6, X[0].max() + 0.6, 220)
    eje_x2 = np.linspace(X[1].min() - 0.6, X[1].max() + 0.6, 220)
    malla_x1, malla_x2 = np.meshgrid(eje_x1, eje_x2)
    X_malla = np.vstack((malla_x1.ravel(), malla_x2.ravel()))
    A2_malla, cache_malla = forward_propagation(X_malla, parameters)
    activacion_malla = cache_malla["A1"][indice].reshape(malla_x1.shape)
    Z_neurona = cache_malla["Z1"][indice].reshape(malla_x1.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    mapa_activacion = axes[0].contourf(
        malla_x1, malla_x2, activacion_malla,
        levels=np.linspace(-1, 1, 21), cmap="coolwarm",
    )
    axes[0].contour(
        malla_x1, malla_x2, Z_neurona,
        levels=[0], colors="black", linewidths=2,
    )
    axes[0].scatter(X[0], X[1], c=Y.ravel(), cmap="bwr", s=15, edgecolor="white")
    axes[0].set(title="Activación de la neurona", xlabel="$x_1$", ylabel="$x_2$")
    fig.colorbar(mapa_activacion, ax=axes[0], label="$a_j^{[1]}$")

    for indice_linea in range(parameters["W1"].shape[0]):
        Z_linea = cache_malla["Z1"][indice_linea].reshape(malla_x1.shape)
        axes[1].contour(
            malla_x1, malla_x2, Z_linea, levels=[0],
            linewidths=3 if indice_linea == indice else 1.3,
        )
    axes[1].scatter(X[0], X[1], c=Y.ravel(), cmap="bwr", s=20, edgecolor="white")
    axes[1].set(title="Rectas de todas las neuronas", xlabel="$x_1$", ylabel="$x_2$")

    axes[2].contourf(
        malla_x1, malla_x2, A2_malla.reshape(malla_x1.shape),
        levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.65,
    )
    axes[2].contour(
        malla_x1, malla_x2, A2_malla.reshape(malla_x1.shape),
        levels=[0.5], colors="black", linewidths=2,
    )
    axes[2].scatter(X[0], X[1], c=Y.ravel(), cmap="bwr", s=20, edgecolor="white")
    axes[2].set(title="Combinación final", xlabel="$x_1$", ylabel="$x_2$")
    fig.tight_layout()
    fig
    return


@app.cell
def _(cache, neurona, plt):
    fig_hist, ax_hist = plt.subplots(figsize=(7.5, 3.5))
    valores_neurona = cache["A1"][neurona.value - 1]
    ax_hist.hist(valores_neurona, bins=25, color="mediumpurple", edgecolor="white")
    ax_hist.set(
        title=f"Distribución de activaciones: neurona {neurona.value}",
        xlabel="$a_j^{[1]}$", ylabel="Número de observaciones",
    )
    ax_hist.grid(axis="y", alpha=0.2)
    fig_hist.tight_layout()
    fig_hist
    return


@app.cell(hide_code=True)
def _(accuracy, mo):
    mo.callout(
        mo.md(
            f"""
            **Exactitud de la red: {accuracy:.1%}**

            La neuronas de la capa oculta no deberían interpretarse como clasificadores independientes.
            Cada una contruye una característica intermedia (o virtual).
            La neurona de salida es la que aprende a combinar todas estas características creadas.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preguntas

    1. Recorre las neuronas con el control y compara sus rectas y activaciones.
    2. Identifica neuronas con representaciones parecidas. ¿Son todas necesarias?
    3. Aumenta el número de neuronas. ¿La frontera siempre mejora?
    4. Observa el peso de salida de cada neurona. ¿Qué significa un valor negativo?
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
