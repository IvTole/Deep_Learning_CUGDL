import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Actividad 1 · Entrenamiento paso a paso de una neurona artificial

    **Deep Learning para Computer Vision**

    **Nombre:**

    ## Objetivo
    Implementar y explicar el entrenamiento de una neurona para regresión con
    NumPy, desde una predicción inicial hasta la actualización de sus parámetros.
    Al finalizar, deberás relacionar las ecuaciones, las dimensiones de los
    arreglos y el comportamiento de la pérdida durante el aprendizaje.

    Usaremos una activación identidad: este modelo corresponde a una regresión
    lineal. En clase lo abordamos como una primera aproximación al perceptrón.

    ## Instrucciones y entrega
    - Completa las celdas con `TODO` y las respuestas marcadas con **Tu respuesta**.
    - Implementa el modelo con NumPy. No utilices modelos de scikit-learn,
      TensorFlow, PyTorch ni herramientas de diferenciación automática.
    - En cada paso incluye explicación, ecuación, código e interpretación.
    - Conserva la convención de ejemplos en las **columnas** de `X`.
    - Entrega este archivo `.py` de marimo con tus respuestas, código y gráficas.
      Debe ejecutarse completo sin errores y permitir reproducir los resultados.
    - Los bloques incompletos devuelven `None` para que puedas abrir la plantilla.
      Sustitúyelos por tu implementación; no dejes resultados escritos a mano
      como reemplazo del código.

    Esta actividad estudia la mecánica del entrenamiento. Los datos de trabajo
    se usan para entrenar; no se evaluará generalización a datos nuevos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Problema y datos

    Predeciremos un **puntaje sintético de rendimiento** a partir de dos
    características numéricas ya preparadas. Estos datos son didácticos y no
    representan mediciones reales. No requieren limpieza ni estandarización.

    1. Identifica las entradas y la salida del modelo.
    2. ¿Por qué este problema es de regresión?
    3. ¿Qué significa un ejemplo de entrenamiento en esta tabla?

    **Respuesta:** …
    """)
    return


@app.cell
def _(np):
    X_datos = np.array([
        [-1.0, -0.5, 0.0, 0.5, 1.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, -1.0, 0.5, -0.5, -1.0, 1.0, 1.0],
    ], dtype=float)
    y_datos = np.array([
        [-0.7, -0.9, 1.3, 1.0, 2.9, 0.6, -0.4, 1.2]
    ], dtype=float)
    return X_datos, y_datos


@app.cell
def _(X_datos, mo, y_datos):
    mo.ui.table([
        {"Ejemplo": i + 1, "Característica 1": float(X_datos[0, i]),
         "Característica 2": float(X_datos[1, i]), "Puntaje": float(y_datos[0, i])}
        for i in range(X_datos.shape[1])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dimensiones e inicialización

    Usaremos $m$ ejemplos, $n_x$ características y una salida:

    | Arreglo | Forma |
    |---|---|
    | $X$ | $(n_x,m)$ |
    | $y$ | $(1,m)$ |
    | $W$ | $(1,n_x)$ |
    | $b$ | $(1,1)$ |
    | $\hat y$ | $(1,m)$ |

    Completa `layer_sizes` e `initialize_parameters`. Inicializa pesos pequeños
    con `np.random.default_rng(seed)` y el sesgo en cero. Para esta neurona,
    inicializar en cero también sería posible; usaremos una semilla para
    comparar experimentos de forma reproducible.

    **Tu respuesta:** indica las dimensiones concretas y explica por qué
    el producto $WX$ y la suma del sesgo son válidos. ¿Qué controla la semilla?
    """)
    return


@app.cell
def _():
    def layer_sizes(X, y):
        # TODO: devolver n_x y n_y a partir de las formas de X e y.
        return None, None

    def initialize_parameters(n_x, n_y, seed=42):
        # TODO: crear y devolver {"W": W, "b": b}.
        return None

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Propagación hacia adelante

    $$Z=WX+b,\qquad \hat y=Z$$

    Implementa `forward_propagation`. Recibe los datos y un diccionario de
    parámetros; devuelve un arreglo con todas las predicciones.

    **Tu respuesta:** ¿qué aportan los pesos y el sesgo? ¿Por qué usamos
    la función identidad para la salida de este ejercicio?
    """)
    return


@app.function
def forward_propagation(X, parameters):
    # TODO: calcular las predicciones utilizando W y b.
    return None


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Función de pérdida

    $$J(W,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y^{(i)}-y^{(i)})^2$$

    Implementa `compute_cost`: debe devolver un escalar. Utiliza `np.sum`;
    si aparecen valores no finitos, investiga su causa.

    **Tu respuesta:** ¿por qué elevamos el error al cuadrado? ¿Qué efecto
    tiene el factor $1/2$ en la derivada? ¿Pérdida cero implica siempre
    que el modelo funcionará bien con datos nuevos?
    """)
    return


@app.function
def compute_cost(y_hat, y):
    # TODO: implementar la pérdida especificada.
    return None


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Gradientes y actualización

    Con pesos organizados como **fila**, las ecuaciones son:

    \[
    dZ=\hat y-y,\qquad dW=\frac{1}{m}dZ X^T,
    \qquad db=\frac{1}{m}\sum_{i=1}^{m}dZ^{(i)}\]

    \[W_{\mathrm{nuevo}}=W-\alpha dW,\qquad
    b_{\mathrm{nuevo}}=b-\alpha db\]

    Implementa las dos funciones. Conserva las formas de `dW` y `db`;
    `np.sum(..., axis=1, keepdims=True)` puede ayudarte con el sesgo.
    La actualización debe devolver un diccionario nuevo sin modificar
    los arreglos originales en el lugar.

    **Tu respuesta:** muestra la derivación de $\partial J/\partial W_j$
    y $\partial J/\partial b$. Explica por qué restamos el gradiente y
    qué representa la tasa de aprendizaje $\alpha$.
    """)
    return


@app.cell
def _():
    def back_propagation(y_hat, y, X):
        # TODO: devolver {"dW": dW, "db": db}.
        return None

    def update_parameters(parameters, grads, learning_rate):
        # TODO: devolver los parámetros actualizados.
        return None

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Una iteración a mano y con NumPy

    Utiliza exclusivamente los siguientes dos ejemplos y parámetros:

    $$X_{\mathrm{manual}}=\begin{bmatrix}1&2\\0&1\end{bmatrix},
    \quad y_{\mathrm{manual}}=\begin{bmatrix}2&3\end{bmatrix}$$
    $$W_0=\begin{bmatrix}0&0\end{bmatrix},\quad b_0=0,\quad\alpha=0.1$$

    **Ejercicio:**, desarrolla a mano las predicciones iniciales,
    la pérdida, `dZ`, `dW`, `db`, los parámetros actualizados y la pérdida
    después de la actualización. Escribe las operaciones, no solo el resultado.

    **Tu desarrollo:** …

    Después, completa la celda para repetir el cálculo usando tus funciones.
    Muestra los valores y comprueba que coinciden con tu
    desarrollo manual. Explica cualquier diferencia por redondeo.

    **Tu interpretación:** ¿disminuyó la pérdida? ¿Cómo cambiaron las predicciones?
    """)
    return


@app.cell
def _(np):
    X_manual = np.array([[1.0, 2.0], [0.0, 1.0]])
    y_manual = np.array([[2.0, 3.0]])
    parametros_manual = {"W": np.zeros((1, 2)), "b": np.zeros((1, 1))}
    alpha_manual = 0.1
    # TODO: usar forward_propagation, compute_cost, back_propagation
    # y update_parameters. Mostrar los resultados antes y después.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Ciclo de entrenamiento

    Integra tus funciones en `train_model`:

    1. Obtén las dimensiones usando los argumentos `X` e `y`.
    2. Inicializa los parámetros una sola vez.
    3. Guarda la pérdida inicial.
    4. En cada iteración: predice, calcula gradientes, actualiza parámetros
       y guarda la pérdida **después** de actualizar.
    5. Devuelve los parámetros finales y el historial de pérdidas.

    El historial debe tener `iterations + 1` valores: el índice cero es
    la pérdida antes de entrenar. Usa únicamente los datos recibidos por
    la función, para que también funcione con el conjunto del cálculo manual.

    Entrena con 200 iteraciones, tasa 0.1 y semilla 42. Muestra los parámetros,
    las predicciones finales, las pérdidas inicial y final y una gráfica
    de pérdida contra número de actualizaciones.

    **Tu respuesta:** describe qué se repite y qué se conserva en cada iteración.
    ¿Una iteración de este entrenamiento utiliza todos los ejemplos?
    """)
    return


@app.function
def train_model(X, y, iterations=200, learning_rate=0.1, seed=42):
    # TODO: integrar las funciones y devolver (parameters, cost_history).
    return None, None


@app.cell
def _():
    # TODO: entrenar con X_datos e y_datos y mostrar los resultados.
    # TODO: graficar el historial con matplotlib; etiquetar ambos ejes.
    # En marimo, muestra la figura como última expresión de una celda.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Experimento con la tasa de aprendizaje

    Repite el entrenamiento con $\alpha\in\{0.01,0.1,3.0\}$.
    Usa los mismos datos, **200 iteraciones y la misma semilla** en todos
    los casos. Reinicia el entrenamiento del modelo para cada experimento.

    - Grafica las tres curvas, con leyenda y ejes etiquetados. Puedes usar
      escala logarítmica o paneles separados si las magnitudes lo requieren.
    - Construye una tabla con tasa, pérdida inicial, pérdida final y
      si observaste convergencia, oscilaciones o divergencia.
    - Si aparecen valores no finitos, indícalos y explica lo ocurrido.

    **Tu análisis:** ¿qué tasa elegirías entre estas opciones y por qué?
    ¿Por qué una tasa mayor no garantiza aprender mejor? ¿Qué puedes
    concluir con solo 200 iteraciones y qué no puedes afirmar todavía?
    """)
    return


@app.cell
def _():
    # TODO: ejecutar los tres experimentos, construir la tabla y las gráficas.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Conclusiones y conexión con Deep Learning

    Responde con referencia a tus resultados:

    1. ¿Qué significa que la neurona haya aprendido? Relaciona parámetros,
       predicciones y pérdida.
    2. ¿Qué diferencia existe entre calcular gradientes y actualizar parámetros?
    3. ¿Cómo comprobarías si una implementación entrega resultados correctos,
       además de observar que la pérdida disminuye?
    4. Si usamos sigmoide y entropía cruzada para clasificación binaria,
       ¿qué partes del proceso cambian y cuáles conservan su estructura?

    **Tus conclusiones:** …

    ## Rúbrica

    | Criterio | Valor | Evidencia esperada |
    |---|---:|---|
    | Implementación correcta | 40 % | Dimensiones coherentes, funciones propias, actualización y ciclo completos; coincidencia con la iteración manual. |
    | Explicación matemática | 30 % | Ecuaciones, derivación de gradientes y explicación del papel de cada paso. |
    | Experimentación e interpretación | 20 % | Comparación controlada de tasas, gráficas, tabla y conclusiones sustentadas. |
    | Claridad y reproducibilidad | 10 % | Notebook ordenado, semilla fija, respuestas completas y ejecución sin errores. |

    Revisa tus resultados antes de entregar.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
