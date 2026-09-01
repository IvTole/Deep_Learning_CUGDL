import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Perceptron (regression)
    """)
    return


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import time

    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    from sklearn.linear_model import LinearRegression

    return (
        ColumnTransformer,
        LinearRegression,
        StandardScaler,
        mo,
        np,
        pd,
        plt,
        time,
    )


@app.cell
def _(np, pd):
    ## Importacion de datos
    df = pd.read_csv("../data/cars/cars_2025.csv", encoding='latin-1')
    df = df.rename(columns={"Performance(0 - 100 )KM/H":"Performance"})

    # parches
    #df["Performance"] = df["Performance"].str.strip(" sec")
    #df["Performance"] = df["Performance"].str.replace(" ", "")
    #df["Performance"] = df["Performance"].str.replace("sec", "")
    #df["Performance"] = df["Performance"].str.replace("/", "")
    #df["Performance"] = df["Performance"].str[:3]
    #df["Performance"] = df["Performance"].astype("float64")

    df['Performance'] = df['Performance'].str.extract(r'([\d.]+)').astype(float)

    #df["HorsePower"] = df["HorsePower"].str.strip(" hp")
    #df["HorsePower"] = df["HorsePower"].str.strip(" HP")
    #df["HorsePower"] = df["HorsePower"].str.strip(" cc")
    #df["HorsePower"] = df["HorsePower"].str.replace(" hp / ","-")
    #df["HorsePower"] = df["HorsePower"].str.strip("~")
    #df["HorsePower"] = df["HorsePower"].str.strip("Up to ")
    #df["HorsePower"] = df["HorsePower"].str.strip(" hp (est.)")
    #df["HorsePower"] = df["HorsePower"].str.strip(' \x96 196')

    #df["HorsePower"] = df["HorsePower"].str.replace(",","")
    #df["HorsePower"] = df["HorsePower"].str.split("-").str[0]
    #df["HorsePower"] = df["HorsePower"].astype("int64")

    df['HorsePower'] = df['HorsePower'].str.extract(r'(\d+)').astype(float)

    df['Total Speed'] = df['Total Speed'].str.extract(r'(\d+)').astype(float)

    df['Cars Prices'] = df['Cars Prices'].str.replace('[$,]', '', regex=True)\
                                  .str.extract(r'(\d+)').astype(float)

    # Standardize Company names to Title Case
    df['Company Names'] = df['Company Names'].str.strip().str.title()
    df['Fuel Types'] = df['Fuel Types'].str.strip().str.title()

    df['HorsePower_log'] = np.log(df["HorsePower"])
    df['Performance_log'] = np.log(df["Performance"])

    # nulls
    df = df.dropna()

    df.head()
    return (df,)


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(ColumnTransformer, StandardScaler, df):
    df_filter = df[["HorsePower", "Performance", "Total Speed",  "Cars Prices"]]
    cols_num = ["HorsePower", "Performance", "Total Speed",  "Cars Prices"]

    # standardize

    # paso 1, definir transformaciones
    scaler = StandardScaler()

    # paso 2, meterlo dentro de un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, cols_num)
        ],
        remainder="passthrough"
    )

    preprocessor.set_output(transform="pandas")

    df_processed = preprocessor.fit_transform(df_filter)

    df_processed.columns = ["HorsePower", "Performance", "Total Speed", "Cars Prices"]

    df_processed
    return df_filter, df_processed


@app.cell
def _():
    return


@app.cell
def _(df_filter, plt):
    plt.scatter(df_filter["HorsePower"], df_filter["Cars Prices"])
    plt.xlabel("HorsePower")
    plt.ylabel("Price")
    return


@app.cell
def _(df_processed):
    X = df_processed.drop(columns =["Cars Prices"])
    y = df_processed["Cars Prices"]
    return X, y


@app.cell
def _(X, y):
    # Transform to numpy
    X_array = X.to_numpy().T
    y_array = y.to_numpy().reshape((1, len(y)))
    return X_array, y_array


@app.cell
def _(X_array):
    X_array.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 1
    Definir dimensiones de entrada y de salida del perceptron
    """)
    return


@app.function
def layer_sizes(X,y):
    """
    Argumentos:
    X - conjunto de datos de entrada (features)
    y - datos de salida (target)
    """
    n_x = X.shape[0]
    n_y = y.shape[0]

    return n_x, n_y


@app.cell
def _(X_array, y_array):
    n_x, n_y = layer_sizes(X_array,y_array)
    return n_x, n_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 2
    Inicializar pesos $w$ y bias $b$ (aleatoriamente)
    """)
    return


@app.cell
def _(np):
    def initialize_parameters(n_x, n_y):
        """
        Argumento:
        n_x - dimensiones de entrada
        n_y - dimensiones de salida
        """

        # Inicializamos pesos
        W = np.random.randn(n_y, n_x)*0.1

        # Inicializamos bias
        b = np.random.randn(n_y, 1)*0.1

        parameters = {
            "W":W,
            "b":b
        }

        return parameters

    return (initialize_parameters,)


@app.cell
def _(initialize_parameters, n_x, n_y):
    # inicializamos parametros 
    parameters = initialize_parameters(n_x=n_x, n_y=n_y)
    parameters
    return (parameters,)


@app.cell
def _(np):
    def forward_propagation(X, parameters):
        """
        Argumentos
        X_array - arreglo de datos de entrada
        parameters - diccionario con pesos W, y sesgo b
        """

        W = parameters["W"]
        b = parameters["b"]

        # propagacion hacia adelante
        z = np.matmul(W,X) + b

        # la prediccion (funcion identidad)
        y_hat = z

        return y_hat

    return (forward_propagation,)


@app.cell
def _(X_array, forward_propagation, parameters):
    # propagacion hacia adelanta
    y_hat = forward_propagation(X=X_array, parameters=parameters)
    y_hat
    return (y_hat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 4
    Cálculo de la función de pérdida (error). La función de pérdida que vamos a utilizar es la siguiente:

    \begin{equation}
    \mathcal{L}(w,b) = \frac{1}{2n} \sum_i^n (\hat{y}^{(i)} - y^{(i)})^2
    \end{equation}
    """)
    return


@app.cell
def _(np):
    def cost(y_hat, y):

        n = y_hat.shape[1]

        # calculo del costo
        cost = np.nansum((y_hat - y)**2.0) / (2*n)

        return cost


    return (cost,)


@app.cell
def _(cost, y_array, y_hat):
    error = cost(y_hat=y_hat, y=y_array)
    error
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 5 Backpropagation

    Se calcula el gradiente,

    \begin{equation}
    \nabla = (\partial/\partial w, \partial / \partial b)
    \end{equation}

    Ahora calculamos las derivadas parciales mostradas en secciones anteriores,

    \begin{equation}
    \frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}
    \end{equation}

    \begin{equation}
    \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)
    \end{equation}
    """)
    return


@app.cell
def _(np):
    def back_propagation(y_hat, y, X):
        """
        Argumentos:
        y_hat - prediccion (que viene de forward propagation)
        y - etiquetas (reales)
        X - matriz de características

        Returns:
        gradient - función gradiente
        """

        # numero de datos
        n = X.shape[1]

        # gradiente (derivadas parciales)
        dZ = y_hat - y
        dW = (1/n) * np.dot(dZ, X.T)
        db = (1/n) * np.sum(dZ, axis=1, keepdims=True)

        grads = {
            "dW": dW,
            "db": db
        }

        return grads

    

    return (back_propagation,)


@app.cell
def _(X_array, back_propagation, y_array, y_hat):
    grads = back_propagation(y_hat=y_hat, y=y_array, X=X_array)
    grads
    return (grads,)


@app.function
def optimize_parameters(parameters, grads, learning_rate=1.0):
    """
    Argumentos:
    parameters - w,b
    learning_rate - tasa de aprendizaje alpha
    grads - cálculo del gradiente para w,b

    Returns:
    parameters - parámetros actualizados w,b
    """

    # pesos W, bias b
    W = parameters["W"]
    b = parameters["b"]

    # grads
    dW = grads["dW"]
    db = grads["db"]

    # método de optimizacion (gradient descent)
    W = W - learning_rate * dW
    b = b - learning_rate * db

    parameters = {
        'W':W,
        'b':b
    }

    return parameters


@app.cell
def _(grads, parameters):
    learning_rate = 0.1
    def _(parameters=parameters, grads=grads, learning_rate=learning_rate):
        parameters = optimize_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)
        return parameters
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural Network
    Juntamos todas las funciones anteriores y creamos un loop de entrenamiento
    """)
    return


@app.cell
def _(
    X_array,
    back_propagation,
    cost,
    forward_propagation,
    initialize_parameters,
    y_array,
):
    def nn_model(X,y, iterations=100, learning_rate=0.1, print_cost=False):
        """
        Argumentos:

        Returns:
        parameters - parametros aprendidos por el modelo. Que se pueden usar luego para hacer predicciones
        """

        # paso 1 definir dimensiones
        n_x, n_y = layer_sizes(X_array,y_array)

        # paso 2 inicializar w,b (aleatoriamente)
        parameters = initialize_parameters(n_x=n_x, n_y=n_y)
        #print(parameters)

        # Ciclo de entrenamiento
        for i in range(0,iterations):

            # paso 3 forward propagation (inferencia)
            y_hat = forward_propagation(X=X_array, parameters=parameters)
    
            # paso 4 calcular error (costo)
            error = cost(y_hat=y_hat, y=y_array)
    
            # paso 5 backpropagation (calcular gradientes)
            grads = back_propagation(y_hat=y_hat, y=y_array, X=X_array)
            #print(grads)
        
            # paso 6 optimizar parametros (descenso del gradiente)
            parameters = optimize_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)
            #print(parameters)

            if print_cost:
                print(f"Costo después de la iteración {i}: {error}")

        return parameters

    return (nn_model,)


@app.cell
def _(X_array, nn_model, time, y_array):
    t_init = time.time()
    parameters_regression = nn_model(X=X_array, y=y_array, iterations=100, learning_rate=0.5, print_cost=True)
    t_end = time.time()
    print(t_end - t_init)
    return parameters_regression, t_end, t_init


@app.cell
def _(parameters_regression):
    # parametros resultantes del aprendizaje del perceptron
    parameters_regression
    return


@app.cell
def _(LinearRegression, X, time, y):
    # paso 1
    model = LinearRegression()

    t_init_sklearn = time.time()
    # paso 2
    model.fit(X,y)
    t_end_sklearn = time.time()
    print(t_end_sklearn - t_init_sklearn)

    # parametros
    print(f"beta0 : {model.intercept_}")
    print(f"betas : {model.coef_}")
    return t_end_sklearn, t_init_sklearn


@app.cell
def _(t_end, t_end_sklearn, t_init, t_init_sklearn):
    (t_end - t_init) / (t_end_sklearn - t_init_sklearn)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
