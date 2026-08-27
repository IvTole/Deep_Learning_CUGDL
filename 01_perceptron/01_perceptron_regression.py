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

    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    return ColumnTransformer, StandardScaler, mo, np, pd, plt


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
    #df = df.dropna()

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
