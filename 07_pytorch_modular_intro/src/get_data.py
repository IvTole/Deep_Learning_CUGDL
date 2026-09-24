import torch # no mover
import pandas as pd
from config import TRAIN_PATH, TEST_PATH

def get_data() -> tuple:
    """
    Descripción

    Carga de datos. Cargar los datos en pandas y transformarlos a arreglos de torch.

    Retorna:
    Tensores de pytorch
    X_train - Features de entrenamiento
    X_test - Features de prueba
    y_train - target de entrenamiento
    y_test - target de prueba
    """

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    X_train = df_train.drop(columns=['target'])
    y_train = df_train["target"]
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    # pasar a numpy
    X_train_array = X_train.to_numpy()
    y_train_array = y_train.to_numpy()
    X_test_array = X_test.to_numpy()
    y_test_array = y_test.to_numpy()

    print(X_train_array.shape)

    # convertir a tensores de pytorch
    X_train_torch = torch.from_numpy(X_train_array).type(torch.float32)
    y_train_torch = torch.from_numpy(y_train_array).type(torch.int64)
    X_test_torch = torch.from_numpy(X_test_array).type(torch.float32)
    y_test_torch = torch.from_numpy(y_test_array).type(torch.int64)

    #print(X_train_torch.shape, X_test_torch.shape, y_train_torch.shape, y_test_torch.shape)


    return X_train_torch, y_train_torch, X_test_torch, y_test_torch

