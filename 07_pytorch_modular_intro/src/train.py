import torch
from get_data import get_data
from model_builder import Model_Classification
from engine import train
from config import HIDDEN_UNITS, LEARNING_RATE, EPOCHS

# Script principal de entrenamiento
def main():

    X_train, y_train, X_test, y_test = get_data()

    # Set up
    # model
    model = Model_Classification(input_shape=64,
                                 hidden_units=HIDDEN_UNITS,
                                 output_features=10)
    print(model.parameters)
    # función de pérdida
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    ## entrenamiento
    train(model=model,
          X=X_train,
          y=y_train,
          optimizer=optimizer,
          loss_fn=loss_fn,
          device="cpu",
          epochs=EPOCHS)


if __name__ == "__main__":
    main()