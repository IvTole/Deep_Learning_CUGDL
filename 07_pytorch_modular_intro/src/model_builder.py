# Modelos o arquitecturas utilizadas (red neuronal)

import torch
from torch import nn
from config import SEED

torch.manual_seed(SEED)

class Model_Classification(nn.Module):
    """
    Modelo base de clasificacion

    Atributos:
    Layer 1
    Layer 2
    Layer 3
    ReLU activation function

    Forward:
    ReLU(layer1) -> ReLU8(layer2) -> layer3
    """
    
    def __init__(self, input_shape:int, hidden_units:int=20, output_features:int=10):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_shape, out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer_3 = nn.Linear(in_features=hidden_units, out_features=output_features)
        self.relu = nn.ReLU()

    def forward(self, x):
      # ReLU se aplica entre capas
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))