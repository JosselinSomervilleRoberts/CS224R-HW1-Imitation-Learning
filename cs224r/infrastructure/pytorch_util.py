'''
Functions to edit:
    1. build_mlp (line 26) 
'''


from typing import Tuple

import torch
from torch import nn
import numpy as np

Activation = Tuple[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # DONE: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.

    # First Layer
    layers = [nn.Linear(input_size, size), activation]

    # Hidden Layers
    for _ in range(n_layers-1):
        layers.append(nn.Linear(size, size))
        layers.append(activation)

    # Output Layer
    layers.append(nn.Linear(size, output_size))
    layers.append(output_activation)

    mlp = nn.Sequential(*layers)
    return mlp


device = None


def init_gpu(use_gpu: bool = True, gpu_id: int = 0) -> None:
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id: int) -> None:
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs) -> torch.FloatTensor:
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor: torch.FloatTensor) -> np.ndarray:
    return tensor.to('cpu').detach().numpy()
