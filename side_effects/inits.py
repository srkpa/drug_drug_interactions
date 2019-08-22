import numpy as np
import torch.nn as nn
import torch as th


def init_uniform(layer):
    if isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight.data, -1, 1)


def init_normal(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight.data, 0, 1)


def init_constant(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0.1)


def init_xavier_normal(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight.data, gain=1)


def init_kaiming_uniform(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight.data, a=1)



def weight_variable_glorot(input_dim, output_dim):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = th.rand(input_dim, output_dim).uniform_(-init_range, init_range)
    return initial


def zeros(input_dim, output_dim):
    """All zeros."""
    initial = th.zeros(input_dim, output_dim)
    return initial


def ones(input_dim, output_dim):
    """All zeros."""
    initial = th.ones(input_dim, output_dim)
    return initial
