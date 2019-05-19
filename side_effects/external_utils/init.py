import torch.nn as nn


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

