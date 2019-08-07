import torch
from torch import nn


class DeepDDI(nn.Module):

    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(DeepDDI, self).__init__()
        layers = []
        in_ = input_dim
        for out_ in hidden_sizes:
            layers.append(nn.Linear(in_, out_))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_))
            in_ = out_
        self.net = nn.Sequential(*layers, nn.Linear(in_, output_dim), nn.Sigmoid())

    def forward(self, batch):
        x = batch
        ddi = x.type('torch.cuda.FloatTensor') if torch.cuda.is_available() else x.type('torch.FloatTensor')
        return self.net(ddi)
