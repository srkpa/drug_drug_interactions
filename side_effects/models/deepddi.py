import csv

import torch
from torch import nn


class DeepDDI(nn.Module):

    def __init__(self, input_dim, hidden_sizes, nb_side_effects, ss_embedding_dim=None, exp_prefix=None,
                 is_binary_output=False, testing=False):
        super(DeepDDI, self).__init__()
        layers = []
        in_ = input_dim
        self.is_binary_output = is_binary_output
        self.testing = True #testing
        self.exp_prefix = exp_prefix
        output_dim = 1 if is_binary_output else nb_side_effects
        if is_binary_output:
            self.embedding = nn.Embedding(nb_side_effects, ss_embedding_dim)
            in_ += ss_embedding_dim
        for out_ in hidden_sizes:
            layers.append(nn.Linear(in_, out_))
            #  layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_))
            layers.append(nn.ReLU())
            in_ = out_
        self.net = nn.Sequential(*layers, nn.Linear(in_, output_dim
                                                    ), nn.Sigmoid())

    def forward(self, batch):

        did1 = did2 = sid = None
        if not self.is_binary_output:
            drugs_a, drugs_b, = batch[:2]
            inputs = torch.cat((drugs_a, drugs_b), 1)
        else:
            if self.testing:
                did1, did2, sid, drugs_a, drugs_b, side_eff = batch[:6]
            else:
                drugs_a, drugs_b, side_eff = batch[:3]
            side_eff_features = self.embedding(side_eff).squeeze()
            inputs = torch.cat((drugs_a, drugs_b, side_eff_features), 1)

        out = self.net(inputs)
        # Just for binary cases, i want to have all output effortless way
        if did1 is not None and did2 is not None:
            output = list(zip(did1, did2, sid, out.squeeze().tolist()))
            with open(self.exp_prefix, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(output)

        return out
