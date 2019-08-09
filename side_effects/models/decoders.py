import torch
import torch.nn as nn
from side_effects import inits


class DEDICOMDecoder(nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=nn.Sigmoid(), edge_type=(), num_types=-1):
        super(DEDICOMDecoder, self).__init__()
        self.dropout = nn.Dropout(1 - dropout)
        self.act = act
        self.edge_type = edge_type
        self.num_types = num_types
        self.vars['global_interaction'] = inits.weight_variable_glorot(
            input_dim, input_dim)
        for k in range(self.num_types):
            tmp = inits.weight_variable_glorot(
                input_dim, 1)
            self.vars['local_variation_%d' % k] = torch.flatten(tmp)

    def forward(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = self.dropout(inputs[i])
            inputs_col = self.dropout(inputs[j])
            relation = torch.diag(self.vars['local_variation_%d' % k])
            product1 = torch.mm(inputs_row, relation)
            product2 = torch.mm(product1, self.vars['global_interaction'])
            product3 = torch.mm(product2, relation)
            rec = torch.mm(product3, torch.transpose(inputs_col, 1, 0))
            outputs.append(self.act(rec))
        return outputs


class DistMultDecoder(nn.Module):
    """DistMult Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=nn.Sigmoid(), edge_type=(), num_types=-1):
        super(DistMultDecoder, self).__init__()
        self.edge_type = edge_type
        self.num_types = num_types
        self.dropout = nn.Dropout(1 - dropout)
        self.act = act
        self.vars = {}
        for k in range(self.num_types):
            tmp = inits.weight_variable_glorot(
                input_dim, 1)
            self.vars['relation_%d' % k] = torch.flatten(tmp)

    def forward(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = self.dropout(inputs[i])
            inputs_col = self.dropout(inputs[j])
            relation = torch.diag(self.vars['relation_%d' % k])
            intermediate_product = torch.mm(inputs_row, relation)
            rec = torch.mm(intermediate_product, torch.transpose(inputs_col, 1, 0))
            outputs.append(self.act(rec))
        return outputs


class BilinearDecoder(nn.Module):
    """Bilinear Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=nn.Sigmoid, edge_type=(), num_types=-1):
        super(BilinearDecoder, self).__init__()
        self.edge_type = edge_type
        self.num_types = num_types
        self.dropout = nn.Dropout(1 - dropout)
        self.act = act
        self.vars = {}
        for k in range(self.num_types):
            self.vars['relation_%d' % k] = inits.weight_variable_glorot(
                input_dim, input_dim)

    def forward(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = self.dropout(inputs[i])
            inputs_col = self.dropout(inputs[j])
            intermediate_product = torch.mm(inputs_row, self.vars['relation_%d' % k])
            rec = torch.mm(intermediate_product, torch.transpose(inputs_col, 1, 0))
            outputs.append(self.act(rec))
        return outputs


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=nn.Sigmoid, edge_type=(), num_types=-1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(1 - dropout)
        self.act = act
        self.edge_type = edge_type
        self.num_types = num_types

    def forward(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = self.dropout(inputs[i])
            inputs_col = self.dropout(inputs[j])
            rec = torch.mm(inputs_row, torch.transpose(inputs_col, 1, 0))
            outputs.append(self.act(rec))
        return outputs
