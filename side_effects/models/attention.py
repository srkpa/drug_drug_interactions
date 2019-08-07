import numpy as np
import torch
from torch import nn


class AttentionLayer(nn.Module):
    """
    Attention layer that performs dot product att
    """
    def __init__(self, input_dim, value_dim, key_dim, pooling_function=None):

        super(AttentionLayer, self).__init__()

        self.query_network = nn.Linear(input_dim, key_dim)
        self.key_network = nn.Linear(input_dim, key_dim)
        self.value_network = nn.Linear(input_dim, value_dim)

        self.norm_layer = nn.LayerNorm(value_dim)
        self.pooling_function = pooling_function
        self._output_dim = value_dim

        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key=None, value=None):
        if key is None and value is None:
            key = query
            value = query
        assert query.dim() == 3

        query = self.query_network(query)
        key = self.key_network(key)
        value = self.value_network(value)

        attention_matrix = torch.bmm(query, key.transpose(1, 2))
        attention_matrix = attention_matrix / np.sqrt(query.size(2))
        attention_matrix = self.softmax(attention_matrix)
        res = self.norm_layer(torch.bmm(attention_matrix, value))

        if self.pooling_function == 'max':
            res = torch.max(res, dim=1)[0]
        elif self.pooling_function == 'mean':
            res = torch.mean(res, dim=1)
        return res

    @property
    def output_dim(self):
        return self._output_dim


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head attention module.
    """

    def __init__(self, num_heads, input_dim, value_dim, key_dim, pooling_function=None, dropout=0.1, residual=True):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_network = nn.Linear(input_dim, num_heads * key_dim)
        self.key_network = nn.Linear(input_dim, num_heads * key_dim)
        self.value_network = nn.Linear(input_dim, num_heads * value_dim)

        self.norm_layer = nn.LayerNorm(input_dim)
        self.out_layer = nn.Linear(num_heads * value_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling_function = pooling_function
        self.residual = residual
        self._output_dim = value_dim

        self.softmax = nn.Softmax(dim=2)

    def forward(self, queries, keys=None, values=None, mask=None):
        if keys is None and values is None:
            keys = queries
            values = queries
        key_dim, value_dim, n_head = self.key_dim, self.value_dim, self.num_heads

        sz_b, len_q, _ = queries.size()
        sz_b, len_k, _ = keys.size()
        sz_b, len_v, _ = values.size()

        residual = queries

        queries = self.query_network(queries).view(sz_b, len_q, n_head, key_dim)
        keys = self.key_network(keys).view(sz_b, len_k, n_head, key_dim)
        values = self.value_network(values).view(sz_b, len_v, n_head, value_dim)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(-1, len_q, key_dim)  # (n*b) x lq x dk
        keys = keys.permute(2, 0, 1, 3).contiguous().view(-1, len_k, key_dim)  # (n*b) x lk x dk
        values = values.permute(2, 0, 1, 3).contiguous().view(-1, len_v, value_dim)  # (n*b) x lv x dv

        attentions = torch.bmm(queries, keys.transpose(1, 2))
        attentions = attentions / np.sqrt(self.key_dim)

        if mask is not None:
            mask = mask.transpose(1, 2)

        attentions = self.softmax(attentions, mask)
        outputs = torch.bmm(attentions, values)

        outputs = outputs.view(n_head, sz_b, len_q, value_dim)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        res = self.dropout(self.out_layer(outputs))
        res = self.norm_layer(res + residual) if self.residual else self.norm_layer(res)

        if self.pooling_function == 'max':
            res = torch.max(res, dim=1)[0]
        elif self.pooling_function == 'mean':
            res = torch.mean(res, dim=1)

        return res

    @property
    def output_dim(self):
        return self._output_dim


class SelfMultiHeadAttentionLayer(nn.Module):
    """
    Concatenation of a self-attention layer and a MLP.
    """

    def __init__(self, input_dim=64, hidden_dims=(128, 64), heads=1, residual=False, dropout=0.):

        super().__init__()

        self.attention = MultiHeadAttentionLayer(
            input_dim=input_dim,
            num_heads=heads,
            value_dim=hidden_dims[0],
            key_dim=hidden_dims[0],
            residual=residual,
            dropout=dropout
        )

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            *[
                nn.Sequential(nn.ReLU(), nn.Linear(h1, h2))
                for h1, h2 in zip(hidden_dims[:-1], hidden_dims[1:])
            ]
        )

        self.output_dim = hidden_dims[-1]

    def forward(self, x, mask=None):
        """

        Parameters
        ----------
        x: torch.Tensor
            B * N * D tensor. The attention layer expects a 3-dimensional tensor.
            The rest of the modules are linear, and can hence work with multi-dimensional tensors.
        mask: torch.Tensor
            Tensor masking incomplete examples. Defaults to None (no masking).

        Returns
        -------
        x: torch.Tensor
            B * N * D tensor.
        """
        x = self.attention(x, mask=mask)
        x = self.hidden(x)

        return x
