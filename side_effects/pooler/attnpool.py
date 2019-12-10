import torch
import math
import torch.nn.functional as F
from torch import nn
from ivbase.nn.graphs.conv import TorchGCNLayer
from ivbase.nn.graphs.conv.gin import TorchGINConv
from ivbase.nn.base import FCLayer
from ivbase.nn.graphs.pool.diffpool import DiffPool 
from functools import partial
import networkx as nx
import numpy as np
from side_effects.utility.sparsegen import Sparsegen, Sparsemax
from .base import *


class MPNAttention(nn.Module):
    r"""
    Base class for computing node-to-cluster attention pooling
    
    """
    def __init__(self, input_size, output_size, ncluster, learned=False, concat=False, **kwargs):

        super(MPNAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ncluster = ncluster
        self.learned = learned
        self.concat = concat
        self.attention_net = nn.Linear(self.output_size + self.input_size, 1, bias=False) # Network for attention (NxM)
        self.feat_net = TorchGINConv(self.input_size, self.output_size) # Network for new feats NxF, where F is features
        self.output_net = nn.Linear(self.output_size, self.output_size) # Network for new output feature (MxF)
        if self.learned:
            self.mapper_net = nn.Parameter(torch.Tensor(size=(self.ncluster, self.output_size)))  # alternative mapper func MxF parameter
        else:
            self.mapper_net = kwargs.get("net", TorchGINConv)(self.input_size, self.ncluster, bias=False) # Mapper function: NxM
        self.init_fn = kwargs.get('init_fn') or partial(nn.init.xavier_normal_, gain=1.414)
        self.reset_parameters()
        self.softmax = Sparsemax()

    def reset_parameters(self, init_fn=None):
        r"""
        Initialize weights of the models, as defined by the input initialization scheme.

        Arguments
        ----------
            init_fn (callable, optional): Function to initialize the linear weights. If it is not provided
                an attempt to use the object `ivbase.nn.graphs.gcn.GCNLayer.init_fn` attributes would be first made, before doing nothing.
                (Default value = None)

        See Also
        --------
            `torch.nn.init` for more information
        """
        init_fn = init_fn or self.init_fn
        if init_fn:
            # see paper for initialization scheme
            if self.learned:
                init_fn(self.mapper_net.data)
            init_fn(self.output_net.weight)
            init_fn(self.attention_net.weight)

    def get_value(self, adj, x):
        _, feats = self.feat_net(adj, x) # temporary output N,D --> N,F
        if self.learned:
            mapper = self.mapper_net # is M, F by itself
        else:
            _, f = self.mapper_net(adj, x) # N, D --> N, M
            mapper = self.output_net(torch.bmm(f.transpose(-2, -1), x)) # M, F # precluster
        return mapper, feats
        
    def forward(self, adj, x):
        b_size = x.size(0) if x.dim() == 3 else 1
        assert x.size(-1) == self.input_size
        query = x # N, D
        mapper, val = self.get_value(adj, x)# M, F for mapper, and values is N, F
        if self.concat:
            h1 = upsample_to(query, mapper.shape[-2]) # NM, D # could potentially replace query by val ?
            h2 = upsample_to(mapper, query.shape[-2]) # NM, F
            h = torch.cat([h1, h2], dim=-1) # NM, F+D
            attention_matrix = self.attention_net(h) # should be N*M, 1 now
        else:
            attention_matrix = torch.matmul(val, mapper.transpose(-1, -2)) # should be N, M now
        # reshaping to B, N, M now
        attention_matrix = attention_matrix.view(b_size, -1, self.ncluster) if x.dim() == 3 else attention_matrix.view(-1, self.ncluster)
        attention_matrix = self.softmax(attention_matrix, dim=-1) # B, N, M
        return attention_matrix, mapper


class AttnCentroidPool(DiffPool):
    r"""
    This layer implements a differentiable pooling layer that uses attention soft attention to compute the feature  features
    and the cluster feature to compute the 

    """
    def __init__(self, input_dim, hidden_dim, feat_dim, **kwargs):
        super(AttnCentroidPool, self).__init__(input_dim, hidden_dim, None)
        self.nclusters = self.hidden_dim
        self.feat_dim = feat_dim
        self.cur_S = None
        self.net = MPNAttention(self.input_dim, self.feat_dim, hidden_dim, **kwargs)

    def _loss(self, adj, mapper):
        r"""
        Compute the auxiliary link prediction loss for the given batch

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Adjacency matrix of size (B, N, N)
            mapper: `torch.FloatTensor`
                Node assignment matrix of size (B, N, M)
        """
        return (adj - torch.matmul(mapper, mapper.transpose(1, 2))).norm()


    def forward(self, adj, x, return_loss=False):
        r"""
        Applies the differential pooling layer on input tensor x and adjacency matrix 

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Unormalized adjacency matrix of size (B, N, N)
            x: torch.FloatTensor 
                Node feature after embedding of size (B, N, D). See paper for more infos
            return_loss: bool, optional
                Whether to return the auxiliary link prediction loss

        Returns
        -------
            new_adj: `torch.FloatTensor`
                New adjacency matrix, describing the reduced graph
            new_feat: `torch.FloatTensor`
                Node features resulting from the pooling operation.
            loss: `torch.FloatTensor`, optional
                optional edge prediction loss for the layer

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """

        # we always want batch computation
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # batch, oh yeaahhh
        S, new_feat = self.net(adj, x)
        new_adj = (S.transpose(-2, -1)).matmul(adj).matmul(S)
        self.cur_S = S
        if return_loss:
            loss = self._loss(adj, S)
            return new_adj, new_feat, loss
        return new_adj, new_feat