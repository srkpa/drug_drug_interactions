import torch
import dgl
import inspect
import torch.nn as nn
from ivbase.nn.commons import (GlobalMaxPool1d, Transpose)
from ivbase.nn.graphs.conv.gcn import GCNLayer, TorchGCNLayer, TorchKGCNLayer
from torch.nn import Conv1d, Embedding
import ivbase.nn.extractors as feat
from ivbase.nn.base import FCLayer
from ivbase.nn.commons import get_activation, get_pooling
from ivbase.nn.graphs.conv import TorchGINConv
from side_effects.pooler.laplacianpooling import LaplacianPool
from side_effects.pooler.oraclepooling import OraclePool
from side_effects.pooler.attnpool import AttnCentroidPool
from side_effects.pooler.diffpool import DiffPool
from ivbase.nn.graphs.pool import TopKPool, ClusterPool, AttnPool
from functools import partial
from .graphs import AggLayer, get_graph_layer


def get_graph_pooler(**params):
    arch = params.pop("arch", None)
    if arch is None:
        return None
    if arch == "diff":
        pool = partial(DiffPool, **params)
    elif arch == "attn":
        pool = partial(AttnPool, **params)
    elif arch == "topk":
        pool = partial(TopKPool, **params)
    elif arch == "cluster":
        pool = partial(ClusterPool, algo="hierachical", gweight=0.5, **params)
    elif arch == "laplacian":
        pool = partial(LaplacianPool, **params)
    return pool


class PCNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, cnn_sizes, kernel_size, dilatation_rate=1, n_blocks=1,
                 output_dim=86):
        super(PCNN, self).__init__()
        self.output_dim = output_dim
        self.embedding = Embedding(vocab_size, embedding_size)
        self.transpose = Transpose(1, 2)
        self.gpool = GlobalMaxPool1d(dim=1)
        layers = []
        for i, (out_channel, ksize) in \
                enumerate(zip(cnn_sizes, kernel_size)):
            pad = ((dilatation_rate ** i) * (ksize - 1) + 1) // 2
            layers.append(Conv1d(embedding_size, out_channel, padding=pad,
                                 kernel_size=ksize, dilation=dilatation_rate ** i))

        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transpose(x)
        f_maps = []
        for layer in self.blocks:
            out = layer(x)
            out = self.gpool(out)
            f_maps.append(out)
        out = torch.cat(f_maps, 1)
        fc = nn.Linear(in_features=out.shape[1], out_features=self.output_dim)
        if torch.cuda.is_available():
            fc = fc.cuda()
        return fc(out)


class GraphNet(nn.Module):
    def __init__(self, input_dim, conv_layer_dims, activation, gather=None, b_norm=False, dropout=0., gather_dim=100,
                 pool_arch=None, glayer="gin", **kwargs):
        super(GraphNet, self).__init__()
        conv_layers_1 = []
        conv_layers_2 = []
        graph_layer = get_graph_layer(glayer)
        activation = get_activation(activation)
        for dim in conv_layer_dims[0]:
            if glayer.startswith("dgl"):
                layer = graph_layer(in_size=input_dim, out_size=dim, dropout=dropout, activation=activation,
                                    b_norm=b_norm, **kwargs)
            else:
                layer = graph_layer(input_dim, kernel_size=dim, b_norm=b_norm, dropout=dropout, activation=activation,
                                    **kwargs)
            conv_layers_1.append(layer)
            input_dim = dim
        self.__before_pooling_layers = nn.ModuleList(conv_layers_1)
        self.pool_arch = pool_arch.get("arch", None)
        self.pool_layer = get_graph_pooler(**pool_arch)
        if self.pool_layer:
            self.pool_layer = self.pool_layer(input_dim)
        if self.pool_arch == "laplacian":
            input_dim = self.pool_layer.cluster_dim

        for dim in conv_layer_dims[-1]:
            if glayer.startswith("dgl"):
                layer = graph_layer(in_size=input_dim, out_size=dim, dropout=dropout, activation=activation,
                                    b_norm=b_norm, **kwargs)
            else:
                layer = graph_layer(input_dim, kernel_size=dim, b_norm=b_norm, dropout=dropout, activation=activation,
                                    **kwargs)
            conv_layers_2.append(layer)
            input_dim = dim

        self.__after_pooling_layers = nn.ModuleList(conv_layers_2)
        if gather == "agg":
            self.agg_layer = AggLayer(input_dim, gather_dim, dropout)
        elif gather in ["attn", "gated"]:
            self.agg_layer = get_pooling("attn", input_dim=input_dim, output_dim=gather_dim, dropout=dropout)
        elif gather is None:
            self.agg_layer = None
        else:
            gather_dim = input_dim
            self.agg_layer = get_pooling(gather)
        input_dim = gather_dim if gather else input_dim
        self.output_dim = input_dim

    def forward(self, x):
        out = []
        h = 0
        if isinstance(x[0], dgl.DGLGraph) and not self.pool_arch:
            G = dgl.batch(x)
            if torch.cuda.is_available():
                G.ndata['hv'] = G.ndata['hv'].cuda()
            x = [(G, 0)]

        for G, h in x:
            args = []
            for cv_layer in self.__before_pooling_layers:
                args = inspect.getfullargspec(cv_layer.forward)[0]
                if 'x' in args:
                    G, h = cv_layer(G, h)
                else:
                    G, h = cv_layer(G)
            if self.pool_layer:
                G, h = self.pool_layer(G, h)
            for cv_layer in self.__after_pooling_layers:
                if 'x' in args:
                    G, h = cv_layer(G, h)
                else:
                    G, h = cv_layer(G)
            if self.agg_layer:
                h = self.agg_layer(h)
                out.append(h)
        h = torch.cat(out, dim=0) if out else h

        return h


class FCNet(nn.Module):
    def __init__(self, input_size, fc_layer_dims, output_dim, activation='relu', last_layer_activation='relu',
                 dropout=0., b_norm=False, bias=True, init_fn=None):
        super(FCNet, self).__init__()
        layers = []
        in_size = input_size

        for layer_dim in fc_layer_dims:
            fc = FCLayer(in_size=in_size, out_size=layer_dim, activation=activation, dropout=dropout, b_norm=b_norm,
                         bias=bias, init_fn=init_fn)
            layers.append(fc)
            in_size = layer_dim

        self.net = nn.Sequential(*layers, nn.Linear(in_features=in_size, out_features=output_dim))
        if last_layer_activation is not None:
            self.net.add_module('last_layer_activation', get_activation(last_layer_activation))

        self.output_dim = output_dim

    def forward(self, inputs):
        return self.net(inputs)


class FeaturesExtractorFactory:
    name_map = dict(
        pcnn=PCNN,
        conv1d=feat.Cnn1dFeatExtractor,
        lstm=feat.LSTMFeatExtractor,
        fclayer=FCLayer,
        fcnet=FCNet,
        dglgraph=GraphNet
    )

    def __init__(self):
        super(FeaturesExtractorFactory, self).__init__()

    def __call__(self, arch, **kwargs):
        if arch not in self.name_map:
            raise Exception(f"Unhandled feature extractor. The name of \
             the architecture should be one of those: {list(self.name_map.keys())}")
        fe_class = self.name_map[arch.lower()]
        feature_extractor = fe_class(**kwargs)
        return feature_extractor
