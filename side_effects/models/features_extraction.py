import torch
import torch.nn as nn
from ivbase.nn.commons import (GlobalMaxPool1d, Transpose)
from ivbase.nn.graphs.conv.gcn import GCNLayer
from torch.nn import Conv1d, Embedding
import ivbase.nn.extractors as feat
from ivbase.nn.base import FCLayer
from ivbase.nn.commons import get_activation


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


class DGLGraph(nn.Module):
    def __init__(self, input_dim, conv_layer_dims, dropout=0., activation="relu",
                 b_norm=False, pooling='sum',
                 bias=False, init_fn=None):
        super(DGLGraph, self).__init__()

        conv_layers = []
        in_size = input_dim
        for dim in conv_layer_dims:
            gconv = GCNLayer(in_size=in_size, out_size=dim, dropout=dropout, activation=activation, b_norm=b_norm,
                             pooling=pooling, bias=bias, init_fn=init_fn)

            conv_layers.append(gconv)
            in_size = dim
        self.conv_layers = nn.ModuleList(conv_layers)
        self.output_dim = in_size

    def forward(self, x):
        h = 0
        G = x

        for cv_layer in self.conv_layers:
            G, h = cv_layer(G)

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

        last_layer_activation = get_activation(last_layer_activation)
        self.net = nn.Sequential(*layers, nn.Linear(in_features=in_size, out_features=output_dim), last_layer_activation)
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
        dglgraph=DGLGraph
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

