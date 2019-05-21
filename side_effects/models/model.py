import torch
import torch.nn as nn
from ivbase.nn.base import FCLayer
from ivbase.nn.commons import (GlobalMaxPool1d, Transpose)
from torch.nn import Conv1d, Embedding, Module
from ivbase.nn.graphs.conv.gcn import GCNLayer

import ivbase.nn.extractors as feat


def _get_network(all_networks, network, parameters):
    assert isinstance(network, str)
    net = network.lower()
    return all_networks[net](**parameters)


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


class DeepDDI(nn.Module):

    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(DeepDDI, self).__init__()
        layers = []
        in_ = input_dim
        for out_ in hidden_sizes:
            layers.append(nn.Linear(in_, out_))
            layers.append(nn.BatchNorm1d(out_))
            layers.append(nn.ReLU())
            in_ = out_
        self.net = nn.Sequential(*layers, nn.Linear(in_, output_dim), nn.Sigmoid())

    def forward(self, batch):
        x = batch
        ddi = x.type('torch.cuda.FloatTensor') if torch.cuda.is_available() else x.type('torch.FloatTensor')
        return self.net(ddi)


class FCNet(Module):
    def __init__(self, input_size, fc_layer_dims, output_dim, activation='relu', dropout=0., b_norm=False, bias=True,
                 init_fn=None, batch_norm=True):
        super(FCNet, self).__init__()
        layers = []
        in_size = input_size
        for layer_dim in fc_layer_dims:
            fc = FCLayer(in_size=in_size, out_size=layer_dim, activation=activation, dropout=dropout, b_norm=b_norm,
                         bias=bias, init_fn=init_fn)
            layers.append(fc)
            in_size = layer_dim
        layers.append(nn.Linear(in_features=input_size, out_features=output_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))  # Add batch normalization if specified
        self.net = nn.Sequential(*layers, nn.Sigmoid())

    def forward(self, inputs):
        return self.net(inputs)


class DGLGraph(nn.Module):
    def __init__(self, input_dim, conv_layer_dims, dropout=0., activation="relu",
                 b_norm=False, pooling='sum',
                 bias=False, init_fn=None):
        super(DGLGraph, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_size = input_dim
        for dim in conv_layer_dims:
            gconv = GCNLayer(in_size=in_size, out_size=dim, dropout=dropout, activation=activation, b_norm=b_norm,
                             pooling=pooling, bias=bias, init_fn=init_fn)
            self.conv_layers.append(gconv)
            in_size = dim
        self.output_dim = in_size

    def forward(self, x):
        h = 0
        G = x
        for cv_layer in self.conv_layers:
            G, h = cv_layer(G)
        return h


# this above is the main architecture

class DRUUD(nn.Module):

    def __init__(self, drug_feature_extractor, fc_layers_dim, output_dim, **kwargs):
        super(DRUUD, self).__init__()
        self.__extractors = dict(
            pcnn=PCNN,
            conv1d=feat.Cnn1dFeatExtractor,
            lstm=feat.LSTMFeatExtractor,
            fcfeat=feat.FcFeatExtractor,
            dgl=DGLGraph
        )
        self.__drug_feature_extractor_network = drug_feature_extractor["net"]
        self.__drug_feature_extractor_params = drug_feature_extractor["params"]
        self.__drug_feature_extractor = _get_network(self.__extractors, self.__drug_feature_extractor_network,
                                                     self.__drug_feature_extractor_params)
        in_size = 2 * self.__drug_feature_extractor.output_dim  # usually check your dim

        self.classifier = FCNet(input_size=in_size, fc_layer_dims=fc_layers_dim, output_dim=output_dim, **kwargs)

    def forward(self, batch):
        x = batch
        x = x.type("torch.cuda.LongTensor") if torch.cuda.is_available() else x.type("torch.LongTensor")
        n = x.shape[1]
        drug1, drug2 = x[:, : n // 2], x[:, n // 2:]
        features_drug1, features_drug2 = self.__drug_feature_extractor(drug1), self.__drug_feature_extractor(drug2)
        ddi = torch.cat((features_drug1, features_drug2), 1).squeeze()
        out = self.classifier(ddi)
        return out
