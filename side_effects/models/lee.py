import numpy as np
import torch
import torch.nn as nn

from .features_extraction import FeaturesExtractorFactory


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_layer_dims, output_dim, **kwargs):
        super(AutoEncoder, self).__init__()
        fe_factory = FeaturesExtractorFactory()
        self.output_dim = output_dim
        self.input_dim = 2 * input_dim
        self.net = nn.Sequential(fe_factory(arch='fcnet', input_size=self.input_dim, fc_layer_dims=hidden_layer_dims,
                                            output_dim=output_dim, last_layer_activation='relu', **kwargs),
                                 fe_factory(arch='fcnet', input_size=output_dim, fc_layer_dims=hidden_layer_dims,
                                            output_dim=self.input_dim, last_layer_activation='Sigmoid', **kwargs))

    def forward(self, x):
        return self.net(x)


class DNN(nn.Module):
    def __init__(self, gen_encoder, drug_encoder, func_encoder, hidden_layer_dims, output_dim, **kwargs):
        super(DNN, self).__init__()
        self.gen_feat_encoder = get_encoder(gen_encoder)
        self.func_feat_encoder = get_encoder(func_encoder)
        self.drug_feat_encoder = get_encoder(drug_encoder)
        in_size = gen_encoder.model.output_dim + drug_encoder.model.output_dim + func_encoder.model.output_dim
        self.classifier = FeaturesExtractorFactory()(arch='fcnet', input_size=in_size, fc_layer_dims=hidden_layer_dims,
                                                     output_dim=output_dim, last_layer_activation='Sigmoid', **kwargs)

    def forward(self, batch):
        x, y = batch
        ssp = torch.cat((x[0], y[0]), 1)
        tsp = torch.cat((x[1], y[1]), 1)
        gsp = torch.cat((x[2], y[2]), 1)
        a = self.drug_feat_encoder.predict(ssp)
        b = self.gen_feat_encoder.predict(tsp)
        c = self.func_feat_encoder.predict(gsp)
        d = np.concatenate((a, b, c), axis=1)
        e = torch.from_numpy(d)
        if torch.cuda.is_available():
            e = e.cuda()
        return self.classifier(e)


def get_encoder(model):
    model.model.net = model.model.net[0]
    model.model.eval()
    return model
