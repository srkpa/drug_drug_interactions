import torch
import torch.nn as nn

from .features_extraction import FeaturesExtractorFactory


class MLTDDI(nn.Module):
    def __init__(self, drug_feature_extractor_params, fc_layers_dim, output_dim, mode='concat', **kwargs):
        super(MLTDDI, self).__init__()
        fe_factory = FeaturesExtractorFactory()
        self.mode = mode
        self.drug_feature_extractor = fe_factory(**drug_feature_extractor_params)
        in_size = 2 * self.drug_feature_extractor.output_dim
        if self.mode in ['sum', 'max', "elementwise"]:
            in_size = self.drug_feature_extractor.output_dim
        self.classifiers = nn.ModuleList([fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                                                     output_dim=1, last_layer_activation='Sigmoid',
                                                     **kwargs)] * output_dim)

    def forward(self, batch):
        drugs_a, drugs_b, = batch[:2]
        features_drug1, features_drug2 = self.drug_feature_extractor(drugs_a), self.drug_feature_extractor(drugs_b)
        if self.mode == "elementwise":
            ddi = torch.mul(features_drug1, features_drug2)
        elif self.mode == "sum":
            ddi = torch.add(features_drug1, features_drug2)
        elif self.mode == "max":
            ddi = torch.max(features_drug1, features_drug2)
        else:
            ddi = torch.cat((features_drug1, features_drug2), 1)
        out = torch.cat([layer(ddi) for layer in self.classifiers[:]], 1)
        return out
