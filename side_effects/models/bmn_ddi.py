import torch
import torch.nn as nn
from .features_extraction import FeaturesExtractorFactory


class BMNDDI(nn.Module):

    def __init__(self, drug_feature_extractor_params, fc_layers_dim, output_dim, use_gpu, mode='concat', **kwargs):
        super(BMNDDI, self).__init__()
        fe_factory = FeaturesExtractorFactory()
        self.use_gpu = use_gpu
        self.mode = mode
        self.drug_feature_extractor = fe_factory(**drug_feature_extractor_params)

        in_size = 2 * self.drug_feature_extractor.output_dim
        if self.mode in ['sum', 'max', "elementwise"]:
            in_size = self.drug_feature_extractor.output_dim

        self.classifier = fe_factory(arch='fc', input_size=in_size, fc_layer_dims=fc_layers_dim,
                                     output_dim=output_dim, last_layer_activation='Sigmoid', **kwargs)

    def forward(self, batch):
        drugs_a, drugs_b = list(zip(*batch))
        drugs_a, drugs_b = torch.stack(drugs_a), torch.stack(drugs_b)
        if self.use_gpu:
            drugs_a = drugs_a.cuda()
            drugs_b = drugs_b.cuda()
        features_drug1, features_drug2 = self.drug_feature_extractor(drugs_a), self.drug_feature_extractor(drugs_b)
        if self.mode == "elementwise":
            ddi = torch.mul(features_drug1, features_drug2)
        elif self.mode == "sum":
            ddi = torch.add(features_drug1, features_drug2)
        elif self.mode == "max":
            ddi = torch.max(features_drug1, features_drug2)
        else:
            ddi = torch.cat((features_drug1, features_drug2), 1)
        out = self.classifier(ddi)
        return out
