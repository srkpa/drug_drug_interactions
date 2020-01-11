import torch
import torch.nn as nn
from torch.nn import Embedding

from .bmn_ddi import BMNDDI, FeaturesExtractorFactory


class BiBMNDDI(BMNDDI):
    def __init__(self, vocab_size=None, embedding_size=None, dropout=0, b_norm=True, **kwargs):
        super(BiBMNDDI, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_size)
        in_size = 2 * self.drug_feature_extractor.output_dim + embedding_size
        self.classifier = nn.Sequential(
            FeaturesExtractorFactory()(arch='fcnet', input_size=in_size,
                                       fc_layer_dims=kwargs['fc_layers_dim'],
                                       output_dim=2, last_layer_activation=None, dropout=dropout, b_norm=b_norm),
            nn.Softmax()
        )

    def forward(self, batch):
        did1, did2, seidx = batch
        features_drug1, features_drug2, features_side_effect = self.drug_feature_extractor(
            did1), self.drug_feature_extractor(did2), self.embedding(seidx).squeeze()
        ddi = torch.cat((features_drug1, features_drug2, features_side_effect), 1)
        output = self.classifier(ddi)
        return output
