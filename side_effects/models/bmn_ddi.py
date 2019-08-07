import torch
import torch.nn as nn
from .features_extraction import FeaturesExtractorFactory
from .attention import AttentionLayer


class SelfAttentionLayer(AttentionLayer):
    """
    A self-attention block. The only difference is that the key,
    query and value tensors are one and the same.
    """

    def __init__(self, key_dim, pooling_function=None):
        super().__init__(1, 1, key_dim, pooling_function)
        self.query_network = self.key_network
        self.value_network = nn.Sequential()

    def forward(self, x):
        """
        Computes the self-attention.

        Parameters
        ----------
        x: torch.Tensor
            This is self attention, which means that x is used as the key, query and value tensors.
        Returns
        -------
        attention: torch.Tensor
            The attention block.
        """
        x = x.reshape(*x.shape, 1)
        return super().forward(x, x, x).squeeze(-1)


class BMNDDI(nn.Module):

    def __init__(self, drug_feature_extractor_params, fc_layers_dim, output_dim,
                 mode='concat', **kwargs):
        super(BMNDDI, self).__init__()
        fe_factory = FeaturesExtractorFactory()
        self.mode = mode
        self.drug_feature_extractor = fe_factory(**drug_feature_extractor_params)

        in_size = 2 * self.drug_feature_extractor.output_dim
        if self.mode in ['sum', 'max', "elementwise"]:
            in_size = self.drug_feature_extractor.output_dim

        self.classifier = fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                                     output_dim=output_dim, last_layer_activation='Sigmoid', **kwargs)

    def forward(self, batch):
        drugs_a, drugs_b = batch
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


class BMNDDI_with_Attention(BMNDDI):

    def __init__(self, drug_feature_extractor_params, fc_layers_dim, output_dim,
                 mode='concat', att_hidden_size=20, **kwargs):
        super().__init__(drug_feature_extractor_params, fc_layers_dim, output_dim, mode, **kwargs)
        fe_factory = FeaturesExtractorFactory()
        self.mode = mode
        self.drug_feature_extractor = fe_factory(**drug_feature_extractor_params)

        in_size = 2 * self.drug_feature_extractor.output_dim
        if self.mode in ['sum', 'max', "elementwise"]:
            in_size = self.drug_feature_extractor.output_dim

        self.classifier = nn.Sequential(
            fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                       output_dim=output_dim, last_layer_activation='Sigmoid', **kwargs),
            SelfAttentionLayer(1, att_hidden_size),
            nn.Sigmoid()
        )
