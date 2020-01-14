import csv

import torch
import torch.nn as nn

from .attention import AttentionLayer
from .features_extraction import FeaturesExtractorFactory


class SelfAttentionLayer(AttentionLayer):
    """
    A self-attention block. The only difference is that the key,
    query and value tensors are one and the same.
    """

    def __init__(self, key_dim, pooling_function=None):
        super().__init__(1, 1, key_dim, pooling_function)
        # self.query_network = self.key_network
        # self.value_network = nn.Sequential()

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
    def __init__(self, drug_feature_extractor_params, fc_layers_dim, nb_side_effects=None, mode='concat',
                 att_mode=None, ss_embedding_dim=None,
                 add_feats_params=None, is_binary_output=False, testing=False, exp_prefix=None, **kwargs):

        super(BMNDDI, self).__init__()
        fe_factory = FeaturesExtractorFactory()
        self.mode = mode
        self.att_mode = att_mode
        self.is_binary_output = is_binary_output
        self.testing = testing
        self.exp_prefix = exp_prefix
        self.drug_feature_extractor = fe_factory(**drug_feature_extractor_params)
        in_size = 2 * self.drug_feature_extractor.output_dim
        if self.mode in ['sum', 'max', "elementwise"]:
            in_size = self.drug_feature_extractor.output_dim
        if is_binary_output:
            self.embedding = nn.Embedding(nb_side_effects, ss_embedding_dim)
            in_size += ss_embedding_dim

        if add_feats_params:
            self.add_feature_extractor = fe_factory(arch='fcnet', last_layer_activation=None, **add_feats_params,
                                                    **kwargs)
            in_size += add_feats_params.pop("nb_side_effects")
        output_dim = 1 if is_binary_output else nb_side_effects
        self.classifier = fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                                     output_dim=output_dim, last_layer_activation='Sigmoid', **kwargs)

    def forward(self, batch):
        did1 = did2 = sid = side_eff_features = None
        if not self.is_binary_output:
            drugs_a, drugs_b, = batch[:2]
            add_feats = batch[2:]
        else:
            if self.testing:
                did1, did2, sid, drugs_a, drugs_b, side_eff = batch[:6]
                add_feats = batch[6:]
                if isinstance(side_eff, tuple):
                    side_eff = torch.cat([s.reshape(1) for s in side_eff])
            else:
                drugs_a, drugs_b, side_eff = batch[:3]
                add_feats = batch[3:]
                if isinstance(side_eff, tuple):
                    side_eff = torch.cat(side_eff)
            side_eff_features = self.embedding(side_eff).squeeze()

        features_drug1, features_drug2 = self.drug_feature_extractor(drugs_a), self.drug_feature_extractor(drugs_b)

        if self.mode == "elementwise":
            ddi = torch.mul(features_drug1, features_drug2)
        elif self.mode == "sum":
            ddi = torch.add(features_drug1, features_drug2)
        elif self.mode == "max":
            ddi = torch.max(features_drug1, features_drug2)
        else:
            ddi = torch.cat((features_drug1, features_drug2), 1)

        if add_feats:
            add_feats = self.add_feature_extractor(torch.cat(add_feats, 1))
            ddi = torch.cat((ddi, add_feats), 1)

        if side_eff_features is not None:
            ddi = torch.cat((ddi, side_eff_features), 1)
        out = self.classifier(ddi)

        # Just for binary cases, i want to have all output effortless way
        if did1 is not None and did2 is not None:
            output = list(zip(did1, did2, sid, out.squeeze().tolist()))
            with open(self.exp_prefix, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(output)
        return out

    def set_graph(self, nodes, edges):
        self.nb_nodes = len(nodes)
        self.nodes = nodes
        self.edges = edges
        self.adj_mat = (edges.sum(2) > 0).float()
