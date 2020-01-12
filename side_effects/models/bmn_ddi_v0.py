import torch
import torch.nn as nn

from .attention import AttentionLayer
from .features_extraction import FeaturesExtractorFactory
from .graphs import DGINConv


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


#

class BMNDDI(nn.Module):
    def __init__(self, drug_feature_extractor_params, fc_layers_dim, output_dim, mode='concat',
                 att_mode=None, att_hidden_dim=None, graph_network_params=None, edges_embedding_dim=None,
                 tied_weights=True,
                 additional_feat_net_params=None, **kwargs):
        super(BMNDDI, self).__init__()
        fe_factory = FeaturesExtractorFactory()
        self.mode = mode
        self.att_mode = att_mode
        self.drug_feature_extractor = fe_factory(**drug_feature_extractor_params)
        dfe_out_dim = self.drug_feature_extractor.output_dim
        in_size = 2 * self.drug_feature_extractor.output_dim
        if self.mode in ['sum', 'max', "elementwise"]:
            in_size = self.drug_feature_extractor.output_dim

        if graph_network_params is not None:
            gcnn = DGINConv(node_size=dfe_out_dim, edge_size=edges_embedding_dim,
                            **graph_network_params)
            self.graph_net = nn.Sequential(gcnn, nn.Linear(gcnn.output_dim, dfe_out_dim))
            if tied_weights:
                self.node_feature_extractor = self.drug_feature_extractor
            else:
                self.node_feature_extractor = fe_factory(**drug_feature_extractor_params)

            self.edge_embedding_layer = nn.Linear(output_dim, edges_embedding_dim, bias=False)

        else:
            self.graph_net = None

        if additional_feat_net_params:
            params = additional_feat_net_params
            self.auxnet_feature_extractor = fe_factory(arch='fcnet', input_size=params.get("input_dim"),
                                                       fc_layer_dims=params.get("fc_layers_dim"),
                                                       output_dim=params.get("output_dim"),
                                                       last_layer_activation=None,
                                                       **kwargs)

            in_size += params.get("output_dim")

        if att_hidden_dim is None:
            self.classifier = fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                                         output_dim=output_dim, last_layer_activation='Sigmoid', **kwargs)

        else:
            self.classifier = nn.Sequential(
                fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                           output_dim=output_dim, last_layer_activation='relu', **kwargs),
                SelfAttentionLayer(att_hidden_dim),
                nn.Sigmoid()
            )

    def forward(self, batch):
        drugs_a, drugs_b, = batch[:2]
        add_feats = batch[2:]
        if self.graph_net is None:
            features_drug1, features_drug2 = self.drug_feature_extractor(drugs_a), self.drug_feature_extractor(drugs_b)
        else:

            if self.training:
                drugs_b = drugs_b.view(-1)
                drugs_b = torch.index_select(self.nodes, dim=0, index=drugs_b)
                mask = torch.ones(self.nb_nodes)
                if torch.cuda.is_available():
                    mask = mask.cuda()
                mask[drugs_b] = 0
                adj_mat = self.adj_mat * mask[None, :] * mask[:, None]
            else:
                drugs_b = drugs_b
                adj_mat = self.adj_mat

            features_drug2 = self.drug_feature_extractor(drugs_b)
            edges_features = self.edge_embedding_layer(self.edges)
            nodes_features = self.node_feature_extractor(self.nodes)
            if torch.cuda.device_count() > 1:
                adj_mat = adj_mat.cuda(1)
                nodes_features = nodes_features.cuda(1)
                edges_features = edges_features.cuda(1)
                self.graph_net = self.graph_net.cuda(1)

            nodes_features = self.graph_net((adj_mat, nodes_features, edges_features))
            features_drug1 = torch.index_select(nodes_features.cuda(), dim=0, index=drugs_a.view(-1))

        if self.mode == "elementwise":
            ddi = torch.mul(features_drug1, features_drug2)
        elif self.mode == "sum":
            ddi = torch.add(features_drug1, features_drug2)
        elif self.mode == "max":
            ddi = torch.max(features_drug1, features_drug2)
        elif self.att_mode == "co-att":
            ddi = self.coatt(features_drug1, features_drug2)
        else:
            ddi = torch.cat((features_drug1, features_drug2), 1)

        if add_feats:
            a_ux_net_feat = self.auxnet_feature_extractor(torch.cat(add_feats, 1))
            ddi = torch.cat((ddi, a_ux_net_feat), 1)

        out = self.classifier(ddi)

        return out

    def set_graph(self, nodes, edges):
        self.nb_nodes = len(nodes)
        self.nodes = nodes
        self.edges = edges
        self.adj_mat = (edges.sum(2) > 0).float()

