import torch
from torch import nn
from .graphs import GINConv
from .features_extraction import FeaturesExtractorFactory


class BMNDDI(nn.Module):
    def __init__(self, graph_network_params, drug_feature_extractor_params, edges_embedding_dim, fc_layers_dim, output_dim,
                 mode='concat', tied_weights=True, **kwargs):
        super(BMNDDI, self).__init__()
        fe_factory = FeaturesExtractorFactory()
        self.mode = mode
        self.drug_feature_extractor = fe_factory(**drug_feature_extractor_params)

        self.graph_net = GINConv(**graph_network_params)
        if tied_weights:
            self.node_feature_extractor = self.drug_feature_extractor
        else:
            self.node_feature_extractor = fe_factory(**drug_feature_extractor_params)

        self.edge_embedding_layer = nn.Linear(output_dim, edges_embedding_dim, bias=False)

        in_size = 2 * self.drug_feature_extractor.output_dim
        if self.mode in ['sum', 'max', "elementwise"]:
            in_size = self.drug_feature_extractor.output_dim

        self.classifier = fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                                     output_dim=output_dim, last_layer_activation='Sigmoid', **kwargs)

    def forward(self, batch):
        drugs_a, drugs_b = batch
        if self.training:
            mask = torch.ones(self.nb_nodes)
            mask[drugs_b] = 0
            adj_mat = self.adj_mat * mask[None, :] * mask[:, None]
            features_drug2 = self.drug_feature_extractor(torch.index_select(self.nodes, dim=0, index=drugs_b))
        else:
            adj_mat = self.adj_mat
            features_drug2 = self.drug_feature_extractor(drugs_b)

        edges_features = self.edge_embedding_layer(self.adj_edges)
        nodes_features = self.node_feature_extractor(self.nodes)
        _, nodes_features, _ = self.graph_net(adj_mat, nodes_features, edges_features)
        features_drug1 = torch.index_select(nodes_features, dim=0, index=drugs_a)

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

    def set_graph(self, nodes, edges):
        self.nb_nodes = len(nodes)
        self.nodes = nodes
        self.edges = edges
        self.adj_mat = (edges.sum(2) > 0).long()
