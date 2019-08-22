from .graphs import GraphConvolutionMulti
from .decoders import *


class DecagonModel(nn.Module):
    def __init__(self, hidden_size, output_dim, dropout=0.):
        super(DecagonModel, self).__init__()
        self.edge_type2decoder = {}
        self.latent_inters = []
        self.latent_varies = []
        self.encoder = nn.Sequential(
            GraphConvolutionMulti(output_dim=hidden_size, dropout=dropout),
            nn.ReLU(),
            GraphConvolutionMulti(input_dim=hidden_size, output_dim=output_dim, dropout=dropout)
        )
        # self.decoder = DEDICOMDecoder(
        #             input_dim=output_dim,
        #             edge_type=(i, j), num_types=self.edge_types[i, j],
        #             act=lambda x: x, dropout=self.dropout)

    def forward(self, batch):
        druga, drugb = batch
        nodes_features = self.drugs[drugb]
        adj_mat = self.adj_mats[drugb, :]
        outputs = self.net((nodes_features, adj_mat))
                 # self.decoder(outputs)

        return outputs

    def set_graph(self, nodes, edges):
        self.drugs, self.genes = nodes
        self.nb_drugs = len(self.drugs)
        self.nb_genes = len(self.drugs)
        self.edges, self.adj_mats = edges
