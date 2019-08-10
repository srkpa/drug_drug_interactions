from .graphs import GraphConvolutionMulti
from .decoders import *


class DecagonModel(nn.Module):
    def __init__(self, inputs, num_feat, edge_types, decoders, output_dim, dropout, adj_mats):
        super(DecagonModel, self).__init__()
        self.edge_types = edge_types
        self.num_edge_types = sum(self.edge_types.values())
        self.num_obj_types = max([i for i, _ in self.edge_types]) + 1
        self.decoders = decoders
        self.hidden1_dim = output_dim[0]
        self.hidden2_dim = output_dim[-1]
        self.inputs = inputs
        self.input_dim = num_feat
        self.dropout = dropout
        self.adj_mats = adj_mats
        self.hidden1 = {i: 0 for i, _ in self.edge_types}
        self.embeddings = [None] * self.num_obj_types
        self.embeddings_reltyp = {i: 0 for i, _ in self.edge_types}
        self.edge_type2decoder = {}
        self.latent_inters = []
        self.latent_varies = []

    def forward(self):
        for i, j in self.edge_types:
            self.hidden1[i] += GraphConvolutionMulti(
                input_dim=self.input_dim, output_dim=self.hidden1_dim,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x, dropout=self.dropout)(self.inputs[j])

        for i, hid1 in self.hidden1.items():
            self.hidden1[i] = nn.RELU(hid1)

        for i, j in self.edge_types:
            self.embeddings_reltyp[i] += GraphConvolutionMulti(
                input_dim=self.hidden1_dim, output_dim=self.hidden2_dim,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout)(self.hidden1[j])

        for i, j in self.edge_types:
            decoder = self.decoders[i, j]
            if decoder == 'innerproduct':
                self.edge_type2decoder[i, j] = InnerProductDecoder(
                    input_dim=self.hidden2_dim, edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'distmult':
                self.edge_type2decoder[i, j] = DistMultDecoder(
                    input_dim=self.hidden2_dim,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'bilinear':
                self.edge_type2decoder[i, j] = BilinearDecoder(
                    input_dim=self.hidden2_dim,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'dedicom':
                self.edge_type2decoder[i, j] = DEDICOMDecoder(
                    input_dim=self.hidden2_dim,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            else:
                raise ValueError('Unknown decoder type')

        for edge_type in self.edge_types:
            decoder = self.decoders[edge_type]
            for k in range(self.edge_types[edge_type]):
                if decoder == 'innerproduct':
                    glb = torch.eye(self.hidden2_dim, self.hidden2_dim)
                    loc = torch.eye(self.hidden2_dim, self.hidden2_dim)
                elif decoder == 'distmult':
                    glb = torch.diag(self.edge_type2decoder[edge_type].vars['relation_%d' % k])
                    loc = torch.eye(self.hidden2_dim, self.hidden2_dim)
                elif decoder == 'bilinear':
                    glb = self.edge_type2decoder[edge_type].vars['relation_%d' % k]
                    loc = torch.eye(self.hidden2_dim, self.hidden2_dim)
                elif decoder == 'dedicom':
                    glb = self.edge_type2decoder[edge_type].vars['global_interaction']
                    loc = torch.diag(self.edge_type2decoder[edge_type].vars['local_variation_%d' % k])
                else:
                    raise ValueError('Unknown decoder type')

                self.latent_inters.append(glb)
                self.latent_varies.append(loc)
