import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

from side_effects.models.rgcn import utils
from side_effects.models.rgcn.model import BaseRGCN


def reformat_data(train, valid, test):
    samples = train.samples + test.samples + valid.samples
    labels = list(set([l for (_, _, labels) in samples for l in labels]))
    labels.sort()
    labels_mapping = {label: i for i, label in enumerate(labels)}
    graph_drugs = list(set([d1 for (d1, _, _) in samples] + [d2 for (_, d2, _) in samples]))
    graph_nodes_mapping = {node: i for i, node in enumerate(graph_drugs)}
    train = np.array(
        [[graph_nodes_mapping[a], labels_mapping[rel], graph_nodes_mapping[b]] for (a, b, rels) in train.samples for rel
         in rels])
    valid = np.array(
        [[graph_nodes_mapping[a], labels_mapping[rel], graph_nodes_mapping[b]] for (a, b, rels) in valid.samples for rel
         in rels])
    test = np.array(
        [[graph_nodes_mapping[a], i, graph_nodes_mapping[b]] for (a, b, _) in test.samples for i in range(len(labels))])
    num_nodes = len(graph_drugs)
    print("# entities: {}".format(num_nodes))
    num_rels = len(labels_mapping)
    print("# relations: {}".format(num_rels))
    print("# edges: {}".format(len(train)))
    print("# test edges: {}".format(len(test)))

    return num_nodes, num_rels, train, valid, test


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)


class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(num_nodes=in_dim, h_dim=h_dim, out_dim=h_dim, num_rels=num_rels * 2, num_bases=num_bases,
                         num_hidden_layers=num_hidden_layers, dropout=dropout, use_cuda=use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def main(train, test, valid, model_state_file, model_params, fit_params):
    # set parameters
    network_params = model_params["network_params"]
    # load graph data
    num_nodes, num_rels, train_data, valid_data, test_data = reformat_data(train=train, test=test, valid=valid)
    # check cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(torch.cuda.current_device())
    # create model
    model = LinkPredict(in_dim=num_nodes,
                        h_dim=network_params.get("h_dim", 500),
                        num_rels=num_rels,
                        num_bases=network_params.get("n_bases", 100),
                        num_hidden_layers=network_params.get("num_hidden_layers", 2),
                        dropout=network_params.get("dropout", 0.2),
                        use_cuda=use_cuda,
                        reg_param=network_params.get("regularization", 0.01))

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
        range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))
    if use_cuda:
        model.cuda()
    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get("lr", 0.01))
    #  model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []
    # training loop
    print("start training...")
    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        epoch += 1
        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, model_params.get("graph_batch_size", 300000), model_params.get("graph_split_size", 0.5),
                num_rels, adj_list, degrees, model_params.get("negative_sample", 10),
                model_params.get("edge_sampler", "uniform"))
        print("Done edge sampling")
        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
        t0 = time.time()
        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model_params.get("grad_norm", 1.0))  # clip gradients
        optimizer.step()
        t2 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))
        optimizer.zero_grad()
        # validation
        if epoch % fit_params.get("evaluate_every", 500) == 0:
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
            model.eval()
            print("start eval")
            embed = model(test_graph, test_node_id, test_rel, test_norm)
            mrr = utils.calc_mrr(embed, model.w_relation, valid_data,
                                 hits=[1, 3, 10], eval_bz=fit_params.get("eval_batch_size", 500))
            # save best model
            if mrr < best_mrr:
                if epoch >= fit_params.get("n_epochs", 100):
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)
            if use_cuda:
                model.cuda()

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))
    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    if use_cuda:
        model.cpu()  # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(test_graph, test_node_id, test_rel, test_norm)
    scores = model.calc_score(embed, test_data)
    scores = F.sigmoid(scores)
    return test_data, scores
