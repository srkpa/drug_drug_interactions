import torch
import networkx as nx
import csv
import inspect
import pandas as pd
import scipy.sparse as sp
from functools import partial
from collections import defaultdict
from itertools import product, combinations
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from ivbase.utils.commons import to_tensor
from .transforms import *

all_transformers_dict = dict(
    seq=sequence_transformer,
    fgp=fingerprints_transformer,
    deepddi=deepddi_transformer,
    dgl=dgl_transformer
)


def load_smiles(fname, download=False, dataset_name="twosides"):
    smiles = {}
    if not download:
        fin = open(fname)
        fin.readline()
        for line in fin:
            content = line.split(",")
            drug_id = content[0] if dataset_name == "twosides" else content[1]
            smile = content[-1].strip("\n")
            smiles[drug_id] = smile
    return smiles


def load_ddis_combinations(fname, header=True, dataset_name="twosides"):
    fn = open(fname)
    if header:
        fn.readline()
    combo2se = defaultdict(list)
    for line in fn:
        content = line.split(",")
        if dataset_name not in ["twosides", "split"]:
            content = content[1:]
        drug1 = content[0]
        drug2 = content[1]
        se = content[-1].strip("\n").split(";")
        combo2se[(drug1, drug2)] = list(set(se))
    return combo2se


def _filter_samples(samples, transformed_smiles_dict=None):
    to_remove = [
        (id_1, id_2) for (id_1, id_2), y in samples.items()
        if (id_1 not in transformed_smiles_dict) or (id_2 not in transformed_smiles_dict)
    ]
    for key in to_remove:
        samples.pop(key)
    return samples


def train_test_valid_split(data, mode='random', test_size=0.25, valid_size=0.25, seed=42):
    assert mode.lower() in ['random', 'leave_drugs_out']
    if mode == 'random':
        train, test = train_test_split(list(data.keys()), test_size=test_size, random_state=seed)
        train, valid = train_test_split(train, test_size=valid_size)
    else:
        drugs = list(set([x1 for (x1, _) in data] + [x2 for (_, x2) in data]))
        drugs = sorted(drugs)

        train_idx, test_idx = train_test_split(drugs, test_size=test_size, random_state=seed)
        train_idx, valid_idx = train_test_split(train_idx, test_size=valid_size, random_state=seed)

        train = set(product(train_idx, repeat=2))
        valid = set(product(train_idx, valid_idx)).union(set(product(valid_idx, train_idx)))
        test = set(product(train_idx, test_idx)).union(set(product(test_idx, train_idx)))

        train = set(data.keys()).intersection(train)
        valid = set(data.keys()).intersection(valid)
        test = set(data.keys()).intersection(test)

    train_data = {k: data[k] for k in train}
    valid_data = {k: data[k] for k in valid}
    test_data = {k: data[k] for k in test}

    train_idx = list(set([x1 for (x1, _) in train_data] + [x2 for (_, x2) in train_data]))
    print(train_idx)
    a = csv.writer(open("valid-mapping.csv", "w"))
    a.writerow(["drug1", "drug2", "intersection"])
    for d1, d2 in valid:
        assert d1 in train_idx or d2 in train_idx
        if d1 in train_idx:
            drug_in_train = d1
        elif d2 in train_idx:
            drug_in_train = d2
        else:
            raise Exception(f'{d1}-{d2} not found')
        a.writerow([d1, d2, drug_in_train])

    b = csv.writer(open("test-mapping.csv", "w"))
    b.writerow(["drug1", "drug2", "intersection"])
    for d1, d2 in test:
        assert d1 in train_idx or d2 in train_idx
        if d1 in train_idx:
            drug_in_train = d1
        elif d2 in train_idx:
            drug_in_train = d2
        else:
            raise Exception(f'{d1}-{d2} not found')
        b.writerow([d1, d2, drug_in_train])

    print("I am done")
    return train_data, valid_data, test_data


class DDIdataset(Dataset):
    def __init__(self, samples, drug_to_smiles, label_vectorizer,
                 build_graph=False, graph_drugs_mapping=None):
        self.samples = [(d1, d2, samples[(d1, d2)]) for (d1, d2) in samples]
        self.drug_to_smiles = drug_to_smiles
        self.labels_vectorizer = label_vectorizer
        self.gpu = torch.cuda.is_available()
        self.has_graph = build_graph
        self.graph_drugs_mapping = graph_drugs_mapping
        if self.has_graph:
            self.build_graph()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        drug1, drug2, label = self.samples[item]
        if self.graph_drugs_mapping is None:
            drug1, drug2 = self.drug_to_smiles[drug1], self.drug_to_smiles[drug2]
        elif self.has_graph:
            drug1, drug2 = np.array([self.graph_drugs_mapping[drug1]]), np.array([self.graph_drugs_mapping[drug2]])
        else:
            if drug1 not in self.graph_drugs_mapping:
                drug1, drug2 = drug2, drug1
            if drug1 not in self.graph_drugs_mapping:
                print("drug2", drug2 in self.graph_drugs_mapping)
                with open('toto.txt', 'w') as fd:
                    fd.writelines([f'{d}\n' for d in self.graph_drugs_mapping])
            assert drug1 in self.graph_drugs_mapping, f"None of those drugs ({drug1}-{drug2}) is in the train, check your train test split"
            drug1 = np.array([self.graph_drugs_mapping[drug1]])
            drug2 = self.drug_to_smiles[drug2]
        target = self.labels_vectorizer.transform([label]).astype(np.float32)[0]
        res = ((to_tensor(drug1, self.gpu), to_tensor(drug2, self.gpu)), to_tensor(target, self.gpu))
        return res

    @property
    def nb_labels(self):
        return len(self.labels_vectorizer.classes_)

    def get_targets(self):
        y = list(zip(*self.samples))[2]
        return to_tensor(self.labels_vectorizer.transform(y).astype(np.float32), self.gpu)

    def build_graph(self):
        graph_drugs = list(set([d1 for (d1, _, _) in self.samples] + [d2 for (_, d2, _) in self.samples]))
        self.graph_drugs_mapping = {node: i for i, node in enumerate(graph_drugs)}
        n_nodes = len(graph_drugs)
        n_edge_labels = len(self.labels_vectorizer.classes_)
        graph_edges = np.zeros((n_nodes, n_nodes, n_edge_labels))
        for (drug1, drug2, label) in self.samples:
            i, j = self.graph_drugs_mapping[drug1], self.graph_drugs_mapping[drug2]
            graph_edges[i, j] = self.labels_vectorizer.transform([label])[0]

        graph_nodes = np.stack([self.drug_to_smiles[drug] for drug in graph_drugs], axis=0)
        self.graph_nodes = to_tensor(graph_nodes, self.gpu)
        self.graph_edges = to_tensor(graph_edges.astype(np.float32), self.gpu)


def get_data_partitions(dataset_name, input_path, transformer, split_mode,
                        seed=None, test_size=0.25, valid_size=0.25, model_name=False):
    # My paths
    all_combo_path = f"{input_path}/{dataset_name}.csv"
    data = load_ddis_combinations(all_combo_path, header=True)

    # Load smiles
    all_drugs_path = f"{input_path}/{dataset_name}-drugs-all.csv"
    drugs2smiles = load_smiles(fname=all_drugs_path, dataset_name=dataset_name)
    drugs, smiles = list(drugs2smiles.keys()), list(drugs2smiles.values())

    # Transformer
    transformer = all_transformers_dict.get(transformer)
    args = inspect.signature(transformer)
    if 'approved_drug' in args.parameters:
        approved_drugs_path = f"{input_path}/approved_drugs.csv"
        approved_drug = pd.read_csv(approved_drugs_path, sep=",").dropna()
        approved_drug = approved_drug['PubChem Canonical Smiles'].values.tolist()
        drugs2smiles = transformer(drugs=drugs, smiles=smiles, approved_drug=approved_drug)
    else:
        drugs2smiles = transformer(drugs=drugs, smiles=smiles)

    data = _filter_samples(data, transformed_smiles_dict=drugs2smiles)
    train_data, valid_data, test_data = train_test_valid_split(data, split_mode, seed=seed,
                                                               test_size=test_size, valid_size=valid_size)

    # a =  list(set([x1 for (x1, _) in train_data] + [x2 for (_, x2) in train_data]))
    # for (d1, d2) in test_data:
    #     print(d1, d2)
    #     assert d1 in a or d2 in a
    exit()
    print(f"len train {len(train_data)}\nlen test_ddi {len(test_data)}\nlen valid {len(valid_data)}")

    labels = list(train_data.values()) + list(test_data.values()) + list(valid_data.values())
    mbl = MultiLabelBinarizer().fit(labels)
    if model_name.lower() == 'bmnddi' and split_mode == 'leave_drugs_out':
        train_dataset = DDIdataset(train_data, drugs2smiles, mbl, build_graph=True)
        valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl, graph_drugs_mapping=train_dataset.graph_drugs_mapping)
        test_dataset = DDIdataset(test_data, drugs2smiles, mbl, graph_drugs_mapping=train_dataset.graph_drugs_mapping)
    elif model_name.lower() == 'decagon':
        train_dataset = DDIdataset(train_data, drugs2smiles, mbl, build_graph=True)
        valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl, graph_drugs_mapping=train_dataset.graph_drugs_mapping)
        test_dataset = DDIdataset(test_data, drugs2smiles, mbl, graph_drugs_mapping=train_dataset.graph_drugs_mapping)
    else:
        train_dataset = DDIdataset(train_data, drugs2smiles, mbl)
        valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl)
        test_dataset = DDIdataset(test_data, drugs2smiles, mbl)

    return train_dataset, valid_dataset, test_dataset


def load_decagon():
    val_test_size = 0.05
    n_genes = 500
    n_drugs = 400
    n_drugdrug_rel_types = 3
    gene_net = nx.planted_partition_graph(50, 10, 0.2, 0.05,
                                          seed=42)  # gene gene

    print("gene net node", len(gene_net.nodes))
    print("gene net edges", len(gene_net.edges))
    gene_adj = nx.adjacency_matrix(gene_net)
    print("gene adj", gene_adj.todense().shape)
    gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
    gene_drug_adj = sp.csr_matrix((10 * np.random.randn(n_genes, n_drugs) > 15).astype(int))  # gene drug
    drug_gene_adj = gene_drug_adj.transpose(copy=True)

    drug_drug_adj_list = []
    tmp = np.dot(drug_gene_adj, gene_drug_adj)
    for i in range(n_drugdrug_rel_types):
        mat = np.zeros((n_drugs, n_drugs))
        for d1, d2 in combinations(list(range(n_drugs)), 2):
            if tmp[d1, d2] == i + 4:  # why?
                mat[d1, d2] = mat[d2, d1] = 1.
        drug_drug_adj_list.append(sp.csr_matrix(mat))
    drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

    # data representation - yep, i understand
    adj_mats_orig = {
        (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
        (0, 1): [gene_drug_adj],
        (1, 0): [drug_gene_adj],
        (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
    }
    degrees = {
        0: [gene_degrees, gene_degrees],
        1: drug_degrees_list + drug_degrees_list,
    }

    # featureless (genes)
    gene_feat = sp.identity(n_genes)
    print("gene feat", gene_feat.shape)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())  # i am here
    # features (drugs)
    drug_feat = sp.identity(n_drugs)
    print("drug feature", drug_feat.shape)
    drug_nonzero_feat, drug_num_feat = drug_feat.shape
    drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

    # data representation
    num_feat = {
        0: gene_num_feat,
        1: drug_num_feat,
    }
    nonzero_feat = {
        0: gene_nonzero_feat,
        1: drug_nonzero_feat,
    }
    feat = {
        0: gene_feat,
        1: drug_feat,
    }

    edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
    print(edge_type2dim)  # ok
    edge_type2decoder = {
        (0, 0): 'bilinear',
        (0, 1): 'bilinear',
        (1, 0): 'bilinear',
        (1, 1): 'dedicom',
    }
    edge_types = {k: len(v) for k, v in adj_mats_orig.items()}  # need to be adapted -yes
    print("edge type", edge_types)
    num_edge_types = sum(edge_types.values())
    print("Edge types:", "%d" % num_edge_types)
    print(adj_mats_orig[(0, 0)][1].shape)
    print([i for i, _ in edge_types])
    exit()