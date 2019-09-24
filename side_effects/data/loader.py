import inspect
from collections import defaultdict
from itertools import product

import pandas as pd
from ivbase.utils.commons import to_tensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from side_effects.data.transforms import *

all_transformers_dict = dict(
    seq=sequence_transformer,
    fgp=fingerprints_transformer,
    deepddi=deepddi_transformer,
    dgl=dgl_transformer,
    adj=dgl_transformer
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


def load_ddis_combinations(fname, header, dataset_name):
    fn = open(fname)
    if header:
        fn.readline()
    combo2se = defaultdict(list)
    for line in fn:
        content = line.split(",")
        if dataset_name != "twosides":
            content = content[1:]
        drug1 = content[0]
        drug2 = content[1]
        se = content[-1].strip("\n").split(";")
        combo2se[(drug1, drug2)] = list(set(se))
    return combo2se


def _filter_pairs_both_exists(samples, filtering_set=None):
    to_remove = [
        (id_1, id_2) for (id_1, id_2) in samples
        if (id_1 not in filtering_set) or (id_2 not in filtering_set)
    ]
    for key in to_remove:
        samples.pop(key)
    return samples


def _filter_pairs_one_exists(samples, filtering_set=None):
    to_remove = [
        (id_1, id_2) for (id_1, id_2) in samples
        if (id_1 not in filtering_set) and (id_2 not in filtering_set)
    ]
    for key in to_remove:
        samples.pop(key)
    return samples


def train_test_valid_split(data, mode='random', test_size=0.25, valid_size=0.25, seed=42):
    assert mode.lower() in ['random', 'leave_drugs_out']
    if mode == 'random':
        train, test = train_test_split(list(data.keys()), test_size=test_size, random_state=seed)
        train, valid = train_test_split(train, test_size=valid_size)

        train_data = {k: data[k] for k in train}
        valid_data = {k: data[k] for k in valid}
        test_data = {k: data[k] for k in test}
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
        train_drugs = list(set([x1 for (x1, _) in train] + [x2 for (_, x2) in train]))

        train_data = {k: data[k] for k in train}
        valid_data = {k: data[k] for k in valid}
        test_data = {k: data[k] for k in test}

        # filter out some pairs due to the intersection
        valid_data = _filter_pairs_one_exists(valid_data, train_drugs)
        test_data = _filter_pairs_one_exists(test_data, train_drugs)

    return train_data, valid_data, test_data


class DDIdataset(Dataset):
    def __init__(self, samples, drug_to_smiles, label_vectorizer,
                 build_graph=False, graph_drugs_mapping=None, gene_net=None, drug_gene_net=None, decagon=False):
        self.samples = [(d1, d2, samples[(d1, d2)]) for (d1, d2) in samples]
        self.drug_to_smiles = drug_to_smiles
        self.labels_vectorizer = label_vectorizer
        self.gpu = torch.cuda.is_available()
        self.has_graph = build_graph
        self.graph_nodes_mapping = graph_drugs_mapping
        self.gene_net = gene_net
        self.drug_gene_net = drug_gene_net
        self.decagon = decagon
        if self.has_graph:
            self.build_graph()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        drug1, drug2, label = self.samples[item]
        if self.graph_nodes_mapping is None:
            drug1, drug2 = self.drug_to_smiles[drug1], self.drug_to_smiles[drug2]
        elif self.has_graph or self.decagon:
            drug1, drug2 = np.array([self.graph_nodes_mapping[drug1]]), np.array([self.graph_nodes_mapping[drug2]])
        else:
            if drug1 not in self.graph_nodes_mapping:
                drug1, drug2 = drug2, drug1
            if drug1 not in self.graph_nodes_mapping:
                with open('toto.txt', 'w') as fd:
                    fd.writelines([f'{d}\n' for d in self.graph_nodes_mapping])
            assert drug1 in self.graph_nodes_mapping, f"None of those drugs ({drug1}-{drug2}) is in the train, check your train test split"
            drug1 = np.array([self.graph_nodes_mapping[drug1]])
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
        self.graph_nodes_mapping = {node: i for i, node in enumerate(graph_drugs)}
        n_nodes = len(graph_drugs)

        n_edge_labels = len(self.labels_vectorizer.classes_)
        graph_edges = np.zeros((n_nodes, n_nodes, n_edge_labels))
        for (drug1, drug2, label) in self.samples:
            i, j = self.graph_nodes_mapping[drug1], self.graph_nodes_mapping[drug2]
            graph_edges[i, j] = self.labels_vectorizer.transform([label])[0]

        graph_nodes = np.stack([self.drug_to_smiles[drug] for drug in graph_drugs], axis=0)

        if all([v is not None for v in [self.gene_net, self.drug_gene_net]]) and self.decagon:
            graph_gene = list(
                set([g1 for (g1, _) in self.gene_net] + [g2 for (_, g2) in self.gene_net] + [g for (_, g) in
                                                                                             self.drug_gene_net]))
            n_drugs = n_nodes
            n_genes = len(graph_gene)
            for i, node in enumerate(graph_gene):
                self.graph_nodes_mapping[node] = n_drugs + i
            n_nodes = n_genes + n_drugs
            adj_mats = np.zeros((n_nodes, n_nodes))
            for (gene1, gene2) in self.gene_net:
                id1, id2 = self.graph_nodes_mapping[gene1], self.graph_nodes_mapping[gene2]
                adj_mats[id1, id2] = adj_mats[id2, id1] = 1

            for (drug, gene) in self.drug_gene_net:
                id1, id2 = self.graph_nodes_mapping[drug], self.graph_nodes_mapping[gene]
                adj_mats[id1, id2] = adj_mats[id2, id1] = 1

            assert not np.any(adj_mats[:n_drugs, :n_drugs])
            drug_drug_adj = graph_edges.sum(2) > 0
            adj_mats[:n_drugs, :n_drugs] = drug_drug_adj + np.transpose(drug_drug_adj)

            # featureless nodes
            drugs_nodes_feat = np.eye(n_drugs)
            gene_nodes_feat = np.eye(n_genes)

            self.graph_nodes = to_tensor(drugs_nodes_feat, self.gpu), to_tensor(gene_nodes_feat, self.gpu)
            self.graph_edges = to_tensor(graph_edges.astype(np.float32), self.gpu), to_tensor(
                adj_mats.astype(np.float32), self.gpu)
        else:
            self.graph_nodes = to_tensor(graph_nodes, self.gpu)
            self.graph_edges = to_tensor(graph_edges.astype(np.float32), self.gpu)


def get_data_partitions(dataset_name, input_path, transformer, split_mode,
                        seed=None, test_size=0.25, valid_size=0.25, use_graph=False, decagon=False,
                        use_side_effects_mapping=False, use_as_filter=None):
    # My paths
    all_combo_path = f"{input_path}/{dataset_name}.csv"
    data = load_ddis_combinations(all_combo_path, header=True, dataset_name=dataset_name)

    if use_side_effects_mapping:
        print(use_as_filter)
        se_path = f"{input_path}/classification - twosides_socc.tsv"
        side_effects_mapping = load_side_effect_mapping(se_path, use_as_filter)
        data = relabel(data, side_effects_mapping)

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

    data = _filter_pairs_both_exists(data, filtering_set=drugs2smiles)
    train_data, valid_data, test_data = train_test_valid_split(data, split_mode, seed=seed,
                                                               test_size=test_size, valid_size=valid_size)
    print(f"len train {len(train_data)}\nlen test_ddi {len(test_data)}\nlen valid {len(valid_data)}")
    labels = list(train_data.values()) + list(test_data.values()) + list(valid_data.values())
    mbl = MultiLabelBinarizer().fit(labels)

    if decagon:
        ppi_path, dgi_path = f"{input_path}/ppi.csv", f"{input_path}/dgi.csv"
        gene_gene_ass = pd.read_csv(ppi_path)
        drug_gene_ass = pd.read_csv(dgi_path)
        gene_gene_ass_list = gene_gene_ass.values.tolist()
        drug_gene_ass_list = drug_gene_ass.values.tolist()
        train_dataset = DDIdataset(train_data, drugs2smiles, mbl, build_graph=False, gene_net=gene_gene_ass_list,
                                   drug_gene_net=drug_gene_ass_list,
                                   decagon=decagon)
        valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl, graph_drugs_mapping=train_dataset.graph_nodes_mapping,
                                   gene_net=train_dataset.gene_net, drug_gene_net=train_dataset.drug_gene_net)
        test_dataset = DDIdataset(test_data, drugs2smiles, mbl, graph_drugs_mapping=train_dataset.graph_nodes_mapping,
                                  gene_net=train_dataset.gene_net, drug_gene_net=train_dataset.drug_gene_net)

    else:
        if use_graph:
            train_dataset = DDIdataset(train_data, drugs2smiles, mbl, build_graph=True)
            valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl,
                                       graph_drugs_mapping=train_dataset.graph_nodes_mapping)
            test_dataset = DDIdataset(test_data, drugs2smiles, mbl,
                                      graph_drugs_mapping=train_dataset.graph_nodes_mapping)
        else:
            train_dataset = DDIdataset(train_data, drugs2smiles, mbl)
            valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl)
            test_dataset = DDIdataset(test_data, drugs2smiles, mbl)
            print(train_dataset.nb_labels)

    return train_dataset, valid_dataset, test_dataset


def load_side_effect_mapping(input_path, use_as_filter=None):
    res = {}
    data = pd.read_csv(input_path, sep="\t")
    data.fillna("Unspecified", inplace=True)
    for row in data.itertuples():
        if use_as_filter == 'SOC':
            res[row.side_effect] = row.system_organ_class
        elif use_as_filter == "HLGT":
            res[row.side_effect] = row.HLGT
        elif use_as_filter == "HLT":
            res[row.side_effect] = row.HLT
        elif use_as_filter == "PT":
            res[row.side_effect] = row.PT
        else:
            res[row.side_effect] = row.side_effect

    print("dataset side effect mapping: ", len(res))
    return res


def relabel(data, labels_mapping):
    final_data = {}
    for (d1, d2), labels in data.items():
        temp = set()
        for label in labels:
            new_label = labels_mapping.get(label, None)
            if (new_label is not None) and not new_label.startswith("Unspecified"):
                temp.add(new_label)
        if len(temp) > 0:
            final_data[(d1, d2)] = list(temp)
    return final_data
