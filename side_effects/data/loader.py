import inspect
import json
import random
from itertools import product, chain

import dgl
from ivbase.utils.commons import to_tensor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset

from side_effects.data.transforms import *
from side_effects.metrics import wrapped_partial

all_transformers_dict = dict(
    seq=sequence_transformer,
    fgp=fingerprints_transformer,
    deepddi=deepddi_transformer,
    dgl=dgl_graph_transformer,
    adj=adj_graph_transformer,
    lee=lee_et_al_transformer
)

TO_REMOVE = ["animal bite", "abnormal laboratory findings", "adverse drug effect", "drug withdrawal",
             "drug addiction", "alcohol consumption", "abuse", "substance abuse", "drug toxicity nos",
             "alcoholic intoxication, road traffic accident"]


def compute_labels_density(y, return_freq=False):
    n_samples, n_labels = y.shape
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    one_w = np.sum(y, axis=0)
    if return_freq:
        return 1 / torch.from_numpy(one_w / n_samples)
    zero_w = y.shape[0] - one_w
    w = np.maximum(one_w, zero_w) / np.minimum(one_w, zero_w)
    return torch.from_numpy(w)


def compute_classes_weight(y, use_exp=False, exp=1):
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    weights = np.array([compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=target)
                        if len(np.unique(target)) > 1 else np.array([1.0, 1.0]) for target in y.T], dtype=np.float32)
    weights = torch.from_numpy(weights).t()
    if use_exp:
        weights = exp * weights
        return torch.exp(weights)
    return weights


def load_smiles(fname, download=False):
    smiles = {}
    idx2name = {}
    if not download:
        fin = open(fname)
        fin.readline()
        for line in fin:
            content = line.split(",")
            drug_id = content[
                0]  # if (dataset_name.startswith("twosides") or dataset_name.endswith("food")) else content[1]
            name = content[1].lower()
            smile = content[-1].strip("\n")
            smiles[drug_id] = smile
            idx2name[drug_id] = name
    return smiles, idx2name


def load_ddis_combinations(fname, header):
    fn = open(fname)
    if header:
        fn.readline()
    combo2se = defaultdict(list)
    for line in fn:
        content = line.split(",")
        # if not dataset_name.startswith("twosides"):
        #     content = content[1:]
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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_all_cross_validation(data, n_folds=5, test_fold=1, seed=42):
    valid_fold = random.choice([i for i in range(n_folds + 1) if i != test_fold])
    print("Validation fold = ", valid_fold)
    drugs = list(set([x1 for (x1, _) in data] + [x2 for (_, x2) in data]))
    drugs = sorted(drugs)
    n_folds = len(drugs) // n_folds
    drugs_idx = list(chunks(drugs, n_folds))
    folds = []
    for i, fold in enumerate(drugs_idx):
        idx = [drug_id for f in drugs_idx[:i] for drug_id in f]
        partition = set(product(fold, idx)).union(set(product(idx, fold))).union(set(product(fold, repeat=2)))
        partition = set(data.keys()).intersection(partition)
        print(i, len(partition))
        folds.append(partition)
    np_pairs = sum([len(f) for f in folds])
    assert len(data) == np_pairs, "We have a problem"
    test_data = folds.pop(test_fold)
    valid_data = folds.pop(valid_fold - 1) if valid_fold > test_fold else folds.pop(valid_fold)
    train_data = list(chain.from_iterable(folds))
    # train_data, valid_data = train_test_split(train_data, test_size=0.15, random_state=seed)
    train_data = {k: data[k] for k in train_data}
    test_data = {k: data[k] for k in test_data}
    valid_data = {k: data[k] for k in valid_data}
    print(len(train_data), len(test_data), len(valid_data))
    return train_data, valid_data, test_data


def get_cv_partitions(samples, n_folds, seed, shuffle=False):
    k_folds = KFold(n_splits=n_folds, random_state=seed)
    folds = defaultdict(tuple)
    total = []
    print("SAMPLES: ", len(samples))
    i = 0
    for train_index, test_index in k_folds.split(samples):
        train_index, valid_index = train_test_split(train_index, test_size=(1 / n_folds) + .05, random_state=seed)
        print("FOLD: ", i, "TRAIN:", len(train_index), "TEST:", len(test_index), "VALID:", len(valid_index))
        train, valid, test = itemgetter(*train_index)(samples), itemgetter(*valid_index)(samples), itemgetter(
            *test_index)(samples)
        folds[i] = train, valid, test
        total.extend(test_index)
        i += 1
    assert len(set(total)) == len(samples), "Problem 1: All the dataset will not be tested"
    return folds


def train_test_valid_split(data, mode='random', test_size=0.25, valid_size=0.25, seed=42, n_folds=0, test_fold=0):
    assert mode.lower() in ['random', 'leave_drugs_out']
    unseen_data = {}
    if mode == 'random':
        if n_folds > 1:
            samples = sorted(list(data.keys()))
            cv_partitions = get_cv_partitions(samples, n_folds, seed, False)
            train, valid, test = cv_partitions[test_fold]
        else:
            train, test = train_test_split(list(data.keys()), test_size=test_size, random_state=seed)
            train, valid = train_test_split(train, test_size=valid_size)

        train_data = {k: data[k] for k in train}
        valid_data = {k: data[k] for k in valid}
        test_data = {k: data[k] for k in test}
    else:
        cv = False
        drugs = list(set([x1 for (x1, _) in data] + [x2 for (_, x2) in data]))
        drugs = sorted(drugs)
        if n_folds > 1:
            cv = True
            cv_partitions = get_cv_partitions(drugs, n_folds, seed, False)
            train_idx, valid_idx, test_idx = cv_partitions[test_fold]
            valid = set(product(valid_idx, repeat=2)).union(set(product(train_idx, valid_idx))).union(
                set(product(valid_idx, train_idx)))
            test = set(product(test_idx, repeat=2)).union(set(product(train_idx, test_idx))).union(
                set(product(test_idx, train_idx)))
            # return split_all_cross_validation(data=data, n_folds=n_folds, test_fold=test_fold, seed=42)
        else:
            train_idx, test_idx = train_test_split(drugs, test_size=test_size, random_state=seed)
            train_idx, valid_idx = train_test_split(train_idx, test_size=valid_size, random_state=seed)
            valid = set(product(train_idx, valid_idx)).union(set(product(valid_idx, train_idx)))
            test = set(product(train_idx, test_idx)).union(set(product(test_idx, train_idx)))
            unseen = set(data.keys()).intersection(set(product(test_idx, repeat=2)))
            unseen_data = {k: data[k] for k in unseen}
        train = set(product(train_idx, repeat=2))
        train = set(data.keys()).intersection(train)
        valid = set(data.keys()).intersection(valid)
        test = set(data.keys()).intersection(test)

        train_drugs = list(set([x1 for (x1, _) in train] + [x2 for (_, x2) in train]))
        train_data = {k: data[k] for k in train}
        valid_data = {k: data[k] for k in valid}
        test_data = {k: data[k] for k in test}

        if not cv:
            # filter out some pairs due to the intersection
            valid_data = _filter_pairs_one_exists(valid_data, train_drugs)
            test_data = _filter_pairs_one_exists(test_data, train_drugs)
    return train_data, valid_data, test_data, unseen_data


class MTKDataset(Dataset):
    def __init__(self, args):
        self.args = args

    def __len__(self):
        return min([len(dataset) for dataset in self.args])

    def __getitem__(self, item):
        pass


class DDIdataset(Dataset):
    def __init__(self, samples, drug_to_smiles, label_vectorizer,
                 build_graph=False, graph_drugs_mapping=None, gene_net=None,
                 drug_gene_net=None, decagon=False,
                 drugse=None, drugstargets=None, drugspharm=None, purpose="feed-forward",
                 negative_sampling=False, density=False):

        self.samples = [(d1, d2, samples[(d1, d2)]) for (d1, d2) in samples]
        self.drug_to_smiles = drug_to_smiles
        self.data_type = all([isinstance(v, (torch.Tensor, np.ndarray)) for v in list(drug_to_smiles.values())])
        self.drugs_targets = drugstargets
        self.drugse = drugse
        self.drugspharm = drugspharm
        self.labels_vectorizer = label_vectorizer
        self.gpu = torch.cuda.is_available()
        self.has_graph = build_graph
        self.graph_nodes_mapping = graph_drugs_mapping
        self.gene_net = gene_net
        self.drug_gene_net = drug_gene_net
        self.decagon = decagon
        self.has_purpose = purpose
        self.neg_sampling = negative_sampling
        self.sampler = density
        if self.has_graph:
            self.build_graph()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        drug1_id, drug2_id, label = self.samples[item]
        if self.labels_vectorizer is not None:
            target = self.labels_vectorizer.transform([label]).astype(np.float32)[0]
        else:
            target = label
        if self.graph_nodes_mapping is None:
            drug1, drug2 = self.drug_to_smiles[drug1_id], self.drug_to_smiles[drug2_id]
        elif self.has_graph or self.decagon:
            drug1, drug2 = np.array([self.graph_nodes_mapping[drug1_id]]), np.array(
                [self.graph_nodes_mapping[drug2_id]])
        else:
            if drug1_id not in self.graph_nodes_mapping:
                drug1_id, drug2_id = drug2_id, drug1_id
            if drug1_id not in self.graph_nodes_mapping:
                with open('toto.txt', 'w') as fd:
                    fd.writelines([f'{d}\n' for d in self.graph_nodes_mapping])
            assert drug1_id in self.graph_nodes_mapping, f"None of those drugs ({drug1_id}-{drug2_id}) is in the train, check your train test split"
            drug1 = np.array([self.graph_nodes_mapping[drug1_id]])
            drug2 = self.drug_to_smiles[drug2_id]
        # Additionnal features added
        res = None
        if self.drugspharm:
            res = (to_tensor(drug1, self.gpu), to_tensor(drug2, self.gpu),
                   to_tensor(self.drugspharm[(drug1_id, drug2_id)], self.gpu), to_tensor(target, self.gpu))
        elif self.drugse:
            res = (to_tensor(drug1, self.gpu), to_tensor(drug2, self.gpu),
                   to_tensor(self.drugse[drug1_id], self.gpu), to_tensor(self.drugse[drug2_id], self.gpu)), to_tensor(
                target,
                self.gpu)
        elif self.drugs_targets:
            res = (to_tensor(drug1, self.gpu), to_tensor(drug2, self.gpu),
                   to_tensor(self.drugs_targets[drug1_id], self.gpu),
                   to_tensor(self.drugs_targets[drug2_id], self.gpu)), to_tensor(target, self.gpu)
        elif not self.data_type:
            print(1)
            if not (isinstance(drug1, dgl.DGLGraph) and isinstance(drug2, dgl.DGLGraph)):
                print(11)
                if isinstance(drug1, tuple) and isinstance(drug2, tuple) and all(
                        [isinstance(x, (torch.Tensor, np.ndarray)) for x in drug1 + drug2]):
                    print(111)
                    drug1 = tuple(map(wrapped_partial(to_tensor, gpu=self.gpu), drug1))
                    drug2 = tuple(map(wrapped_partial(to_tensor, gpu=self.gpu), drug2))
                    target = to_tensor(np.expand_dims(target, axis=0), self.gpu)
                    res = ((drug1, drug2), to_tensor(target, self.gpu))

            else:
                print(2)
                res = ((drug1, drug2), to_tensor(np.expand_dims(target, axis=0), self.gpu))
        else:

            res = ((to_tensor(drug1, gpu=self.gpu), to_tensor(drug2, self.gpu)),
                   to_tensor(target, self.gpu))
        # if self.has_purpose == "autoencoder":
        #     res = 2 * (to_tensor(torch.cat(res[0]), gpu=self.gpu),)
        return res

    def set_input_dim(self):
        adme_dim = list(self.drugspharm.values())[0].shape[0] if self.drugspharm else 0
        offside_dim = 2 * list(self.drugse.values())[0].shape[0] if self.drugse else 0
        target_dim = 2 * list(self.drugs_targets.values())[0].shape[0] if self.drugs_targets else 0
        return max({adme_dim, offside_dim, target_dim})

    @property
    def nb_labels(self):
        if self.labels_vectorizer is None:
            return len(list(zip(*self.samples))[2][0])
        return len(self.labels_vectorizer.classes_)

    def get_samples(self):
        samples = [(np.expand_dims(self.drug_to_smiles[d2], axis=0), np.expand_dims(self.drug_to_smiles[d2], axis=0))
                   for d1, d2, _ in self.samples]
        return samples

    def get_targets(self):
        if self.labels_vectorizer is None:
            return np.stack(list(zip(*self.samples))[2], axis=0)
        return to_tensor(self.labels_vectorizer.transform(list(zip(*self.samples))[2]).astype(np.float32),
                         False)

    def build_graph(self):
        graph_drugs = list(set([d1 for (d1, _, _) in self.samples] + [d2 for (_, d2, _) in self.samples]))
        self.graph_nodes_mapping = {node: i for i, node in enumerate(graph_drugs)}
        n_nodes = len(graph_drugs)

        n_edge_labels = len(self.labels_vectorizer.classes_)
        graph_edges = np.zeros((n_nodes, n_nodes, n_edge_labels))  #
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


class BinaryClassDataset(DDIdataset):
    # Need to be cleaned
    def __init__(self, side_effects_idx_dict, criteria, read_as_triplets, **kwargs):
        super(BinaryClassDataset, self).__init__(**kwargs)
        self.criteria = criteria
        self.drug_pairs, self.se_pos_dps, self.se_neg_dps = [], [], []
        self.prepare_feeding_insts()
        self.idx_side_effects_dict = {idx: sid for sid, idx in side_effects_idx_dict.items()}
        self.side_effects_idx_dict = side_effects_idx_dict
        self.use_binary_labels = (side_effects_idx_dict is not None) and len(self.samples) > 0
        self.testing = (len(self.samples) > 0) and read_as_triplets
        self.targets = []
        if self.testing:
            self.feeding_insts = []
            self.prepare_test_feeding_insts()

    def __len__(self):
        return len(self.samples) if not self.testing else len(self.samples) * len(
            self.labels_vectorizer.classes_)

    def __getitem__(self, item):
        if self.testing:
            drug1_id, drug2_id, seidx, target = self.feeding_insts[item]
            sid = self.idx_side_effects_dict[seidx.item()]
            drug1, drug2 = self.drug_to_smiles[drug1_id], self.drug_to_smiles[drug2_id]
            if not (isinstance(drug1, dgl.DGLGraph) and isinstance(drug2, dgl.DGLGraph)):
                if isinstance(drug1, tuple) and isinstance(drug2, tuple) and all(
                        [isinstance(x, (torch.Tensor, np.ndarray)) for x in drug1 + drug2]):
                    drug1 = tuple(map(wrapped_partial(to_tensor, gpu=self.gpu), drug1))
                    drug2 = tuple(map(wrapped_partial(to_tensor, gpu=self.gpu), drug2))
                else:
                    drug1 = to_tensor(drug1, self.gpu)
                    drug2 = to_tensor(drug2, self.gpu)
            return (
                       drug1_id, drug2_id, sid, drug1, drug2, seidx), to_tensor(target, self.gpu)
        drug1_id, drug2_id = self.drug_pairs[item]
        drug1, drug2 = self.drug_to_smiles[drug1_id], self.drug_to_smiles[drug2_id]
        if not (isinstance(drug1, dgl.DGLGraph) and isinstance(drug2, dgl.DGLGraph)):
            if isinstance(drug1, tuple) and isinstance(drug2, tuple) and all(
                    [isinstance(x, (torch.Tensor, np.ndarray)) for x in drug1 + drug2]):
                drug1 = tuple(map(wrapped_partial(to_tensor, gpu=self.gpu), drug1))
                drug2 = tuple(map(wrapped_partial(to_tensor, gpu=self.gpu), drug2))
            else:
                drug1 = to_tensor(drug1, self.gpu)
                drug2 = to_tensor(drug2, self.gpu)
        prob = np.random.uniform()
        chosen_sid, y = (self.se_pos_dps[item],
                         np.array([1]).astype(np.float32)) if prob >= self.criteria else (
            self.se_neg_dps[item], np.array([0]).astype(np.float32))

        return (
                   drug1, drug2,
                   np.random.choice(chosen_sid, 1)), to_tensor(y, gpu=self.gpu)

    def prepare_feeding_insts(self):
        did1, did2, pos_dps = list(zip(*self.samples))
        drug_pairs = list(zip(did1, did2))
        neg_dps = list(map(lambda x: set(self.side_effects_idx_dict.keys()) - set(x), pos_dps))
        neg_dps = list(map(lambda x: [self.side_effects_idx_dict.get(seidx, None) for seidx in x], neg_dps))
        pos_dps = list(map(lambda x: [self.side_effects_idx_dict.get(seidx, None) for seidx in x], pos_dps))
        self.drug_pairs = drug_pairs
        self.se_pos_dps = pos_dps
        self.se_neg_dps = neg_dps

    def prepare_test_feeding_insts(self):
        mbl = to_tensor(self.labels_vectorizer.transform(list(zip(*self.samples))[2]).astype(np.float32),
                        self.gpu)
        pos_insts = [(*self.samples[item][:2], seidx, np.array([1]).astype(np.float32)) for (item, seidx) in
                     (mbl == 1).nonzero()]
        neg_insts = [(*self.samples[item][:2], seidx, np.array([0]).astype(np.float32)) for (item, seidx) in
                     (mbl == 0).nonzero()]
        self.feeding_insts = pos_insts + neg_insts
        print("test feeding ex len", len(pos_insts), len(neg_insts), len(self.feeding_insts))
        self.targets = np.array([1] * len(pos_insts) + [0] * len(neg_insts))
        print("target shape", self.targets.shape)

    def get_targets(self):
        if self.labels_vectorizer is None:
            targets = np.concatenate(list(zip(*self.samples))[2])
            return to_tensor(targets, self.gpu)
            # if not self.use_binary_labels else to_tensor(self.targets, False)


class MultitaskDDIDataset(Dataset):
    def __init__(self, task_data_lst, smi, is_ordered=False, init=False, nb_labels=None, testing=False, **kwargs):
        super().__init__()
        self.nb_tasks = len(task_data_lst)
        self.is_ordered = is_ordered
        self.raw_data = task_data_lst
        self.gpu = torch.cuda.is_available()
        self.drugs2smi = smi
        self.min_len = min([len(task) for task in task_data_lst])
        self.max_len = max([len(task) for task in task_data_lst])
        self.target_sizes = [len(list(task.values())[0]) if len(task) > 0 else 1 for task in task_data_lst]
        self.is_merged = init
        self.nb_labels = nb_labels
        self.testing = testing
        self.masks = None
        self.is_graph = kwargs.get("graph", False)
        if self.min_len != 0:
            self.ratios = [round(len(task) / self.min_len) for task in task_data_lst]
            print(self.ratios)

        if init:
            self.samples = self.init_samples(task_data_lst)
        else:
            self.batch_generator = generator_fn

        if self.is_graph:
            self.collate_fn = self.collate_drug_pairs

        # else:
        #     if self.min_len * max(self.ratios) > self.max_len:
        #         diff = (self.min_len * max(self.ratios)) - self.max_len
        #         self.fill_with_nan(diff, 1)
        #         self.max_len += diff
        #
        #     elif self.min_len * max(self.ratios) < self.max_len:
        #         diff = round(self.max_len / max(self.ratios)) - self.min_len #- (self.min_len * max(self.ratios)) + self.max_len
        #         self.fill_with_nan(diff, 0)
        #         self.min_len += diff

        if testing:
            self.get_targets()

    # print(self.target_sizes)

    def __len__(self):
        if self.is_merged:
            return len(self.samples)

        return self.min_len

    def init_samples(self, task_samples_list):
        if not self.is_ordered:
            drug_pairs = set([input for task in task_samples_list for input, _ in task.items()])
            tmp = [x for tk in task_samples_list[1:] for x in tk.keys()]
            dup_pairs = set([input[::-1] for input in drug_pairs if input[::-1] in tmp])
            print("Len of dup. pairs-->", len(dup_pairs))
            drug_pairs -= dup_pairs

        else:
            for dp in task_samples_list[0].keys():

                if dp[::-1] in task_samples_list[0]:
                    if dp not in task_samples_list[-1] and dp[::-1] in task_samples_list[-1]:
                        print("First case")
                        task_samples_list[-1][dp] = task_samples_list[-1][dp[::-1]]
                else:
                    print("2nd case....")
                    if dp not in task_samples_list[-1] and dp[::-1] in task_samples_list[-1]:
                        task_samples_list[-1][dp] = task_samples_list[-1][dp[::-1]]
                        del (task_samples_list[-1][::-1])

            drug_pairs = set([input for task in task_samples_list for input, _ in task.items()])

        return list(drug_pairs)

    def __getitem__(self, item):
        if self.is_merged:
            drug_1, drug_2 = self.samples[item]
            labels = []
            for i in range(self.nb_tasks):
                label = self.raw_data[i].get((drug_1, drug_2), [np.nan] * self.target_sizes[i])
                label = torch.tensor(label)
                if (torch.isnan(label).sum() > 0) and not self.is_ordered:
                    label = self.raw_data[i].get((drug_2, drug_1), torch.tensor([np.nan] * self.target_sizes[i]))
                # if (torch.isnan(label).sum() == 0):
                label = to_tensor(label, gpu=self.gpu)
                labels += [label]
            drug_1, drug_2 = self.drugs2smi[drug_1], self.drugs2smi[drug_2]

            if isinstance(drug_1, dgl.DGLGraph) and isinstance(drug_2, dgl.DGLGraph):
                return (drug_1, drug_2, labels)

            return (to_tensor(drug_1, gpu=self.gpu), to_tensor(drug_2, gpu=self.gpu)), labels
        else:
            x, y = [], []
            # print("it", item)
            for i in range(self.nb_tasks):
                start = min(item * self.ratios[i], len(self.raw_data[i]))
                end = min(start + self.ratios[i], len(self.raw_data[i]))
                slice = list(self.raw_data[i].items())[start:end]
                # print(i, start, end, len(slice))

                smi_a = [self.drugs2smi.get(drug_1, np.array([0] * self.set_input_dim())) for ((drug_1, _), _) in
                         slice]
                smi_b = [self.drugs2smi.get(drug_2, np.array([0] * self.set_input_dim())) for ((drug_1, drug_2), _)
                         in
                         slice]

                xa = np.concatenate([np.expand_dims(smi.astype(np.int64), 0) for smi in
                                     smi_a], axis=0)
                xb = np.concatenate([np.expand_dims(smi, 0).astype(np.int64)
                                     for smi in smi_b], axis=0)
                yy = np.concatenate([np.expand_dims(lb, axis=0) for (_, _), lb in
                                     slice]).astype(np.float32)

                labels = to_tensor(yy, gpu=self.gpu)
                xx = to_tensor(xa, gpu=self.gpu), to_tensor(xb, gpu=self.gpu)
                x.append(xx)
                y.append(labels)
            return x, y

    def fill_with_nan(self, diff, index):
        print("I am diff", diff, index)
        yy = np.empty((self.target_sizes[index]))
        yy[:] = np.nan
        for i in range(diff):
            xa, xb = f"OUT_OF_{i}_1", f"OUT_OF_{i}_2"
            self.raw_data[index][(xa, xb)] = yy.astype(np.float32)

        print("Final fill", len(self.raw_data[index]))

    def set_input_dim(self):
        return max([len(l) for l in self.drugs2smi.values()])

    def get_targets(self):
        self.gpu = False
        if self.is_merged:
            y = [self.__getitem__(j)[-1] for j, _ in enumerate(self.samples)]
            yy = [list(x) for x in zip(*y)]
            true_y = [np.concatenate([np.expand_dims(l, axis=0) for l in k]) for k in yy]
            # yy = [[b for b in lb if (torch.isnan(b).sum() == 0)] for lb in yy]
        # targets = [np.concatenate([np.expand_dims(l, axis=0) for l in k]) for k in yy]

        else:
            true_y = [np.concatenate([np.expand_dims(lb, axis=0) for lb in
                                      list(task.values())]) for task in self.raw_data]

        if self.masks is None:
            masks = [(y[:, 0] == y[:, 0]) for y in true_y]
            self.masks = masks
            print("m", sum(masks[0]), sum(masks[1]))

        targets = [true_y[i][mask, :] for i, mask in enumerate(self.masks)]
        assert len(targets) == self.nb_tasks, "Ewww"

        print("targets", targets[0].shape, targets[1].shape)

        return targets

    def collate_drug_pairs(self, batch):
        drug_1, drug_2, labels = list(zip(*batch))
        label1, label2 = list(zip(*labels))
        label1, label2 = torch.stack(label1), torch.stack(label2)
        return (drug_1, drug_2), (label1, label2)


def _rank(data, v=100):
    from collections import Counter
    freq = [e for l in data.values() for e in l]
    freqs = dict(Counter(freq))
    # newA = dict(sorted(freqs.items(), key=itemgetter(1), reverse=True)[:v])
    newA = {x: y for x, y in freqs.items() if y >= v}
    data = {pair: set(labels).intersection(list(newA.keys())) for pair, labels in
            data.items()}

    return data


def _mol_repr(input_path, drugs2smiles, transformer_fn):
    drugs, smiles = list(drugs2smiles.keys()), list(drugs2smiles.values())
    transformer = all_transformers_dict.get(transformer_fn)
    args = inspect.signature(transformer)
    if 'approved_drug' in args.parameters:
        approved_drugs_path = f"{input_path}/approved_drugs.csv"
        approved_drug = pd.read_csv(approved_drugs_path, sep=",").dropna()
        approved_drug = approved_drug['PubChem Canonical Smiles'].values.tolist()
        drugs2smiles = transformer(drugs=drugs, smiles=smiles, approved_drug=approved_drug)
    else:
        drugs2smiles = transformer(drugs=drugs, smiles=smiles)
    return drugs2smiles


def load_data(input_path, dataset_name, use_side_effect=False, use_targets=False, use_pharm=False,
              use_clusters=False, remove_syn=False, use_as_filter="", drugname_as_id=False):
    if dataset_name == "mixed":
        data, drugs2smiles = join_datasets()
    else:
        all_combo_path = f"{input_path}/{dataset_name}/{dataset_name}.csv"
        all_drugs_path = f"{input_path}/{dataset_name}/{dataset_name}-drugs-all.csv"

        drugs2smiles, drugs2name = load_smiles(fname=all_drugs_path)
        assert len(drugs2smiles) == len(drugs2name), "Mismatch.."
        data = load_ddis_combinations(all_combo_path, header=True)
        if dataset_name == "L1000":
            data = {(d1, d2): [float(x) / 100 for x in y] for (d1, d2), y in data.items()}
        if drugname_as_id:
            data = {(drugs2name[d1], drugs2name[d2]): val for ((d1, d2), val) in data.items() if
                    d1 in drugs2name and d2 in drugs2name}
            drugs2smiles = {drugs2name[idx]: smi for idx, smi in drugs2smiles.items()}
    drug_offsides = {}
    if use_side_effect:
        drug2se = load_mono_se()
        data = {pair: set(labels).difference(drug2se[pair[0]].union(drug2se[pair[1]])) for pair, labels in
                data.items()}
        drug_offsides = dict(
            zip(drug2se.keys(), MultiLabelBinarizer().fit_transform(drug2se.values()).astype(np.float32)))
    drug2targets = {}
    if use_targets:
        targets = download_drug_gene_targets()
        drug2targets = gene_entity_transformer(targets.keys(), targets.values())
        print(list(drug2targets.values())[0].shape)
    drugs2pharm = {}
    if use_pharm:
        drugs2pharm = load_pharm()
        drugs2pharm = dict(
            zip(drugs2pharm.keys(), MultiLabelBinarizer().fit_transform(drugs2pharm.values()).astype(np.float32)))
    if use_clusters:
        side_effects_mapping = load_side_effect_mapping(input_path, use_as_filter)
        data = relabel(data, side_effects_mapping)
    if remove_syn:
        data = group_by_concept(input_path, data)

    return drugs2smiles, data, (drug_offsides, drug2targets, drugs2pharm)


def get_data_partitions(dataset_name, input_path, transformer, split_mode,
                        seed=None, test_size=0.25, valid_size=0.25, use_graph=False, decagon=False,
                        use_clusters=False, use_as_filter=None, use_targets=False, use_side_effect=False,
                        use_pharm=False, n_folds=0, test_fold=0, label='ml', debug=False, density=False,
                        remove_syn=False, **kwargs):
    drugs2smiles, data, (
        drug_offsides, drug2targets, drugs2pharm) = load_data(input_path, dataset_name, use_side_effect, use_targets,
                                                              use_pharm, use_clusters, remove_syn, use_as_filter)
    drugs2smiles = _mol_repr(input_path, drugs2smiles, transformer)
    data = _filter_pairs_both_exists(data, filtering_set=drugs2smiles)
    train_data, valid_data, test_data, unseen_data = train_test_valid_split(data, split_mode, seed=seed,
                                                                            test_size=test_size, valid_size=valid_size,
                                                                            n_folds=n_folds, test_fold=test_fold)
    # using randomized smiles

    labels = list(train_data.values()) + list(test_data.values()) + list(valid_data.values())

    mbl = MultiLabelBinarizer().fit(labels) if dataset_name in ["twosides", "drugbank"] else None

    # labels_mapping = {label: i for i, label in enumerate(mbl.classes_)}
    if debug:
        train_data = dict(sorted(train_data.items())[:53])
        valid_data = dict(sorted(valid_data.items())[:53])
        test_data = dict(sorted(test_data.items())[:53])
        unseen_data = dict(sorted(unseen_data.items())[:53])
    print(
        f"Dataset:\nlen train: {len(train_data)}\nlen test_ddi: {len(test_data)}\nlen valid: {len(valid_data)}\nlen unseen: {len(unseen_data)}")

    train_dataset = DDIdataset(train_data, drugs2smiles, mbl, drugspharm=drugs2pharm, drugstargets=drug2targets,
                               drugse=drug_offsides, density=density)
    valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl, drugspharm=drugs2pharm, drugstargets=drug2targets,
                               drugse=drug_offsides, density=density)
    test_dataset = DDIdataset(test_data, drugs2smiles, mbl, drugspharm=drugs2pharm, drugstargets=drug2targets,
                              drugse=drug_offsides, density=density)
    print(train_dataset.nb_labels)
    unseen_dataset = DDIdataset(unseen_data, drugs2smiles, mbl, drugspharm=drugs2pharm,
                                drugstargets=drug2targets,
                                drugse=drug_offsides, density=density)

    return train_dataset, valid_dataset, test_dataset, unseen_dataset


def get_multi_data_partitions(dataset_name, input_path, transformer, split_mode,
                              seed=None, test_size=0.25, valid_size=0.25, use_graph=False, decagon=False,
                              use_clusters=False, use_as_filter=None, use_targets=False, use_side_effect=False,
                              use_pharm=False, n_folds=0, test_fold=0, label='ml', debug=False, density=False,
                              remove_syn=False,
                              is_ordered=False, init=False, drugname_as_id=False, **kwargs):
    mols_smi = {}
    samples_dict = {}
    output = defaultdict(list)
    nb_labels = 0

    evaluation_schemes = dict(zip(dataset_name, split_mode))
    split_mode = ""

    for name in dataset_name:
        smiles, dt, _ = load_data(input_path, name, use_side_effect,
                                  use_targets,
                                  use_pharm, use_clusters, remove_syn,
                                  use_as_filter, drugname_as_id=drugname_as_id)
        mols_smi.update(smiles)
        samples_dict[name] = dt

    drugs2smiles = _mol_repr(input_path, mols_smi,
                             transformer)  # let's assume that we will use the same transformation for all tasks

    for name, dt in samples_dict.items():

        print(f"Dataset:{name} - split_mode: {evaluation_schemes[name]}")
        data = _filter_pairs_both_exists(dt, filtering_set=drugs2smiles)
        train_data, valid_data, test_data, unseen_data = train_test_valid_split(data, evaluation_schemes[name],
                                                                                seed=seed,
                                                                                test_size=test_size,
                                                                                valid_size=valid_size,
                                                                                n_folds=n_folds,
                                                                                test_fold=test_fold)
        if name in ["twosides", "drugbank"]:
            split_mode = evaluation_schemes[name]
            if debug:
                train_data = dict(sorted(train_data.items())[:6])
                valid_data = dict(sorted(valid_data.items())[:6])
                test_data = dict(sorted(test_data.items())[:6])
                unseen_data = dict(sorted(unseen_data.items())[:6])
            labels = list(train_data.values()) + list(test_data.values()) + list(valid_data.values())
            mbl = MultiLabelBinarizer().fit(labels)
            labels_mapping = {label: i for i, label in enumerate(mbl.classes_)}
            nb_labels = len(labels_mapping)
            print(name, len(labels_mapping), nb_labels)
            train_data = {input: mbl.transform([label]).astype(np.float32)[0] for input, label in train_data.items()}
            valid_data = {input: mbl.transform([label]).astype(np.float32)[0] for input, label in valid_data.items()}
            test_data = {input: mbl.transform([label]).astype(np.float32)[0] for input, label in test_data.items()}
            unseen_data = {input: mbl.transform([label]).astype(np.float32)[0] for input, label in unseen_data.items()}

        if debug:
            train_data = dict(sorted(train_data.items())[:8])
            valid_data = dict(sorted(valid_data.items())[:8])
            test_data = dict(sorted(test_data.items())[:8])
            unseen_data = dict(sorted(unseen_data.items())[:8])
        print(
            f"Dataset:{name}\nlen train: {len(train_data)}\nlen test_ddi: {len(test_data)}\nlen valid: {len(valid_data)}\nlen unseen: {len(unseen_data)}")

        output["train"] += [train_data]
        output["valid"] += [valid_data]
        output["test"] += [test_data]
        output["unseen"] += [unseen_data]

    assert len(samples_dict) == len(output["train"]) == len(output["valid"]) == len(output["test"]) == len(output[
                                                                                                               "unseen"]), "Length Check failed!"
    train_dataset = MultitaskDDIDataset(task_data_lst=output["train"], smi=drugs2smiles, is_ordered=is_ordered,
                                        init=init, nb_labels=nb_labels, **kwargs)
    valid_dataset = MultitaskDDIDataset(task_data_lst=output["valid"], smi=drugs2smiles, is_ordered=is_ordered,
                                        init=init, nb_labels=nb_labels, **kwargs)
    test_dataset = MultitaskDDIDataset(task_data_lst=output["test"], smi=drugs2smiles, is_ordered=is_ordered,
                                       init=init, nb_labels=nb_labels, testing=True, **kwargs)

    if split_mode != "random":
        unsen_dataset = MultitaskDDIDataset(task_data_lst=output["unseen"], smi=drugs2smiles, is_ordered=is_ordered,
                                            init=init, nb_labels=nb_labels, testing=True, **kwargs)
        return train_dataset, valid_dataset, test_dataset, unsen_dataset
    else:
        return train_dataset, valid_dataset, test_dataset, []
    # print(train_dataset.get_targets())


# if auxiliary_dataset is not None:
#     auxl_smiles, (auxl_train_data, auxl_valid_data, auxl_test_data, auxl_unseen_data), _ = load_data(input_path,
#                                                                                                      auxiliary_dataset,
#                                                                                                      transformer,
#                                                                                                      split_mode,
#                                                                                                      seed,
#                                                                                                      test_size,
#                                                                                                      valid_size,
#                                                                                                      n_folds,
#                                                                                                      test_fold)
#     if debug:
#         auxl_train_data = dict(sorted(auxl_train_data.items())[:53])
#         auxl_valid_data = dict(sorted(auxl_valid_data.items())[:53])
#         auxl_test_data = dict(sorted(auxl_test_data.items())[:53])
#         auxl_unseen_data = dict(sorted(auxl_unseen_data.items())[:53])
#
#     print(auxl_train_data)
#     train_dataset.auxl_samples = [(x, auxl_train_data[x]) for x in auxl_train_data]
#     test_dataset.auxl_samples = [(x, auxl_test_data[x]) for x in auxl_test_data]
#     valid_dataset.auxl_samples = [(x, auxl_valid_data[x]) for x in auxl_valid_data]
#     unseen_dataset.auxl_samples = [(x, auxl_unseen_data[x]) for x in auxl_unseen_data]
#     train_dataset.auxl_drug2smi = test_dataset.auxl_drug2smi = valid_dataset.auxl_drug2smi = unseen_dataset.auxl_drug2smi
#     train_dataset.fusion_mode = test_dataset.fusion_mode = valid_dataset.fusion_mode = unseen_dataset.fusion_mode = disjointed
#
# exit()

#
# if use_cmap_scores:
#     l1000_train_dataset = DDIdataset(l1000_train_data, drugs2smiles, label_vectorizer=None, read_as_triplets=True)
#     l1000_valid_dataset = DDIdataset(l1000_valid_data, drugs2smiles, label_vectorizer=None, read_as_triplets=True)
#     l1000_test_dataset = DDIdataset(l1000_test_data, drugs2smiles, label_vectorizer=None, read_as_triplets=True)
#     l1000_unseen_dataset = DDIdataset(l1000_unseen_data, drugs2smiles, label_vectorizer=None, read_as_triplets=True)
#     return (train_dataset, l1000_train_dataset), (valid_dataset, l1000_valid_dataset), (
#         test_dataset, l1000_test_dataset), (unseen_dataset, l1000_unseen_dataset)

# if decagon:
#     ppi_path, dgi_path = f"{input_path}/ppi.csv", f"{input_path}/dgi.csv"
#     gene_gene_ass = pd.read_csv(ppi_path)
#     drug_gene_ass = pd.read_csv(dgi_path)
#     gene_gene_ass_list = gene_gene_ass.values.tolist()
#     drug_gene_ass_list = drug_gene_ass.values.tolist()
#     train_dataset = DDIdataset(train_data, drugs2smiles, mbl, build_graph=False, gene_net=gene_gene_ass_list,
#                                drug_gene_net=drug_gene_ass_list,
#                                decagon=decagon, drugspharm=drugs2pharm)
#     valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl, graph_drugs_mapping=train_dataset.graph_nodes_mapping,
#                                gene_net=train_dataset.gene_net, drug_gene_net=train_dataset.drug_gene_net,
#                                drugspharm=train_dataset.drugspharm)
#     test_dataset = DDIdataset(test_data, drugs2smiles, mbl, graph_drugs_mapping=train_dataset.graph_nodes_mapping,
#                               gene_net=train_dataset.gene_net, drug_gene_net=train_dataset.drug_gene_net,
#                               drugspharm=drugs2pharm)
#     unseen_dataset = DDIdataset(unseen_data, drugs2smiles, mbl,
#                                 graph_drugs_mapping=train_dataset.graph_nodes_mapping,
#                                 gene_net=train_dataset.gene_net, drug_gene_net=train_dataset.drug_gene_net,
#                                 drugspharm=drugs2pharm)
#
# else:
#     if use_graph:
#         train_dataset = DDIdataset(train_data, drugs2smiles, mbl, build_graph=True,
#                                    drugspharm=drugs2pharm)
#         valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl,
#                                    graph_drugs_mapping=train_dataset.graph_nodes_mapping,
#                                    drugspharm=train_dataset.drugspharm)
#         test_dataset = DDIdataset(test_data, drugs2smiles, mbl,
#                                   graph_drugs_mapping=train_dataset.graph_nodes_mapping,
#                                   drugspharm=train_dataset.drugspharm)
#         unseen_dataset = DDIdataset(unseen_data, drugs2smiles, mbl,
#                                     graph_drugs_mapping=train_dataset.graph_nodes_mapping,
#                                     drugspharm=train_dataset.drugspharm)
#     else:
#         train_dataset = DDIdataset(train_data, drugs2smiles, mbl, drugspharm=drugs2pharm, drugstargets=drug2targets,
#                                    drugse=drug_offsides, side_effects_idx_dict=labels_mapping, switch=label,
#                                    read_as_triplets=True, density=density, l1000_pairs=l1000_train_data)
#         valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl, drugspharm=drugs2pharm, drugstargets=drug2targets,
#                                    drugse=drug_offsides, side_effects_idx_dict=labels_mapping, switch=label,
#                                    read_as_triplets=True, density=density, l1000_pairs=l1000_valid_data)
#         test_dataset = DDIdataset(test_data, drugs2smiles, mbl, drugspharm=drugs2pharm, drugstargets=drug2targets,
#                                   drugse=drug_offsides, side_effects_idx_dict=labels_mapping, switch=label,
#                                   read_as_triplets=True, density=density, l1000_pairs=l1000_test_data)
#         print(train_dataset.nb_labels)
#         unseen_dataset = DDIdataset(unseen_data, drugs2smiles, mbl, drugspharm=drugs2pharm,
#                                     drugstargets=drug2targets, side_effects_idx_dict=labels_mapping, switch=label,
#                                     drugse=drug_offsides, read_as_triplets=True, l1000_pairs=l1000_unseen_data)
#
#         if all([len(x) == 3 for x in list(drugs2smiles.values())]):
#             drugs2smiles_dict_1 = {i: j[0] for i, j in drugs2smiles.items()}
#             drugs2smiles_dict_2 = {i: j[1] for i, j in drugs2smiles.items()}
#             drugs2smiles_dict_3 = {i: j[2] for i, j in drugs2smiles.items()}
#             feats = dict(a=drugs2smiles_dict_1, b=drugs2smiles_dict_2, c=drugs2smiles_dict_3)
#             x = [DDIdataset(train_data, feats.get(list(string.ascii_lowercase)[i]), mbl, drugspharm=drugs2pharm,
#                             drugstargets=drug2targets,
#                             drugse=drug_offsides, purpose="autoencoder") for i in range(3)]
#             y = [DDIdataset(valid_data, feats.get(list(string.ascii_lowercase)[i]), mbl, drugspharm=drugs2pharm,
#                             drugstargets=drug2targets,
#                             drugse=drug_offsides, purpose="autoencoder") for i in range(3)]
#             train_dataset = x + [train_dataset]
#             valid_dataset = y + [valid_dataset]


def load_side_effect_mapping(input_path, use_as_filter=None):
    res = {}
    label_mapping = f"{input_path}/classification - twosides_socc.tsv"
    data = pd.read_csv(label_mapping, sep="\t")
    data.fillna("Unspecified", inplace=True)
    for row in data.itertuples():
        if use_as_filter == 'SOC':
            res[row.side_effect] = row.system_organ_class
        elif use_as_filter == "HLGT":
            res[row.side_effect] = row.HLGT
        elif use_as_filter == "HLT":
            res[row.side_effect] = row.HLT
        elif use_as_filter == "LLT":
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


def replace_by_drug_name(dataset_name):
    dc = DataCache()
    data = pd.read_csv(dc.get_file(f"s3://datasets-ressources/DDI/{dataset_name}/{dataset_name}.csv", force=False))
    drug_mapping = pd.read_csv(
        dc.get_file(f"s3://datasets-ressources/DDI/{dataset_name}/{dataset_name}-drugs-all.csv", force=False),
        index_col="drug_id")
    data.replace({entry: drug_mapping.loc[entry, "drug_name"].lower() for entry in drug_mapping.index}, inplace=True)
    drug_name_smiles = {
        drug_mapping.loc[entry, "drug_name"].lower(): drug_mapping.loc[entry, str(list(drug_mapping)[-1])] for
        entry in
        drug_mapping.index}
    return data, drug_name_smiles


def group_by_concept(input_path, data):
    output = defaultdict(set)
    with open(f"{input_path}/twosides_umls_fin.json", "r") as cf:
        cls = json.load(cf)
        for par, chld in cls.items():
            for ld in chld:
                output[ld].add(par)
    for pair, lab in data.items():
        nw = []
        for phen in lab:
            if phen in output:
                nw.extend(list(output[phen]))
            else:
                nw += [phen]
        nw = set(nw) - set(TO_REMOVE)
        data[pair] = list(nw)
    return data


def join_datasets():
    res = []
    drugb, drug_name_smiles_mapping_1 = replace_by_drug_name(dataset_name="drugbank")
    two_sides, drug_name_smiles_mapping_2 = replace_by_drug_name(dataset_name="twosides")
    j_1 = pd.merge(drugb, two_sides, how="inner", on=["Drug 1", "Drug 2"])
    two_sides.rename(columns={
        "Drug 1": "Drug 2",
        "Drug 2": "Drug 1"
    }, inplace=True)
    j_2 = pd.merge(drugb, two_sides, how="inner", on=["Drug 1", "Drug 2"])
    res.extend(j_1.to_dict(orient="records"))
    res.extend(j_2.to_dict(orient="records"))
    # drugs2pharm = {(item["Drug 1"], item["Drug 2"]): item["ddi type_x"].split(";") for item in res}
    data = {(item["Drug 1"], item["Drug 2"]): item["ddi type_y"].split(";") for item in
            res}
    smiles_dict = {i: drug_name_smiles_mapping_1[i] for i in
                   set(drug_name_smiles_mapping_2.keys()) & set(drug_name_smiles_mapping_1.keys())}
    print("Merge datasets contains:{} drug_drug  and {} drugs".format(len(data), len(smiles_dict)))
    return data, smiles_dict


def load_pharm():
    dc = DataCache()
    data = pd.read_csv(dc.get_file(f"s3://datasets-ressources/DDI/drugbank/drugbank.csv", force=False), index_col=0)
    drugs2pharm = {(item[0], item[1]): item[2].split(";") for item in data.itertuples(index=False)}
    return drugs2pharm


def load_meddra(level=None):
    dc = DataCache()
    meddra_mapping = pd.read_csv(dc.get_file("s3://datasets-ressources/side_effect/meddra.tsv.gz"),
                                 compression="gzip", sep="\t", index_col=None, header=None,
                                 names=["drug_id", "level", "notation_id", "side_effect"])

    meddra_mapping.set_index("drug_id", inplace=True)
    if level is not None:
        use_as_filter = meddra_mapping["level"] == level
        meddra_mapping.where(use_as_filter, inplace=True)
    meddra_mapping.dropna(inplace=True)
    return {umls_id: meddra_mapping.loc[[umls_id], "side_effect"].values.tolist()[-1] for umls_id in
            meddra_mapping.index}


def load_mono_se(fname='bio-decagon-mono.csv'):
    dc = DataCache()
    stitch2se = defaultdict(set)
    fin = open(dc.get_file(f"s3://datasets-ressources/DDI/twosides/{fname}"))
    print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        contents = line.strip().split(',')
        stitch_id, se = contents[0], contents[-1].lower()
        stitch2se[stitch_id].add(se)
    return stitch2se


def generator_fn(datasets, batch_size=32, infinite=True, shuffle=True):
    loop = 0
    min_len = min([len(dataset) for dataset in datasets])
    n_steps = int(min_len / batch_size)
    batch_sizes = [int(len(dataset) / n_steps) for dataset in datasets]
    idx = [np.arange(len(dataset)) for dataset in datasets]

    while loop < 1 or infinite:
        for i in range(n_steps):
            batch_x = []
            batch_y = []
            for task_id, dataset in enumerate(datasets[:]):
                start = batch_sizes[task_id] * i
                end = min(i * batch_sizes[task_id] + batch_sizes[task_id], len(dataset))
                a = [dataset[idx[task_id][ii]] for ii in range(start, end)]
                x, y = zip(*a)
                y = list(map(wrapped_partial(torch.unsqueeze, dim=0), y))
                y = torch.cat(y, dim=0)
                # print(len(x), y.shape, len(list(zip(*x))))
                assert y.shape[0] == batch_sizes[task_id], f"task_{task_id}: wrong shape for y"
                assert len(x) == batch_sizes[task_id], "wrong number of instances fecthed!"
                assert all([len(k) == 2 for k in x]), "Must be pair of drugs"
                x = [torch.cat(list(map(wrapped_partial(torch.unsqueeze, dim=0), drugs_list)), dim=0) for drugs_list in
                     list(zip(*x))]
                assert len(x) == 2, "Must be drug a and b, not more!"
                assert x[0].shape[0] == batch_sizes[task_id], " > batch_size"
                batch_x += [x]
                # print(len(batch_x))
                batch_y += [y]
                # print(len(batch_y))
            assert len(batch_x) == len(batch_y) == len(datasets), "Must be equal to len(tasks)!"
            yield batch_x, batch_y
        loop += 1

# if __name__ == '__main__':
#     mol = Chem.MolFromSmiles('CC(N)C1CC1')
#     smis = []
#     for i in range(100):
#         smis.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
#
#     smis_set = list(set(smis))
#     print(smis_set)
#     transformer = SequenceTransformer(alphabet=SMILES_ALPHABET, one_hot=False)
#     tf = transformer.fit()
# #
