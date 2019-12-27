"""
	Split data for cross-validation and write to file.
"""
import csv
import pickle
import random

import numpy as np
import ujson as json
from tqdm import tqdm


def read_graph_structure(drug_feat_idx_jsonl):
    with open(drug_feat_idx_jsonl) as f:
        drugs = [l.split('\t') for l in f]
        print("drugs ", drugs)
        drugs = {idx: json.loads(graph) for idx, graph in tqdm(drugs)}
    return drugs


def read_ddi_instances(ddi_csv, threshold=498, use_small_dataset=True):
    # Building side-effect dictionary and
    # keeping only those which appear more than threshold (498) times.
    side_effects = {}
    with open(ddi_csv) as csvfile:
        drug_reader = csv.reader(csvfile)
        for i, row in enumerate(drug_reader):
            if i > 0:
                did1, did2, sid, *_ = row
                assert did1 != did2
                if sid not in side_effects:
                    side_effects[sid] = []
                side_effects[sid] += [(did1, did2)]

    side_effects = {se: ddis for se, ddis in side_effects.items() if len(ddis) >= threshold}
    if use_small_dataset:  # just for debugging
        side_effects = {se: ddis for se, ddis in
                        sorted(side_effects.items(), key=lambda x: len(x[1]), reverse=True)[:20]}
    print('Total types of polypharmacy side effects =', len(side_effects))
    side_effect_idx_dict = {sid: idx for idx, sid in enumerate(side_effects)}
    return side_effects, side_effect_idx_dict


def prepare_dataset(se_dps_dict, drug_structure_dict):
    def create_negative_instances(drug_idx_list, positive_set, size=None):
        ''' For test and validation set'''
        negative_set = set()
        if not size:
            size = len(positive_set)

        while len(negative_set) < size:
            drug1, drug2 = np.random.choice(drug_idx_list, size=2, replace=False)
            assert drug1 != drug2, 'Shall never happen.'

            neg_se_ddp1 = (drug1, drug2)
            neg_se_ddp2 = (drug2, drug1)

            if neg_se_ddp1 in negative_set or neg_se_ddp2 in negative_set:
                continue
            if neg_se_ddp1 in positive_set or neg_se_ddp2 in positive_set:
                continue

            negative_set |= {neg_se_ddp1}
        return list(negative_set)

    drug_idx_list = list(drug_structure_dict.keys())
    pos_datasets = {}
    neg_datasets = {}

    for i, se in enumerate(tqdm(se_dps_dict)):
        pos_se_ddp = list(se_dps_dict[se])  # copy
        neg_se_ddp = create_negative_instances(
            drug_idx_list, set(pos_se_ddp), size=len(pos_se_ddp))

        random.shuffle(pos_se_ddp)
        random.shuffle(neg_se_ddp)
        pos_datasets[se] = pos_se_ddp
        neg_datasets[se] = neg_se_ddp
    return pos_datasets, neg_datasets


# To use our splitting scheme, I will have to modify this.  # May be add a args=dict(f{f1:}
def split_decagon_cv(path, pos_datasets, neg_datasets, n_fold):
    for se, k in pos_datasets.items():
        print(se, k[0])
    exit()

    def split_all_cross_validation_datasets(datasets, n_fold):
        cv_dataset = {x: {} for x in range(1, n_fold + 1)}
        for se in datasets:
            fold_len = len(datasets[se]) // n_fold
            for fold_i in range(1, n_fold + 1):
                fold_start = (fold_i - 1) * fold_len

                if fold_i < n_fold:
                    fold_end = fold_i * fold_len
                else:
                    fold_end = len(datasets[se])
                cv_dataset[fold_i][se] = datasets[se][fold_start:fold_end]
        return cv_dataset

    # to define another function ==
    pos_cv_dataset = split_all_cross_validation_datasets(pos_datasets, n_fold)
    neg_cv_dataset = split_all_cross_validation_datasets(neg_datasets, n_fold)

    for i in range(1, n_fold + 1):
        with open(path + "/" + str(i) + "fold.npy", 'wb') as file:
            fold_dataset = {'pos': pos_cv_dataset[i], 'neg': neg_cv_dataset[i]}
            pickle.dump(fold_dataset, file)


def prepare_decagon_cv(path, decagon_graph_data, ddi_data, debug, n_atom_type=100, n_bond_type=12,
                       n_fold=3):
    # graph_dict is ex drug_dict.
    graph_dict = read_graph_structure(path + "/" + decagon_graph_data)
    print(len(graph_dict))
    side_effects, side_effect_idx_dict = read_ddi_instances(
        path + "/" + ddi_data, use_small_dataset=debug)
    pos_datasets, neg_datasets = prepare_dataset(side_effects, graph_dict)
    n_side_effect = len(side_effects)
    split_decagon_cv(path=path, pos_datasets=pos_datasets, neg_datasets=neg_datasets, n_fold=n_fold)
    return n_atom_type, n_bond_type, graph_dict, n_side_effect, side_effect_idx_dict


def split_qm9_cv(graph_dict, labels_dict, opt):
    data_size = len(graph_dict)  # ids are in [1,133,885]
    print("data_size", data_size)

    test_graph_dict = {}
    test_labels_dict = {}
    while len(test_graph_dict) < 10000 and len(test_graph_dict) < data_size:
        x = random.randint(1, data_size)
        if x not in test_graph_dict.keys():
            test_graph_dict[x] = graph_dict[str(x)]
            test_labels_dict[x] = labels_dict[str(x)]

    valid_graph_dict = {}
    valid_labels_dict = {}
    while len(valid_graph_dict) < 10000:
        x = random.randint(1, data_size)
        if x not in valid_graph_dict.keys() and x not in test_graph_dict.keys():
            valid_graph_dict[x] = graph_dict[str(x)]
            valid_labels_dict[x] = labels_dict[str(x)]

    train_graph_dict = {}
    train_labels_dict = {}
    for x in range(1, data_size + 1):
        if x not in valid_graph_dict.keys():
            if x not in test_graph_dict.keys():
                train_graph_dict[x] = graph_dict[str(x)]
                train_labels_dict[x] = labels_dict[str(x)]

    with open(opt.path + "folds/" + "train_graphs.npy", 'wb') as f:
        f.write(pickle.dumps(train_graph_dict))
    with open(opt.path + "folds/" + "train_labels.npy", 'wb') as f:
        f.write(pickle.dumps(train_labels_dict))

    with open(opt.path + "folds/" + "valid_graphs.npy", 'wb') as f:
        f.write(pickle.dumps(valid_graph_dict))
    with open(opt.path + "folds/" + "valid_labels.npy", 'wb') as f:
        f.write(pickle.dumps(valid_labels_dict))

    with open(opt.path + "folds/" + "test_graphs.npy", 'wb') as f:
        f.write(pickle.dumps(test_graph_dict))
    with open(opt.path + "folds/" + "test_labels.npy", 'wb') as f:
        f.write(pickle.dumps(test_labels_dict))


def prepare_qm9_cv(opt):
    def read_qm9_labels(drug_labels_jsonl):
        with open(drug_labels_jsonl) as f:
            labels_dict = [l.split('\t') for l in f]
            labels_dict = {idx: json.loads(labels) for idx, labels in tqdm(labels_dict)}
        return labels_dict

    # this is missing the "datasets" (graph pairs), because
    # i will compute them on the fly to be able to do it multiple times
    # if it's unstable.
    opt.graph_dict = read_graph_structure(opt.path + opt.qm9_graph_data)
    labels_dict = read_qm9_labels(opt.path + opt.qm9_labels)
    opt.n_atom_type = 5  # CHONF
    opt.n_bond_type = 5  # single, double, triple, aromatic, self
    split_qm9_cv(opt.graph_dict, labels_dict, opt)
    return opt


def main(path, dataset_name, ddi_data='bio-decagon-combo.csv', decagon_graph_data="drug.feat.wo_h.self_loop.idx.jsonl",
         n_fold=3, debug=True, qm9_labels='viz_drug.labels.jsonl', qm9_graph_data="viz_drug.feat.self_loop.idx.jsonl"):
    n_atom_type, n_bond_type, graph_dict, n_side_effect, side_effect_idx_dict = 0, 0, 0, 0, 0
    if dataset_name == "twosides":
        n_atom_type, n_bond_type, graph_dict, n_side_effect, side_effect_idx_dict = prepare_decagon_cv(path=path,
                                                                                                       decagon_graph_data=decagon_graph_data,
                                                                                                       debug=debug,
                                                                                                       n_fold=n_fold,
                                                                                                       ddi_data=ddi_data)

    # print('Dump to file:', path + "/input_data.npy")
    # np.save(path + "/input_data.npy", opt)
    return n_atom_type, n_bond_type, graph_dict, n_side_effect, side_effect_idx_dict
