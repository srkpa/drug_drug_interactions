import inspect
import math
import re
from collections import defaultdict
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from side_effects.preprocess.transforms import deepddi_transformer
from ivbase.utils.datasets.dataset import DGLDataset, GenericDataset
from torch.utils.data import Dataset
from ivbase.utils.commons import to_tensor


class TDGLDataset(DGLDataset):

    def __init__(self, X, y, cuda=True):
        assert isinstance(X[0], (tuple, list))
        x_1, x_2 = zip(*X)
        assert len(x_1) == len(X)
        assert len(x_1) == len(x_2)
        self.x1 = DGLDataset(x_1, y, cuda=cuda)
        self.x2 = DGLDataset(x_2, y, cuda=cuda)
        assert len(self.x1.G) == len(self.x2.G)
        assert self.x1.y.shape == self.x2.y.shape
        self.y = self.x2.y
        self.X = list(zip(self.x1.G, self.x2.G))

    def __len__(self):
        return len(self.x1.G)

    def __getitem__(self, idx):
        return (self.x1.G[idx], self.x2.G[idx]), self.y[idx, None]


class MyDataset(Dataset):
    def __init__(self, X, y, cuda):
        super(MyDataset, self).__init__()
        self.X = X
        self.y = to_tensor(y, gpu=cuda)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index, None]


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


def train_test_split_1(fname, header=True, train_split=0.8, shuffle=True, save=None, prefix=None):
    ddis = load_ddis_combinations(fname=fname, header=header)
    drugs = list(set([x1 for (x1, _) in ddis] + [x2 for (_, x2) in ddis]))

    n = len(drugs)
    indices = np.arange(n)
    np.random.seed(10)
    if shuffle:
        np.random.shuffle(indices)

    split = math.floor(train_split * n)
    train_idx, test_idx, valid_idx = indices[:split], indices[split:], indices[split:]

    train_drugs = [drugs[idx] for idx in train_idx]
    test_drugs = [drugs[idx] for idx in test_idx]
    valid_drugs = [drugs[idx] for idx in valid_idx]

    train_samples = []
    test_samples = []
    valid_samples = []

    for drug in valid_drugs:
        possible_valid_ass = zip(train_drugs, [drug] * len(train_idx))
        valid_samples.extend(possible_valid_ass)

    for drug in train_drugs:
        possible_trains_associations = list(zip([drug] * len(train_idx), train_drugs))
        train_samples.extend(possible_trains_associations)

    for drug in test_drugs:
        possible_test_associations = list(zip([drug] * len(drugs), drugs))
        test_samples.extend(possible_test_associations)

    absent_valid_associations = set(valid_samples) - set(ddis.keys())
    valid_samples = set(valid_samples) - set(absent_valid_associations)

    absent_train_associations = set(train_samples) - set(ddis.keys())
    absent_test_associations = set(test_samples) - set(ddis.keys())

    train_samples = set(train_samples) - set(absent_train_associations)
    test_samples = set(test_samples) - set(absent_test_associations)

    print(len(test_samples), len(train_samples), len(valid_samples))  # 12108 51364
    print(len(test_samples) + len(train_samples) + len(valid_samples))
    assert len(test_samples) + len(train_samples) + len(valid_samples) == len(ddis)

    if save is not None:
        fn = open("../dataset/{}-train_samples.csv".format(prefix), "w")
        fn.writelines(["{}"
                       ",{},{}\n".format(comb[0], comb[1], ";".join(ddis[comb])) for comb in train_samples])
        fn.close()
        fn = open("../dataset/{}-test_samples.csv".format(prefix), "w")
        fn.writelines(["{},{},{}\n".format(comb[0], comb[1], ";".join(ddis[comb])) for comb in test_samples])
        fn.close()
        fn = open("../dataset/{}-valid_samples.csv".format(prefix), "w")
        fn.writelines(["{}"
                       ",{},{}\n".format(comb[0], comb[1], ";".join(ddis[comb])) for comb in valid_samples])
        fn.close()


def _filter(samples, transformed_smiles_dict=None):
    return {(transformed_smiles_dict[id_1], transformed_smiles_dict[id_2]): y for (id_1, id_2), y in samples.items() if
            id_1 in transformed_smiles_dict and id_2 in transformed_smiles_dict}


def load_train_test_files(input_path, dataset_name, transformer):
    # My paths
    all_drugs_path = f"{input_path}/{dataset_name}-drugs-all.csv"
    train_path = f"{input_path}/{dataset_name}-train_samples.csv"
    test_path = f"{input_path}/{dataset_name}-test_samples.csv"
    valid_path = f"{input_path}/{dataset_name}-valid_samples.csv"

    # Load files
    files = [train_path, test_path, valid_path]
    train_data, test_data, valid_data = list(
        map(partial(load_ddis_combinations, dataset_name="split", header=False), files))

    # Load smiles
    drugs2smiles = load_smiles(fname=all_drugs_path, dataset_name=dataset_name)
    drugs = list(drugs2smiles.keys())
    smiles = list(drugs2smiles.values())

    # Transformer
    args = inspect.signature(transformer)
    if 'approved_drug' in args.parameters:
        approved_drugs_path = f"{input_path}/drugbank_approved_drugs.csv"
        approved_drug = pd.read_csv(approved_drugs_path, sep=",").dropna()
        approved_drug = approved_drug['PubChem Canonical Smiles'].values.tolist()
        drugs = transformer(drugs=drugs, smiles=smiles, approved_drug=approved_drug)
    else:
        drugs = transformer(drugs=drugs, smiles=smiles)

    train_data, test_data, valid_data = list(
        map(partial(_filter, transformed_smiles_dict=drugs), [train_data, test_data, valid_data]))
    x_train, x_test, x_valid = list(map(lambda x: list(x.keys()), [train_data, test_data, valid_data]))

    assert len(set(x_train) - set(x_test)) == len(train_data)
    assert len(set(x_test) - set(x_valid)) == len(test_data)

    print("len train", len(train_data))
    print("len test", len(test_data))
    print("len valid", len(valid_data))

    mbl = MultiLabelBinarizer()
    labels = list(train_data.values()) + list(test_data.values()) + list(valid_data.values())
    targets = mbl.fit_transform(labels).astype(np.float32)
    y_train, y_test, y_valid = targets[:len(train_data), :], targets[len(train_data):len(train_data) + len(test_data),
                                                             :], targets[len(train_data) + len(test_data):, ]

    return list(mbl.classes_), x_train, x_test, x_valid, y_train, y_test, y_valid


def split(fname, train_split=0.8, shuffle=True, prefix="twosides"):
    twoside2se = load_ddis_combinations(fname=fname, header=True)
    drugs = defaultdict(set)
    for (d1, d2), se in twoside2se.items():
        drugs[d1].add(d2)

    train_samples = []
    test_samples = []
    valid_samples = []

    for drug, interactions in drugs.items():
        interactions = list(interactions)
        n = len(interactions)
        indices = np.arange(n)
        np.random.seed(42)

        if shuffle:
            np.random.shuffle(indices)
        split = math.floor(train_split * n)
        train_idx, test_idx = indices[:split], indices[split:]
        np.random.shuffle(train_idx)
        train_idx, valid_idx = train_idx[:math.floor(train_split * len(train_idx))], train_idx[math.floor(
            train_split * len(train_idx)):]

        train_drugs = [interactions[idx] for idx in train_idx]
        test_drugs = [interactions[idx] for idx in test_idx]
        valid_drugs = [interactions[idx] for idx in valid_idx]

        train_samples.extend(list(zip([drug] * len(train_idx), train_drugs)))
        valid_samples.extend(list(zip([drug] * len(valid_idx), valid_drugs)))
        test_samples.extend(list(zip([drug] * len(test_idx), test_drugs)))

    assert len(set(test_samples) - set(valid_samples)) == len(set(test_samples))
    assert len(set(test_samples) - set(train_samples)) == len(set(test_samples))
    assert len(train_samples) + len(valid_samples) + len(test_samples) == len(twoside2se)

    fn = open("{}-train_samples.csv".format(prefix), "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(twoside2se[comb])) for comb in train_samples])
    fn.close()
    fn = open("{}-test_samples.csv".format(prefix), "w")
    fn.writelines(["{},{},{}\n".format(comb[0], comb[1], ";".join(twoside2se[comb])) for comb in test_samples])
    fn.close()
    fn = open("{}-valid_samples.csv".format(prefix), "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(twoside2se[comb])) for comb in valid_samples])
    fn.close()


def train_test_valid_split(input_path, dataset_name, header=True, train_split=0.8, shuffle=True):
    combo_path = f"{input_path}/{dataset_name}-combo.csv"
    interactions = load_ddis_combinations(fname=combo_path, header=header, dataset_name=dataset_name)
    drugs = list(set([x1 for (x1, _) in interactions] + [x2 for (_, x2) in interactions]))

    n = len(drugs)
    print(n)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)
    split = math.floor(train_split * n)
    train_idx, test_idx = indices[:split], indices[split:]

    train = set(interactions.keys()).intersection(set(product([drugs[index] for index in train_idx], repeat=2)))
    test = set(interactions.keys()).intersection(set(product([drugs[index] for index in test_idx], repeat=2)))
    valid = set(interactions.keys()).intersection(set(product(drugs, repeat=2)) - train - test)

    print(len(train), len(test), len(valid))
    fn = open(f"{input_path}/{dataset_name}-train_samples.csv", "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in train])
    fn.close()
    fn = open(f"{input_path}/{dataset_name}-test_samples.csv", "w")
    fn.writelines(["{},{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in test])
    fn.close()
    fn = open(f"{input_path}/{dataset_name}-valid_samples.csv", "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in valid])
    fn.close()


def train_test_valid_split_2(input_path, dataset_name, header=True, train_split=0.8, shuffle=True):
    combo_path = f"{input_path}/{dataset_name}-combo.csv"
    interactions = load_ddis_combinations(fname=combo_path, header=header, dataset_name=dataset_name)
    print("len", len(interactions))
    drugs = list(set([x1 for (x1, _) in interactions] + [x2 for (_, x2) in interactions]))
    res_1 = defaultdict(list)
    res_2 = defaultdict(list)
    for (drug_1, drug_2) in interactions:
        res_1[drug_1].append((drug_1, drug_2))
        res_2[drug_2].append((drug_1, drug_2))
    # Split
    n = len(drugs)
    indices = np.arange(n)
    np.random.seed(42)
    if shuffle:
        np.random.shuffle(indices)
    split = math.floor(train_split * n)
    train_idx, _ = indices[:split], indices[split:]
    # Interactions
    train = set(interactions.keys()).intersection(set(product([drugs[index] for index in train_idx], repeat=2)))
    drug_i = [drug_i for (drug_i, _) in train]
    drug_j = [drug_j for (_, drug_j) in train]
    assert len(drug_i) == len(drug_j)

    result_1 = {drug: set(res_1[drug]) - train for drug in drug_i}
    temp_list = [len(values.intersection(train)) for _, values in result_1.items()]
    assert sum(temp_list) == 0, 'There is some example that are availbale in the train and the test/valid subsets'
    result_2 = {drug: set(res_2[drug]) - train for drug in drug_j}
    temp_list = [len(values.intersection(train)) for _, values in result_2.items()]
    assert sum(temp_list) == 0, 'There is some example that are availbale in the train and the test/valid subsets'

    temp_list = [cle for cle in result_1 if len(list(result_1[cle])) == 0]
    with open('result_1_unique_example_only_available_in_the_train_set.txt', 'w') as f:
        for cle in temp_list:
            temp_l = [pair for pair in train if pair[0] == cle]
            assert (len(temp_l) == len(res_1[cle]))
            # This is a verification for the drug example combination that only show up in the train since we don't have enough example of those combination
            f.writelines("\n".join([f"{x1},{x2}" for x1, x2 in temp_l] + ["\n"]))

    with open('result_2_unique_example_only_available_in_the_train_set.txt', 'w') as f:
        temp_list = [cle for cle in result_2 if len(list(result_2[cle])) == 0]
        for cle in temp_list:
            temp_l = [pair for pair in train if pair[1] == cle]
            assert (len(temp_l) == len(res_2[cle]))
            # This is a verification for the drug example combination that only show up in the train since we don't have enough example of those combination
            f.writelines("\n".join([f"{x1},{x2}" for x1, x2 in temp_l] + ["\n"]))

    rest = []
    rest.extend(list(result_1.values()))
    rest.extend(list(result_2.values()))
    print(rest)
    exit()
    assert len(rest) == len(result_1) + len(result_2)
    valid_samples, test_samples = [], []
    for liste_interactions in rest:
        liste = list(liste_interactions)
        n = len(liste)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        split = math.floor(0.3 * n)
        valid_idx, test_idx = indices[:split], indices[split:]
        valid = [liste[idx] for idx in valid_idx]
        test = [liste[idx] for idx in test_idx]
        valid_samples.extend(valid)
        test_samples.extend(test)
    print('len train', len(list(train)))
    print('len test', len(list(test_samples)))
    print('len valid', len(list(valid_samples)))

    print(len(train), len(test_samples), len(valid_samples))
    fn = open(f"{input_path}/{dataset_name}-train_samples.csv", "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in train])
    fn.close()
    fn = open(f"{input_path}/{dataset_name}-test_samples.csv", "w")
    fn.writelines(["{},{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in test_samples])
    fn.close()
    fn = open(f"{input_path}/{dataset_name}-valid_samples.csv", "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in valid_samples])
    fn.close()


def train_test_valid_split_3(input_path, dataset_name, header=True, shuffle=True):
    combo_path = f"{input_path}/{dataset_name}-combo.csv"
    interactions = load_ddis_combinations(fname=combo_path, header=header, dataset_name=dataset_name)
    drugs = list(set([x1 for (x1, _) in interactions] + [x2 for (_, x2) in interactions]))

    np.random.seed(42)
    train_idx, test_idx = train_test_split(drugs, test_size=0.1, shuffle=shuffle)
    train_idx, valid_idx = train_test_split(train_idx, test_size=0.15, shuffle=shuffle)

    train = set(product(train_idx, repeat=2))
    valid = set(product(train_idx, valid_idx)).union(set(product(valid_idx, train_idx)))
    test = set(product(train_idx, test_idx)).union(set(product(test_idx, train_idx)))

    train = set(interactions.keys()).intersection(train)
    valid = set(interactions.keys()).intersection(valid)
    test = set(interactions.keys()).intersection(test)

    print('len train', len(list(train)))
    print('len test', len(list(test)))
    print('len valid', len(list(valid)))
    print("len region grise", len(interactions) - (len(train) + len(test) + len(valid)))

    fn = open(f"{input_path}/{dataset_name}-train_samples.csv", "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in train])
    fn.close()
    fn = open(f"{input_path}/{dataset_name}-test_samples.csv", "w")
    fn.writelines(["{},{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in test])
    fn.close()
    fn = open(f"{input_path}/{dataset_name}-valid_samples.csv", "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in valid])
    fn.close()


def deepddi_train_test_split(input_path):
    pattern = "DB[0-9]+"
    filepath1 = f"{input_path}/pnas.1803294115.sd02.xlsx"
    filepath2 = f"{input_path}/pnas.1803294115.sd01.xlsx"

    df2 = pd.read_excel(filepath2, sheet_name="Dataset S1")
    side_effects = df2.values.tolist()
    side_effects = {description: ddi_type for ddi_type, description in side_effects}

    df = pd.read_excel(filepath1, sheet_name="Dataset S2")
    data = df.values.tolist()
    train, test, valid = defaultdict(list), defaultdict(list), defaultdict(list)

    i = 0
    for desc, partition in data:
        tr = False
        ddi = [word for word in desc.split() if not word.startswith("DB")]
        drugs = list(re.findall(pattern, desc))
        for info, ddi_type in side_effects.items():
            inter = set(desc.split()).intersection(set(info.split()))
            if inter == set(ddi):
                tr = True
                i += 1
                cle = tuple(np.unique(drugs))
                if partition == "training":
                    train[cle].append(ddi_type)
                elif partition == "testing":
                    test[cle].append(ddi_type)
                else:
                    valid[cle].append(ddi_type)
                break
        if not tr:
            print(desc, ddi)

    assert len(data) == i
    #### Analysis.

    print("Quick Summary #1.\n There are:")
    print(f"- {len(data)} interactions")
    print(f"- {len(side_effects)} side effects")
    k = list(train.keys()) + list(test.keys()) + list(valid.keys())
    print(f"- {len(set(k))} pairs of drugs")

    print(f"- {len(train)} pairs of drugs and ",
          f"{sum([len(v) for i, v in train.items() if len])} interactions for train ")
    print(f"- {len(test)} pairs of drugs and ", f"{sum([len(v) for i, v in test.items()])} interactions for test")
    print(f"- {len(valid)} pairs of drug and ", f"{sum([len(v) for i, v in valid.items()])} interactions for valid")

    b = [pair for pair, se in train.items() if len(se) > 1]
    print(f"- {len(b)} pairs of drugs with 2 type of ddi in train subset only.")

    c = [pair for pair, se in valid.items() if len(se) > 1]
    print(f"- {len(c)} pairs of drugs with 2 type of ddi in valid subset only.")

    d = [pair for pair, se in test.items() if len(se) > 1]
    print(f"- {len(d)} pairs of drugs with 2 type of ddi in test subset only.")

    ####
    train_and_test, train_and_valid, test_and_valid = [], [], []

    print("\nQuick Summary #2\n There are:")
    train_test_inter = set(train.keys()).intersection(set(test.keys()))
    if len(train_test_inter) > 0:
        print(f"- {len(train_test_inter)} pairs of drugs are present in train and test subsets.")
    for k in train_test_inter:
        train_and_test.append([k, "-".join([str(x) for x in train[k]]), "-".join([str(x) for x in test[k]])])

    train_valid_inter = set(train.keys()).intersection(set(valid.keys()))
    if len(train_valid_inter) > 0:
        print(f"- {len(train_valid_inter)} pairs of drugs are present in train and valid subsets.")
    for k in train_valid_inter:
        train_and_valid.append([k, "-".join([str(x) for x in train[k]]), "-".join([str(x) for x in valid[k]])])

    test_valid_inter = set(valid.keys()).intersection(set(test.keys()))
    if len(test_valid_inter) > 0:
        print(f"- {len(test_valid_inter)} pairs of drugs are present in test and valid subsets.")
    for k in test_valid_inter:
        test_and_valid.append([k, "-".join([str(x) for x in test[k]]), "-".join([str(x) for x in valid[k]])])

    train_test_valid_inter = set(train.keys()).intersection(set(test.keys())).intersection(set(valid.keys()))
    print(f"- {len(train_test_valid_inter)} pairs of drugs are present in train, test and valid subsets.")

    print("\nQuick Summary #3:")
    print(
        "- pairs of drugs that belong to two of the three subsets have always two types of ddi which have been split into the subsets")
    print(
        "- pairs of drugs that belong only to one of the three subsets, can have 1 or 2 types of ddi, but all the type "
        "will belong to the choosen subset")

    return train, test, valid, train_and_test, train_and_valid, test_and_valid


# correct finalement
def load(train, test, valid, method, input_path="../data/violette/drugbank/", transformer=None):
    dataset_name = "drugbank"
    all_drugs_path = f"{input_path}/{dataset_name}-drugs-all.csv"
    approved_drugs_path = f"{input_path}/drugbank_approved_drugs.csv"

    drugs2smiles = load_smiles(fname=all_drugs_path, dataset_name=dataset_name)
    res = transformer(drugs=list(drugs2smiles.keys()), smiles=list(drugs2smiles.values()), return_one_hot=False)

    if method == "deepddi":
        approved_drug = pd.read_csv(approved_drugs_path, sep=",").dropna()
        approved_drug = approved_drug['PubChem Canonical Smiles'].values.tolist()
        res = deepddi_transformer(inputs=drugs2smiles, approved_drug=approved_drug)

    train_data = {(res[drug1_id], res[drug2_id]): train[(drug1_id, drug2_id)] for drug1_id, drug2_id in train if
                  drug1_id in res and drug2_id in res}
    test_data = {(res[drug1_id], res[drug2_id]): test[(drug1_id, drug2_id)] for drug1_id, drug2_id in test if
                 drug1_id in res and drug2_id in res}
    valid_data = {(res[drug1_id], res[drug2_id]): valid[(drug1_id, drug2_id)] for drug1_id, drug2_id in valid if
                  drug1_id in res and drug2_id in res}

    x_train, x_test, x_valid = list(train_data.keys()), list(test_data.keys()), list(valid_data.keys())

    labels = list(train_data.values()) + list(test_data.values()) + list(valid_data.values())
    mbl = MultiLabelBinarizer()
    targets = mbl.fit_transform(labels).astype(np.float32)

    y_train, y_test, y_valid = targets[:len(x_train), :], targets[len(x_train):len(x_train) + len(x_test), :], \
                               targets[len(x_train) + len(x_test):, ]

    assert len(x_valid) == len(y_valid)
    print("Quick summary #4\n Final subsets:\n")
    print(f"- {targets.shape[1]} side effects")
    print(f"- {len(train_data)} pairs of drugs for train")
    print(f"- {len(test_data)} pairs of drugs for test")
    print(f"- {len(valid_data)} pairs of drug for valid")

    return x_train, x_test, x_valid, np.array(y_train).astype(np.float32), np.array(y_test).astype(
        np.float32), np.array(y_valid).astype(np.float32)
