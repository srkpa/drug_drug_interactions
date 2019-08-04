import inspect
import math
import re
from collections import defaultdict
from functools import partial
from itertools import product
import numpy as np
import pandas as pd
import csv
import torch as th
from ivbase.utils.commons import to_tensor
from ivbase.utils.datasets.dataset import DGLDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import ast

from side_effects.data.transforms import sequence_transformer


def save_results(filename, contents, engine='xlsxwriter'):
    writer = pd.ExcelWriter(filename, engine=engine)
    for sheet, data in contents:
        data.to_excel(writer, sheet_name=sheet)
    writer.save()


def make_tensor(X, gpu=False):
    a = th.stack([th.cat(pair) for pair in X])
    if gpu:
        a = a.cuda()
    return a


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
        self.X = make_tensor(X)
        #
        if cuda & th.cuda.is_available():
            self.X = self.X.cuda()
        self.y = to_tensor(y, gpu=cuda)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.X[index, :self.X.shape[1] // 2], self.X[index, self.X.shape[1] // 2:]), self.y[index, None]


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


def train_test_valid_split_3(input_path, dataset_name, header=True, shuffle=True, seed=42):
    combo_path = f"{input_path}/{dataset_name}-combo.csv"
    interactions = load_ddis_combinations(fname=combo_path, header=header, dataset_name="twosides")
    drugs = list(set([x1 for (x1, _) in interactions] + [x2 for (_, x2) in interactions]))

    np.random.seed(seed=seed)
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
    print("len gray region", len(interactions) - (len(train) + len(test) + len(valid)))

    fn = open(f"{input_path}/{dataset_name}-train_samples_{seed}.csv", "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in train])
    fn.close()
    fn = open(f"{input_path}/{dataset_name}-test_samples_{seed}.csv", "w")
    fn.writelines(["{},{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in test])
    fn.close()
    fn = open(f"{input_path}/{dataset_name}-valid_samples_{seed}.csv", "w")
    fn.writelines(["{}"
                   ",{},{}\n".format(comb[0], comb[1], ";".join(interactions[comb])) for comb in valid])
    fn.close()


def _extract_interactions(phrase):
    ans = list(re.findall(pattern="DB[0-9]+", string=phrase))
    ans = np.unique(ans)
    # indexes = []
    # for elem in ans:
    #     indexes.append(phrase.find(elem))
    # if indexes[0] > indexes[-1]:
    #     ans = ans[::-1]
    #     # print(indexes, ans)
    #     # print(phrase, "give", ans, label)

    return tuple(ans)


def invivo_directed_train_test_split(input_path, dataset_name="drugbank"):
    train_path = f"{input_path}/{dataset_name}-train_samples.csv"
    test_path = f"{input_path}/{dataset_name}-test_samples.csv"
    valid_path = f"{input_path}/{dataset_name}-valid_samples.csv"

    # Prepare the writer
    a = csv.writer(open("drugbank-train_samples.csv", "w"))
    b = csv.writer(open("drugbank-test_samples.csv", "w"))
    d = csv.writer(open("drugbank-valid_samples.csv", "w"))
    # Load files
    files = [train_path, test_path, valid_path]
    train_data, test_data, valid_data = list(
        map(partial(load_ddis_combinations, dataset_name="split", header=False), files))

    c = load_ddis_combinations(fname=f"{input_path}/directed-drugbank-combo.csv",
                               header=True, dataset_name="split")
    for paire in train_data:
        rev_paire = (paire[-1], paire[0])
        if paire in c:
            a.writerow([paire[0], paire[1], ";".join(c[paire])])
        if rev_paire in c:
            a.writerow([rev_paire[0], rev_paire[1], ";".join(c[rev_paire])])

    for paire in test_data:
        rev_paire = (paire[-1], paire[0])
        if paire in c:
            b.writerow([paire[0], paire[1], ";".join(c[paire])])
        if rev_paire in c:
            b.writerow([rev_paire[0], rev_paire[1], ";".join(c[rev_paire])])

    for paire in valid_data:
        rev_paire = (paire[-1], paire[0])
        if paire in c:
            d.writerow([paire[0], paire[1], ";".join(c[paire])])
        if rev_paire in c:
            d.writerow([rev_paire[0], rev_paire[1], ";".join(c[rev_paire])])


def jy_train_test_split(input_path):
    df = pd.read_csv("s2_mapping_01.csv", sep=",")
    training_set = df[df['Data type used to optimize the DNN architecture'] == "training"]
    testing_set = df[df['Data type used to optimize the DNN architecture'] == "testing"]
    validation_set = df[df['Data type used to optimize the DNN architecture'] == "validation"]

    print("train", len(training_set), " interactions")
    print("test", len(testing_set), " interactions")
    print("validation", len(validation_set), " interactions")

    save_results("jy_split.xlsx", contents=[("train", training_set), ("test", testing_set), ("valid", validation_set)])

    training_set_ref = training_set.groupby('paire')
    testing_set_ref = testing_set.groupby('paire')
    val_set_ref = validation_set.groupby('paire')
    print("len train drugs paires", len(training_set_ref))
    print("len train drugs paires", len(testing_set_ref))
    print("len train drugs paires", len(val_set_ref))

    a = csv.writer(open("drugbank-train_samples.csv", "w"))
    b = csv.writer(open("drugbank-test_samples.csv", "w"))
    d = csv.writer(open("drugbank-valid_samples.csv", "w"))
    c = load_ddis_combinations(fname="directed-drugbank-combo.csv",
                               header=True, dataset_name="split")
    #
    for dar, gr in training_set_ref:
        dar = ast.literal_eval(dar)
        # print(type(dar))
        # liste = []
        # for desc in gr.values:
        #     typ = desc[-1]
        #     liste.append(str(typ))
        a.writerow([dar[0], dar[1], ";".join(c[dar])])

    for dar, gr in testing_set_ref:
        dar = ast.literal_eval(dar)
        # liste = []
        # for desc in gr.values:
        #     typ = desc[-1]
        #     liste.append(str(typ))
        b.writerow([dar[0], dar[1], ";".join(c[dar])])

    for dar, gr in val_set_ref:
        dar = ast.literal_eval(dar)
        # liste = []
        # for desc in gr.values:
        #     typ = desc[-1]
        #     liste.append(str(typ))
        d.writerow([dar[0], dar[1], ";".join(c[dar])])


def analyze_jy_split(input_path):
    filepath2 = f"{input_path}/pnas.1803294115.sd01.xlsx"

    df2 = pd.read_excel(filepath2, sheet_name="Dataset S1")
    side_effects = df2.values.tolist()
    df = pd.read_csv(f"{input_path}/s2_mapping_01.csv", sep=",")
    data = df.values

    train, test, valid = defaultdict(list), defaultdict(list), defaultdict(list)
    inter = defaultdict(list)
    for _, desc, partition, paire, ddi_type in data:
        cle = paire
        inter[cle].append((desc, partition))
        if partition == "training":
            train[cle].append(ddi_type)

        elif partition == "testing":
            test[cle].append(ddi_type)
        else:
            valid[cle].append(ddi_type)

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

    return train, test, valid, train_and_test, train_and_valid, test_and_valid



def mytest(input_path):
    dir = load_ddis_combinations(fname=f"{input_path}/directed-drugbank-combo.csv",
                                 header=True, dataset_name="split")
    undir = load_ddis_combinations(fname=f"{input_path}/drugbank-combo.csv", header=True,
                                   dataset_name="drugbank")

    for cle, val in dir.items():
        found = False
        paire = None
        rev = (cle[-1], cle[0])
        labels = val
        if rev in dir:
            labels += dir[rev]
        if cle in undir:
            found = True
            paire = cle
        elif rev in undir:
            found = True
            paire = rev

        if found:
            if set(undir[paire]) != set(labels):
                print(paire, print(labels), print(undir[paire]))


if __name__ == '__main__':
    drugb_path = "/home/rogia/Documents/analysis/data/TS23_INV"
    all_seeds = [0, 64, 55, 101, 350, 21, 33, 10, 505, 42]
    for seed in all_seeds:
        train_test_valid_split_3(input_path=drugb_path, dataset_name="twosides", seed=seed)

