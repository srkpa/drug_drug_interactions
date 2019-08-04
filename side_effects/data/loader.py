import torch
import inspect
import pandas as pd
from functools import partial
from collections import defaultdict
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


class DDIdataset(Dataset):
    def __init__(self, samples, drug_to_smiles, label_vectorizer):
        self.samples = samples
        self.drug_to_smiles = drug_to_smiles
        self.labels_vectorizer = label_vectorizer
        self.gpu = torch.cuda.is_available()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        (drug1, drug2), label = self.samples[item]
        target = self.labels_vectorizer.transformer([label]).astype(np.float32)[0]
        return ((to_tensor(self.drug_to_smiles[drug1], self.gpu),
                 to_tensor(self.drug_to_smiles[drug1], self.gpu)),
                to_tensor(target, self.gpu))

    @property
    def nb_labels(self):
        return len(self.labels_vectorizer.classes_)

    def get_targets(self):
        y = zip(*self.samples)[1]
        return to_tensor(self.labels_vectorizer.transformer(y).astype(np.float32), self.gpu)


def get_data_partitions(dataset_name, input_path, transformer, seed=None, test_size=0.25, valid_size=0.25):
    # My paths
    all_drugs_path = f"{input_path}/{dataset_name}-drugs-all.csv"
    if seed is None:
        all_combo_path = f"{input_path}/directed-drugbank-combo.csv"
        data = load_ddis_combinations(all_combo_path, header=True)
        train_data, test_data = train_test_split(data, test_size=test_size)
        train_data, valid_data = train_test_split(train_data, test_size=valid_size)
    else:
        train_path = f"{input_path}/{dataset_name}-train_samples_{seed}.csv"
        test_path = f"{input_path}/{dataset_name}-test_samples_{seed}.csv"
        valid_path = f"{input_path}/{dataset_name}-valid_samples_{seed}.csv"
        # Load files
        files = [train_path, test_path, valid_path]
        train_data, test_data, valid_data = list(
            map(partial(load_ddis_combinations, dataset_name="split", header=False), files))
    print(f"len train {len(train_data)}\nlen test {len(test_data)}\nlen valid {len(valid_data)}")

    # Load smiles
    drugs2smiles = load_smiles(fname=all_drugs_path, dataset_name=dataset_name)
    drugs, smiles = list(drugs2smiles.keys()), list(drugs2smiles.values())

    # Transformer
    args = inspect.signature(transformer)
    if 'approved_drug' in args.parameters:
        approved_drugs_path = f"{input_path}/drugbank_approved_drugs.csv"
        approved_drug = pd.read_csv(approved_drugs_path, sep=",").dropna()
        approved_drug = approved_drug['PubChem Canonical Smiles'].values.tolist()
        drugs2smiles = transformer(drugs=drugs, smiles=smiles, approved_drug=approved_drug)
    else:
        drugs2smiles = transformer(drugs=drugs, smiles=smiles)
    train_data, test_data, valid_data = list(
        map(partial(_filter_samples, transformed_smiles_dict=drugs), [train_data, test_data, valid_data]))

    labels = list(train_data.values()) + list(test_data.values()) + list(valid_data.values())
    mbl = MultiLabelBinarizer().fit(labels)
    train_dataset = DDIdataset(train_data, drugs2smiles, mbl)
    valid_dataset = DDIdataset(valid_data, drugs2smiles, mbl)
    test_dataset = DDIdataset(test_data, drugs2smiles, mbl)

    return train_dataset, valid_dataset, test_dataset

