import numpy as np
import torch
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from .loader import *
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from side_effects.metrics import compute_metrics


class DeepRF:

    def __init__(self, network_params, metrics_names, option='default'):
        super(DeepRF, self).__init__()
        del (network_params["network_name"])
        self.use_pretrained_features = network_params.pop("use_pretrained_features", True)
        self.nb_chains = network_params.pop("nb_chains", None)
        drug_features_extractor_params = network_params.pop("pretrained_drug_features_extractor_params", None)
        if self.use_pretrained_features:
            self.drug_features_extractor = load_pretrained_model(**drug_features_extractor_params)
        else:
            self.drug_features_extractor = None
        self.mode = option
        if option is "chain":
            nb_chains = self.nb_chains
            self.model = [ClassifierChain(RandomForestClassifier(**network_params), order='random', random_state=i) for
                          i in range(nb_chains)]
        elif option == "multi-output":
            self.model = MultiOutputClassifier(RandomForestClassifier(**network_params))
        else:
            self.model = RandomForestClassifier(**network_params)
        self.metrics = {name: get_metrics()[name] for name in metrics_names}

    def __call__(self, train_dataset, valid_dataset, test_dataset, batch_size=256, **kwargs):
        x_train, y_train, x_test, y_test = self.get_partitions(train_dataset, valid_dataset, test_dataset, batch_size,
                                                               self.use_pretrained_features)
        print(f"train target, {y_train.shape}")
        print(f"test target, {y_test.shape}")
        if self.mode is "chain":
            for chain in self.model:
                chain.fit(x_train, y_train)
            y_pred = np.array([chain.predict(x_test) for chain in self.model]).mean(axis=0)
        else:
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
        metrics = compute_metrics(y_pred, y_test, self.metrics)
        print(f"Output metrics, {metrics}")
        return y_test, y_pred, metrics

    def get_partitions(self, train_dataset, valid_dataset, test_dataset, batch_size=256, use_pretrained_features=True,
                       **kwargs):

        if use_pretrained_features:
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
            x_train = self.drug_features_extractor.test_generator(train_loader)
            x_test = self.drug_features_extractor.test_generator(test_loader)
            x_valid = self.drug_features_extractor.test_generator(valid_loader)
        else:
            x_train = list(zip(*train_dataset.get_samples()))
            x_train = np.concatenate((np.concatenate(x_train[0], axis=0), np.concatenate(x_train[-1], axis=0)), axis=1)
            x_test = list(zip(*test_dataset.get_samples()))
            x_test = np.concatenate((np.concatenate(x_test[0], axis=0), np.concatenate(x_test[-1], axis=0)), axis=1)
            x_valid = list(zip(*valid_dataset.get_samples()))
            x_valid = np.concatenate((np.concatenate(x_valid[0], axis=0), np.concatenate(x_valid[-1], axis=0)), axis=1)

        x_train = np.concatenate((x_train, x_valid), axis=0)
        print(f"train dataset, {x_train.shape}")
        print(f"test dataset, {x_test.shape}")
        y_train = train_dataset.get_targets()
        y_test = test_dataset.get_targets()
        y_valid = valid_dataset.get_targets()
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.numpy()
        if isinstance(y_valid, torch.Tensor):
            y_valid = y_valid.numpy()
        y_train = np.concatenate((y_train, y_valid), axis=0)

        return x_train, y_train, x_test, y_test
