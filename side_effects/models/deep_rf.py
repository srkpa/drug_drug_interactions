import numpy as np
import torch
from sklearn.multioutput import ClassifierChain
from .loader import *
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from side_effects.metrics import compute_metrics


class DeepRF:

    def __init__(self, network_params, metrics_names):
        super(DeepRF, self).__init__()
        del (network_params["network_name"])
        drug_features_extractor_params = network_params.pop("pretrained_drug_features_extractor_params")
        self.drug_features_extractor = load_pretrained_model(**drug_features_extractor_params)
        nb_chains = network_params.pop("nb_chains")
        base_lr = RandomForestClassifier(**network_params)
        self.chains = [ClassifierChain(base_lr, order='random', random_state=i) for i in range(nb_chains)]
        self.metrics = {name: get_metrics()[name] for name in metrics_names}

    def __call__(self, train_dataset, valid_dataset, test_dataset, batch_size=256, **kwargs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        valid_loader = DataLoader(test_dataset, batch_size=batch_size)
        x_train = self.drug_features_extractor.test_generator(train_loader)
        x_test = self.drug_features_extractor.test_generator(test_loader)
        x_valid = self.drug_features_extractor.test_generator(valid_loader)
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
            y_valid = y_train.numpy()
        y_train = np.concatenate((y_train, y_valid), axis=0)
        print(f"train target, {y_train.shape}")
        print(f"test target, {y_test.shape}")
        for chain in self.chains:
            chain.fit(x_train, y_train)
        y_pred = np.array([chain.predict(x_test) for chain in self.chains])
        y_pred_ensemble = y_pred.mean(axis=0)
        ens_metrics = compute_metrics(y_pred_ensemble, y_test, self.metrics)
        print(f"ensemble metrics, {ens_metrics}")
        # chain_metrics = [compute_metrics(Y_pred_chain, y_test, self.metrics) for Y_pred_chain in y_pred]
        return y_test, y_pred_ensemble, ens_metrics  # chain_metrics
