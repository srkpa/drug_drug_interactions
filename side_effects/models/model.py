from math import ceil
import numpy as np
import torch
import torch.nn as nn
from pytoune.framework import Model
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, BestModelRestore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, \
    roc_auc_score as auroc, average_precision_score as auprc, auc
from side_effects.external_utils.metrics import mapk, apk
from torch.nn import AvgPool1d, BatchNorm1d, Conv1d, Dropout, Embedding, MaxPool1d, Sequential, Module
from ivbase.nn.commons import (
    GlobalAvgPool1d, GlobalMaxPool1d, Transpose, UnitNormLayer, get_activation)
from sklearn.metrics import classification_report


class PCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, cnn_sizes, kernel_size, dilatation_rate=1, n_blocks=1,
                 output_dim=86):
        super(PCNN, self).__init__()
        self.output_dim = output_dim
        self.embedding = Embedding(vocab_size, embedding_size)
        self.transpose = Transpose(1, 2)
        self.gpool = GlobalMaxPool1d(dim=1)
        layers = []
        for i, (out_channel, ksize) in \
                enumerate(zip(cnn_sizes, kernel_size, )):
            pad = ((dilatation_rate ** i) * (ksize - 1) + 1) // 2
            layers.append(Conv1d(embedding_size, out_channel, padding=pad,
                                 kernel_size=ksize, dilation=dilatation_rate ** i))

        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transpose(x)
        f_maps = []
        for layer in self.blocks:
            out = layer(x)
            out = self.gpool(out)
            f_maps.append(out)
        out = torch.cat(f_maps, 1)
        fc = nn.Linear(in_features=out.shape[1], out_features=self.output_dim)
        if torch.cuda.is_available():
            fc = fc.cuda()
        return fc(out)


class DeepDDI(nn.Module):

    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(DeepDDI, self).__init__()
        layers = []
        in_ = input_dim
        for out_ in hidden_sizes:
            layers.append(nn.Linear(in_, out_))
            layers.append(nn.BatchNorm1d(out_))
            layers.append(nn.ReLU())
            in_ = out_
        self.net = nn.Sequential(*layers, nn.Linear(in_, output_dim), nn.Sigmoid())

    def forward(self, batch):
        ddi = [torch.cat((x1, x2)) for (x1, x2) in batch]
        x = torch.stack(ddi).type('torch.FloatTensor')
        if torch.cuda.is_available():
            x = x.cuda()
        return self.net(x)


class DDINetwork(nn.Module):

    def __init__(self, drugs_feature_extractor, hidden_sizes, output_dim, use_tar, targets_feature_extractor=None):
        super(DDINetwork, self).__init__()
        self.drug_feature_extractor = drugs_feature_extractor
        self.use_tar = use_tar
        if targets_feature_extractor is not None and use_tar:
            self.target_feature_extractor = targets_feature_extractor
            in_ = self.drug_feature_extractor.output_dim * self.target_feature_extractor.output_dim
        else:
            in_ = 2 * self.drug_feature_extractor.output_dim  # usually check your dim
        self.net = nn.Sequential(nn.Linear(in_, output_dim)) #nn.Sigmoid())

    def forward(self, batch):
        n = len(batch)

        if self.use_tar:
            all_drugs = [drug1[0] for (drug1, _) in batch] + [drug2[0] for (_, drug2) in batch]
            all_targets = [drug1[-1] for (drug1, _) in batch] + [drug2[-1] for (_, drug2) in batch]
            drugs_features = self.drug_feature_extractor(torch.stack(all_drugs))
            targets_features = self.target_feature_extractor(torch.stack(all_targets))
            features_drug1, features_drug2 = torch.split(drugs_features, n)
            features_target1, features_target2 = torch.split(targets_features, n)
            pair1 = torch.bmm(torch.unsqueeze(features_drug1, 2), torch.unsqueeze(features_target1, 1))
            pair2 = torch.bmm(torch.unsqueeze(features_drug2, 2), torch.unsqueeze(features_target2, 1))

            ddi = torch.cat((pair1, pair2), 1)
            ddi = ddi.view((-1, ddi.shape[1] * ddi.shape[-1]))
            out = self.net(ddi)
        else:
            # features = self.drug_feature_extractor(batch)   # bfc
            # ddi = features
            # out = self.net(ddi)
            n1 = batch.shape[1] // 2
            n2 = batch.shape[0]
            drug1, drug2 = torch.split(batch, n1, 1)
            features = self.drug_feature_extractor(torch.cat((drug1, drug2)))   # normal
            features_drug1, features_drug2 = torch.split(features, n2)
            ddi = torch.cat((features_drug1, features_drug2), 1)
            out = self.net(ddi)

            # drug1, drug2 = torch.split(batch, n1, 1)
            # pair = torch.stack((drug1, drug2), dim=1).type("torch.FloatTensor")
            # if torch.cuda.is_available():
            #     pair = pair.cuda()
            # features = self.drug_feature_extractor(pair)
            # out = self.net(features)
        return out


class DDIModel(Model):

    def __init__(self, net, optimizer, lr=0.01, loss=None, patience=10):

        self.network = net
        self.loss = loss
        self.lr = lr
        self.history = None
        self.patience = patience
        Model.__init__(self, model=self.network, optimizer=optimizer, loss_function=self.loss)

    def train(self, x_train, y_train, x_valid, y_valid, n_epochs=10, batch_size=256,
              log_filename=None, checkpoint_filename=None, with_early_stopping=False):

        callbacks = []
        if with_early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=True)
            callbacks += [early_stopping]
        reduce_lr = ReduceLROnPlateau(patience=2, factor=1 / 2, min_lr=1e-6, verbose=True)
        best_model_restore = BestModelRestore()
        callbacks += [reduce_lr, best_model_restore]
        if log_filename:
            logger = CSVLogger(log_filename, batch_granularity=False, separator='\t')
            callbacks += [logger]
        if checkpoint_filename:
            checkpointer = ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True)
            callbacks += [checkpointer]

        nb_steps_train, nb_step_valid = int(len(x_train) / batch_size), int(len(x_valid) / batch_size)
        self.history = self.fit_generator(train_generator=generator(x_train, y_train, batch_size),
                                          steps_per_epoch=nb_steps_train,
                                          valid_generator=generator(x_valid, y_valid, batch_size),
                                          validation_steps=nb_step_valid,
                                          epochs=n_epochs, callbacks=callbacks)

        return self

    def test(self, x, y, batch_size=256):
        valid_gen = generator(x, y, batch=batch_size)
        nsteps = ceil(len(x) / (batch_size * 1.0))
        _, y_pred = self.evaluate_generator(valid_gen, steps=nsteps, return_pred=True)
        y_pred = np.concatenate(y_pred, axis=0)
        if torch.cuda.is_available():
            y_true = y.cpu().numpy()

        else:
            y_true = y

        return y_true, y_pred

    @staticmethod
    def cmetrics(y_true, y_pred, threshold=None, dataset_name="twosides"):
        result = []
        if threshold is None:
            threshold = 0.5

        y_pred = np.where(y_pred >= threshold, 1, 0)

        clr = classification_report(y_true, y_pred)
        print("clasification report", clr)

        first_group = dict(mean_acc=np.mean(
            np.array([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])),
            micro_f1=f1_score(y_true, y_pred, average='micro'),
            macro_f1=f1_score(y_true, y_pred, average='macro'),
            macro_prc=precision_score(y_true, y_pred, average='macro'),
            micro_prc=precision_score(y_true, y_pred, average='micro'),
            micro_rec=recall_score(y_true, y_pred, average='micro'),
            macro_rec=recall_score(y_true, y_pred, average='macro'),
            weight_prec=precision_score(y_true, y_pred, average='weighted'),
            weight_rec=recall_score(y_true, y_pred, average='weighted'),
            weight_f1=f1_score(y_true, y_pred, average='weighted'))
        result.append(first_group)

        if dataset_name == "twosides":
            snd_group = dict(
                accuracy=accuracy_score(y_true, y_pred),
                micro_f1=np.mean(
                    np.array([f1_score(y_true[:, i], y_pred[:, i], average="micro") for i in
                              range(y_true.shape[1])])),
                macro_f1=np.mean(
                    np.array([f1_score(y_true[:, i], y_pred[:, i], average="macro") for i in
                              range(y_true.shape[1])])),
                macro_prc=np.mean(
                    np.array([precision_score(y_true[:, i], y_pred[:, i], average="macro") for i in
                              range(y_true.shape[1])])),
                micro_prc=np.mean(
                    np.array([precision_score(y_true[:, i], y_pred[:, i], average="micro") for i in
                              range(y_true.shape[1])])),
                micro_rec=np.mean(
                    np.array([recall_score(y_true[:, i], y_pred[:, i], average="micro") for i in
                              range(y_true.shape[1])])),
                macro_rec=np.mean(
                    np.array([recall_score(y_true[:, i], y_pred[:, i], average="macro") for i in
                              range(y_true.shape[1])])),
                weight_prec=np.mean(
                    np.array([precision_score(y_true[:, i], y_pred[:, i], average="weighted") for i in
                              range(y_true.shape[1])])),
                weight_rec=np.mean(
                    np.array([recall_score(y_true[:, i], y_pred[:, i], average="weighted") for i in
                              range(y_true.shape[1])])),
                weight_f1=np.mean(
                    np.array([f1_score(y_true[:, i], y_pred[:, i], average="weighted") for i in
                              range(y_true.shape[1])])),
                ham_loss=np.mean(
                    np.array([hamming_loss(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])),
                auroc=auroc(y_true, y_pred),
                sap=np.mean(
                    np.array([auprc(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])),
                ap50=np.mean(
                    np.array(
                        [apk(y_true[:, i].tolist(), y_pred[:, i].tolist(), k=int(y_true.shape[0] * 0.5)) for i in
                         range(y_true.shape[1])])),
                auc=np.mean(
                    np.array([auc(y_true[:, i], y_pred[:, i], reorder=True) for i in range(y_true.shape[1])]))
            )
            result.append(snd_group)

        result = {i: v for i, v in enumerate(result)}
        print(f'The result of the model for threshold {threshold} is: {result}')

        return result

    def load(self, checkpoint_filename):
        self.load_weights(checkpoint_filename)

    def save(self, save_filename):
        self.save_weights(save_filename)


def generator(x, y, batch):
    n = len(x)
    while True:
        for i in range(0, n, batch):
            yield x[i:i + batch], y[i:i + batch]
