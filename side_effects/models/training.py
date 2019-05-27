from math import ceil

import torch
from pytoune.framework import Model
from pytoune.framework.callbacks import BestModelRestore
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score as auroc, average_precision_score as auprc

from side_effects.external_utils.metrics import *


class DDIModel(Model):

    def __init__(self, network, optimizer, loss=None, patience=3):
        self.history = None
        self.patience = patience
        Model.__init__(self, model=network, optimizer=optimizer, loss_function=loss)

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
            y_true = y.numpy()

        return y_true, y_pred

    def load(self, checkpoint_filename):
        self.load_weights(checkpoint_filename)

    def save(self, save_filename):
        self.save_weights(save_filename)


def compute_metrics(y_true, y_pred):
    # AUROC and AUPRC are not dependent on the thresholds
    thresholds, average_precisions, precision, recall = auprc(actual=y_true, scores=y_pred)
    false_pos_rates, true_pos_rates, roc_auc_scores = auroc(actual=y_true, scores=y_pred)

    best_thresholds = thresholds["micro"]

    # Theses metrics below are dependent on the thresholds
    predicted = list(map(predict, [y_pred] * len(best_thresholds), best_thresholds))
    assert len(predicted) == len(best_thresholds)
    macro_metrics = list(map(acc_precision_f1_recall, [y_true] * len(predicted), predicted))
    assert len(predicted) == len(macro_metrics)
    micro_metrics = list(
        map(acc_precision_f1_recall, [y_true] * len(predicted), predicted, ["micro"] * len(predicted)))
    assert len(predicted) == len(micro_metrics)

    classification_reports = list(map(model_classification_report, [y_true] * len(predicted), predicted))
    assert len(predicted) == len(classification_reports)

    return dict(zip(["thresholds", "ap", "prec", "recall", "fpr", "tpr", "ROC", "macro", "micro", "report"],
                    [thresholds, average_precisions, precision, recall, false_pos_rates, true_pos_rates,
                     roc_auc_scores,
                     macro_metrics, micro_metrics, classification_reports]))


def generator(x, y, batch):
    n = len(x)
    while True:
        for i in range(0, n, batch):
            yield x[i:i + batch], y[i:i + batch]
