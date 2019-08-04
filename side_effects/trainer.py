import torch
from torch.utils.data import DataLoader
from pytoune.framework import Model
from pytoune.framework.callbacks import BestModelRestore
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from side_effects.metrics import *
from side_effects.models import *
from side_effects.loss import get_loss
from ivbase.nn.commons import get_optimizer

all_networks_dict = dict(
    deepddi=DeepDDI,
    bmnddi=BMNDDI
)


class Trainer(Model):

    def __init__(self, network_params, optimizer, lr, weight_decay, loss=None, **loss_params):
        self.history = None
        network_name = network_params.pop('name')
        network = all_networks_dict[network_name.lower()](**network_params)
        if torch.cuda.is_available():
            network = network.cuda()
        optimizer = get_optimizer(optimizer)(network.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_name = loss
        self.loss_params = loss_params
        Model.__init__(self, model=network, optimizer=optimizer, loss_function='bce')

    def train(self, train_dataset, valid_dataset, n_epochs=10, batch_size=256,
              log_filename=None, checkpoint_filename=None, with_early_stopping=False,
              patience=3, min_lr=1e-06, ):
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
        self.loss_function = get_loss(self.loss_name, y_train=train_dataset.targets(), **self.loss_params)

        callbacks = []
        if with_early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=True)
            callbacks += [early_stopping]
        reduce_lr = ReduceLROnPlateau(patience=5, factor=1 / 4, min_lr=min_lr, verbose=True)
        best_model_restore = BestModelRestore()
        callbacks += [reduce_lr, best_model_restore]
        if log_filename:
            logger = CSVLogger(log_filename, batch_granularity=False, separator='\t')
            callbacks += [logger]
        if checkpoint_filename:
            checkpointer = ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True)
            callbacks += [checkpointer]

        self.history = self.fit_generator(train_generator=train_loader,
                                          valid_generator=valid_loader,
                                          epochs=n_epochs, callbacks=callbacks)
        exit(33)

        return self

    def test(self, dataset, batch_size=256):
        loader = DataLoader(dataset, batch_size=batch_size)
        y = dataset.get_targets()
        _, y_pred = self.evaluate_generator(loader, return_pred=True)
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

    print('Micro ROC score: {0:0.2f}'.format(
        roc_auc_scores["micro"]))

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precisions["micro"]))

    best_thresholds = thresholds["micro"]
    print(f"Length Best thresholds: {len(best_thresholds)}")

    return dict(zip(["thresholds", "ap", "prec", "recall", "fpr", "tpr", "ROC"],
                    [thresholds, average_precisions, precision, recall, false_pos_rates, true_pos_rates,
                     roc_auc_scores]))


def compute_macro_m(y_true, y_pred, best_thresholds):
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

    return dict(zip(["macro", "micro", "report"],
                    [macro_metrics, micro_metrics, classification_reports]))
