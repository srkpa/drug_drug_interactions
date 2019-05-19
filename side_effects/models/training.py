import torch
import os
from ivbase.utils.gradcheck import GradFlow
from ivbase.utils.gradinspect import GradientInspector
from ivbase.utils.trainer import Trainer
from pytoune.framework.callbacks import BestModelRestore

from side_effects.external_utils.metrics import *


def generator(dataset, batch_size=32, infinite=True, shuffle=True):
    # If the batch_size is chosen to be -1, then we assume all the elements are in the same batch
    if batch_size == -1:
        while True:
            yield dataset  # return the complete dataset
    else:
        # Prepare the indexes necessary for the batch selection
        loop = 0
        idx = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(idx)

        # Loop the dataset once (for testing and validation) or infinite times (for training)
        while loop < 1 or infinite:

            # Loop the whole dataset and create the batches according to batch_size
            for i in range(0, len(dataset), batch_size):
                start = i  # starting index of the batch
                end = min(i + batch_size, len(dataset))  # ending index of the batch
                a = [dataset[idx[ii]] for ii in range(start, end)]  # Generate the batch 'a'
                x, y = zip(*a)
                x, y = list(map(torch.stack, [x, y]))
                yield x, y  # return the x and y values of the batch
            loop += 1


class DDIModel(Trainer):

    def __init__(self, network, optimizer, loss, model_dir, gpu):
        super(DDIModel, self).__init__(net=network, optimizer=optimizer, loss_fn=loss, model_dir=model_dir, gpu=gpu)
        self.__grad_outfile = os.path.join(model_dir, "grad.dot")
        self.__tboardx = "tlogs"
        self.__history = "history"
        self.__checkpoint = "checkpoint"

    def train(self, train_dt, valid_dt, n_epochs, batch_size, early_stopping=True, reduce_lr=True):
        best_model_restore = BestModelRestore()
        gradflow = GradFlow(outfile=self.__grad_outfile, enforce_sanity=False, max_val=1e4)
        gradinspector = GradientInspector(top_zoom=0.2, update_at="epoch")

        callbacks = [best_model_restore, gradflow, gradinspector]

        nb_steps_train, nb_step_valid = int(len(train_dt) / batch_size), int(len(valid_dt) / batch_size)

        self.fit(train_dt, valid_dt, epochs=n_epochs, shuffle=True, batch_size=batch_size,
                 callbacks=callbacks, log_path=self.__history, checkpoint=self.__checkpoint,
                 tboardX=self.__tboardx,
                 steps_per_epoch=nb_steps_train, generator_fn=generator,
                 validation_steps=nb_step_valid, reduce_lr=reduce_lr, early_stopping=early_stopping)

    def test(self, x, y):
        y_pred = self.predict_on_batch(x)
        if torch.cuda.is_available():
            y_true = y.cpu().numpy()
        else:
            y_true = y.numpy()
        return y_true, y_pred

    @staticmethod
    def predictions_given_scores(y_true, y_pred):
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
