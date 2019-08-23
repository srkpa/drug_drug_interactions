import torch
import os
from ivbase.nn.commons import get_optimizer
from torch.nn.functional import binary_cross_entropy
from poutyne.framework import Model
from poutyne.framework.callbacks import BestModelRestore
from poutyne.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Logger
from tensorboardX.writer import SummaryWriter
from torch.utils.data import DataLoader
from side_effects.metrics import *
from side_effects.models import all_networks_dict
from side_effects.loss import get_loss, BinaryCrossEntropyP
from ivbase.nn.commons import get_optimizer
import ivbase.utils.trainer as ivbt
from ivbase.utils.snapshotcallback import SnapshotCallback
from ivbase.utils.trainer import TrainerCheckpoint


all_metrics_dict = dict(
    micro_roc=wrapped_partial(roc_auc_score, average='micro'),
    macro_roc=wrapped_partial(roc_auc_score, average='macro'),
    micro_auprc=wrapped_partial(auprc_score, average='micro'),
    macro_auprc=wrapped_partial(auprc_score, average='macro'),
    micro_f1=wrapped_partial(ivbm.f1_score, average='micro'),
    macro_f1=wrapped_partial(ivbm.f1_score, average='macro'),
    micro_acc=wrapped_partial(accuracy_score, average='micro'),
    macro_acc=wrapped_partial(accuracy_score, average='macro'),
    micro_prec=wrapped_partial(precision_score, average='micro'),
    macro_prec=wrapped_partial(precision_score, average='macro'),
    micro_rec=wrapped_partial(recall_score, average='micro'),
    macro_rec=wrapped_partial(recall_score, average='macro'),
    apk=apk,
    mapk=mapk
)


class TensorBoardLogger2(Logger):
    def __init__(self, writer):
        super().__init__(batch_granularity=True)
        self.writer = writer
        self.train_step = 0
        self.test_step = 0

    def get_scalars(self, logs):
        is_val = False
        for k, v in logs.items():
            if 'val_' in k:
                is_val = True
        return is_val, {k: v for k, v in logs.items() if k not in ['epoch', 'time', 'lr', 'size', 'batch']}

    def _on_batch_end_write(self, batch, logs):
        is_val, scalars = self.get_scalars(logs)
        if is_val:
            t = self.test_step
            self.test_step += 1
        else:
            t = self.train_step
            self.train_step += 1

        for k, v in scalars.items():
            self.writer.add_scalar(f'batch/{k}', v, t)

    def _on_epoch_end_write(self, epoch, logs):
        is_val, scalars = self.get_scalars(logs)
        for k, v in scalars.items():
            self.writer.add_scalar(f'epoch/{k}', v, epoch)


class Trainer(ivbt.Trainer):

    def __init__(self, network_params, optimizer='adam', lr=1e-3, weight_decay=0.0, loss=None,
                 metrics_names=None, use_negative_sampled_loss=False, **loss_params):
        self.history = None
        network_name = network_params.pop('network_name')
        network = all_networks_dict[network_name.lower()](**network_params)
        gpu = torch.cuda.is_available()
        if gpu:
            network = network.cuda()
        optimizer = get_optimizer(optimizer)(network.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_name = loss
        self.loss_params = loss_params
        metrics_names = ['micro_roc', 'micro_auprc'] if metrics_names is None else metrics_names
        metrics = {name: all_metrics_dict[name] for name in metrics_names}
        ivbt.Trainer.__init__(self, net=network, optimizer=optimizer, gpu=gpu, metrics=metrics,
                              loss_fn=BinaryCrossEntropyP(use_negative_sampled_loss))

        # Model.__init__(self, model=network, optimizer=optimizer,
        #                loss_function=BinaryCrossEntropyP(use_negative_sampled_loss), metrics=metrics)
        # self.metrics_names = metrics_names

    def train(self, train_dataset, valid_dataset, n_epochs=10, batch_size=256,
              log_filename=None, checkpoint_filename=None, tensorboard_dir=None, with_early_stopping=False,
              patience=3, min_lr=1e-06, **kwargs):
        if hasattr(self.model, 'set_graph') and hasattr(train_dataset, 'graph_nodes'):
            self.model.set_graph(train_dataset.graph_nodes, train_dataset.graph_edges)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
        # self.loss_function = get_loss(self.loss_name, y_train=train_dataset.get_targets(), **self.loss_params)

        callbacks = []
        restore_path, checkpoint_path = kwargs.pop("restore_path"), kwargs.pop("checkpoint_path")
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
            checkpointer = TrainerCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True)
            callbacks += [checkpointer]
        if tensorboard_dir:
            tboard = TensorBoardLogger2(SummaryWriter(tensorboard_dir))
            callbacks += [tboard]
        if restore_path:
            print("This is your snapshot!")
            snapshoter = SnapshotCallback(s3_path=restore_path)
            callbacks += [snapshoter]
        self.history = self.fit_generator(train_generator=train_loader,
                                          valid_generator=valid_loader,
                                          epochs=n_epochs,
                                          callbacks=callbacks)

        return self

    def test(self, dataset, batch_size=256):
        loader = DataLoader(dataset, batch_size=batch_size)
        y = dataset.get_targets()
        _, metrics_val, y_pred = self.evaluate_generator(loader, return_pred=True)
        y_pred = np.concatenate(y_pred, axis=0)
        if torch.cuda.is_available():
            y_true = y.cpu().numpy()
        else:
            y_true = y.numpy()

        return y_true, y_pred, dict(zip(self.metrics_names, metrics_val))

    def load(self, checkpoint_filename):
        self.load_weights(checkpoint_filename)

    def save(self, save_filename):
        self.save_weights(save_filename)
