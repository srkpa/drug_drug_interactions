import torch
from ivbase.nn.commons import get_optimizer
from poutyne.framework import Model
from poutyne.framework.callbacks import BestModelRestore
from poutyne.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoardLogger
from tensorboardX.writer import SummaryWriter
from torch.utils.data import DataLoader
from side_effects.metrics import *
from side_effects.models import all_networks_dict
from side_effects.loss import get_loss
from ivbase.nn.commons import get_optimizer

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


class TensorBoardLogger2(TensorBoardLogger):
    def __init__(self, writer):
        super().__init__(writer)

    def _on_epoch_end_write(self, epoch, logs):
        grouped_items = dict()
        for k, v in logs.items():
            if 'val_' in k:
                primary_key = k[4:]
                if primary_key not in grouped_items:
                    grouped_items[primary_key] = dict()
                grouped_items[k[4:]][k] = v
            else:
                if k not in grouped_items:
                    grouped_items[k] = dict()
                grouped_items[k][k] = v
        for k, v in grouped_items.items():
            if k not in ['epoch', 'time']:
                self.writer.add_scalars(k, v, epoch)


class Trainer(Model):

    def __init__(self, network_params, optimizer='adam', lr=1e-3, weight_decay=0.0, loss=None,
                 metrics_names=None,
                 **loss_params):
        self.history = None
        network_name = network_params.pop('network_name')
        network = all_networks_dict[network_name.lower()](**network_params)
        if torch.cuda.is_available():
            network = network.cuda()
        optimizer = get_optimizer(optimizer)(network.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_name = loss
        self.loss_params = loss_params
        metrics_names = ['micro_roc', 'micro_auprc'] if metrics_names is None else metrics_names
        metrics = [all_metrics_dict[name] for name in metrics_names]
        Model.__init__(self, model=network, optimizer=optimizer, loss_function='bce', metrics=metrics)
        self.metrics_names = metrics_names

    def train(self, train_dataset, valid_dataset, n_epochs=10, batch_size=256,
              log_filename=None, checkpoint_filename=None, tensorboard_dir=None, with_early_stopping=False,
              patience=3, min_lr=1e-06, **kwargs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
        self.loss_function = get_loss(self.loss_name, y_train=train_dataset.get_targets(), **self.loss_params)

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
        if tensorboard_dir:
            tboard = TensorBoardLogger2(SummaryWriter(tensorboard_dir))
            callbacks += [tboard]

        self.history = self.fit_generator(train_generator=train_loader,
                                          valid_generator=valid_loader,
                                          epochs=n_epochs, callbacks=callbacks)

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
