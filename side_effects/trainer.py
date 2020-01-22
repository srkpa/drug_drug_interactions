from math import ceil

import ivbase.utils.trainer as ivbt
import torch
from ivbase.nn.commons import get_optimizer
from ivbase.utils.snapshotcallback import SnapshotCallback
from ivbase.utils.trainer import TrainerCheckpoint
from poutyne.framework.callbacks import BestModelRestore
from poutyne.framework.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, Logger
from tensorboardX.writer import SummaryWriter
from torch.utils.data import DataLoader

from side_effects.loss import BinaryCrossEntropyP
from side_effects.metrics import *
from side_effects.models import all_networks_dict


def get_metrics():
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
    return all_metrics_dict


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

    def __init__(self, network_params, loss_params, optimizer='adam', lr=1e-3, weight_decay=0.0, loss=None,
                 metrics_names=None, snapshot_dir="", dataloader=True):

        self.history = None
        network_name = network_params.pop('network_name')
        network = all_networks_dict[network_name.lower()](**network_params)
        self.n_gpu = torch.cuda.device_count()
        # if self.n_gpu > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     network = DataParallel(network, device_ids=[0, 1, 2, 3])
        gpu = torch.cuda.is_available()
        optimizer = get_optimizer(optimizer)(network.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_name = loss
        self.loss_params = loss_params
        self.dataloader_from_dataset = dataloader
        metrics_names = [] if metrics_names is None else metrics_names
        metrics = {name: get_metrics()[name] for name in metrics_names}

        ivbt.Trainer.__init__(self, net=network, optimizer=optimizer, gpu=gpu, metrics=metrics,
                              loss_fn=BinaryCrossEntropyP(loss=loss, **loss_params), snapshot_path=snapshot_dir)

        # Model.__init__(self, model=network, optimizer=optimizer,
        #                loss_function=BinaryCrossEntropyP(use_negative_sampled_loss), metrics=metrics)
        # self.metrics_names = metrics_names

    def train(self, train_dataset, valid_dataset, n_epochs=10, batch_size=256,
              log_filename=None, checkpoint_filename=None, tensorboard_dir=None, with_early_stopping=False,
              patience=3, min_lr=1e-06, checkpoint_path=None, **kwargs):
        # Data parallel
        # if self.n_gpu > 1:
        #     if hasattr(self.model.module, 'set_graph') and hasattr(train_dataset, 'graph_nodes'):
        #         print("we are here")
        #         self.model.module.set_graph(train_dataset.graph_nodes, train_dataset.graph_edges)

        if hasattr(self.model, 'set_graph') and hasattr(train_dataset, 'graph_nodes'):
            self.model.set_graph(train_dataset.graph_nodes, train_dataset.graph_edges)

        # self.loss_function = get_loss(self.loss_name, y_train=train_dataset.get_targets(), **self.loss_params)
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
            checkpointer = TrainerCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True)
            callbacks += [checkpointer]
        if tensorboard_dir:
            tboard = TensorBoardLogger2(SummaryWriter(tensorboard_dir))
            callbacks += [tboard]
        if checkpoint_path:
            snapshoter = SnapshotCallback(s3_path=checkpoint_path, save_every=1)
            callbacks += [snapshoter]

        if self.dataloader_from_dataset:
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
            self.history = self.fit_generator(train_generator=train_loader,
                                              valid_generator=valid_loader,
                                              epochs=n_epochs,
                                              callbacks=callbacks)
        else:
            step_train, step_valid = int(len(train_dataset) / batch_size), int(len(valid_dataset) / batch_size)
            self.history = self.fit(train_dataset, valid_dataset, epochs=n_epochs, steps_per_epoch=step_train,
                                    validation_steps=step_valid, generator_fn=batch_generator,
                                    batch_size=batch_size, shuffle=True, callbacks=callbacks)
        return self

    def test(self, dataset, batch_size=256):
        self.model.testing = True
        metrics_val = []
        y = dataset.get_targets()
        if self.model.is_binary_output:
            self.metrics_names = []
            self.metrics = {}

        if self.dataloader_from_dataset:
            loader = DataLoader(dataset, batch_size=batch_size)
            if self.metrics:
                _, metrics_val, y_pred = self.evaluate_generator(loader, return_pred=True)
            else:
                _, y_pred = self.evaluate_generator(loader, return_pred=True)
        else:
            loader = batch_generator(dataset, batch_size, infinite=False, shuffle=False)
            nsteps = ceil(len(dataset) / (batch_size * 1.0))
            if self.metrics:
                _, metrics_val, y_pred = self.evaluate_generator(loader, steps=nsteps, return_pred=True)
            else:
                _, y_pred = self.evaluate_generator(loader, steps=nsteps, return_pred=True)
        y_pred = np.concatenate(y_pred, axis=0)
        if torch.cuda.is_available():
            y_true = y.cpu().numpy()
        else:
            y_true = y.numpy() if not isinstance(y, np.ndarray) else y
        y_pred = y_pred.squeeze()
        res = dict(auprc=auprc_score(y_pred=y_pred, y_true=y_true, average="micro"),
                   roc=roc_auc_score(y_pred=y_pred, y_true=y_true, average="micro")) if not self.metrics else dict(
            zip(self.metrics_names, metrics_val))
        print(res)

        return y_true, y_pred, res

    def load(self, checkpoint_filename):
        self.load_weights(checkpoint_filename)

    def save(self, save_filename):
        self.save_weights(save_filename)


def batch_generator(dataset, batch_size=32, infinite=True, shuffle=True):
    loop = 0
    idx = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(idx)

    while loop < 1 or infinite:
        for i in range(0, len(dataset), batch_size):
            start = i
            end = min(i + batch_size, len(dataset))
            a = [dataset[idx[ii]] for ii in range(start, end)]
            x, y = zip(*a)
            y = torch.cat(y).unsqueeze(dim=1)
            yield list(zip(*x)), y
        loop += 1
