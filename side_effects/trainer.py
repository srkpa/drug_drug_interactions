import traceback
from math import ceil

import ivbase.utils.trainer as ivbt
import torch
from ivbase.nn.commons import get_optimizer
from ivbase.utils.commons import params_getter
from ivbase.utils.snapshotcallback import SnapshotCallback
from ivbase.utils.trainer import TrainerCheckpoint
from poutyne.framework.callbacks import BestModelRestore
from poutyne.framework.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from pytoune.framework.callbacks import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

name2callback = {"early_stopping": EarlyStopping,
                 "reduce_lr": ReduceLROnPlateau,
                 "norm_clip": ClipNorm}
from side_effects.data.loader import MTKDataset
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
        mapk=mapk,
        miroc=wrapped_partial(roc, average='micro'),
        miaup=wrapped_partial(aup, average='micro'),
        maroc=wrapped_partial(roc, average='macro'),
        maaup=wrapped_partial(aup, average='macro'),
        mse=mse
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
            if isinstance(train_dataset, tuple) and isinstance(valid_dataset, tuple):
                train_dataset = MTKDataset(train_dataset)
                valid_dataset = MTKDataset(valid_dataset)
                step_train, step_valid = int(min([len(dataset) for dataset in train_dataset.args]) / batch_size), int(
                    min([len(dataset) for dataset in valid_dataset.args]) / batch_size)
                self.history = self.fit(train_dataset, valid_dataset, epochs=n_epochs, steps_per_epoch=step_train,
                                        validation_steps=step_valid, batch_size=batch_size, shuffle=True,
                                        callbacks=callbacks,
                                        generator_fn=generator_fn)
            else:
                step_train, step_valid = int(len(train_dataset) / batch_size), int(len(valid_dataset) / batch_size)
                self.history = self.fit(train_dataset, valid_dataset, epochs=n_epochs, steps_per_epoch=step_train,
                                        validation_steps=step_valid, generator_fn=batch_generator,
                                        batch_size=batch_size, shuffle=True, callbacks=callbacks)
        return self

    def fit(self, train_dt, valid_dt, epochs=1000, shuffle=False, batch_size=32,
            callbacks=[], log_path=None, checkpoint=None, tboardX=".logs", **kwargs):
        valid_generator = None
        generator_fn = kwargs.get(
            "generator_fn", self._dataloader_from_dataset)
        if isinstance(train_dt, Dataset):
            train_generator = generator_fn(
                train_dt, batch_size=batch_size, shuffle=shuffle)
        else:
            train_generator = self._dataloader_from_data(
                *train_dt, batch_size=batch_size)

        if valid_dt is not None: # changed
            if isinstance(valid_dt, Dataset):
                valid_generator = generator_fn(
                    valid_dt, batch_size=batch_size, shuffle=shuffle)
            else:
                valid_generator = self._dataloader_from_data(
                    valid_dt, batch_size=batch_size)

        callbacks = (callbacks or [])
        # Check if early stopping is asked for
        for name in ["early_stopping", "reduce_lr"]:
            cback = kwargs.get(name)
            if cback:
                if not isinstance(cback, dict):
                    cback = {}
                try:
                    cback = name2callback[name](**cback)
                    callbacks += [cback]
                except:
                    print("Exception in callback {}".format(name))
                    traceback.print_exc(file=sys.stdout)

        clip_val = kwargs.get("norm_clip", None)
        if clip_val is not None:
            clip_cback = name2callback["norm_clip"](self.model.parameters(), clip_val)
            callbacks.append(clip_cback)

        if checkpoint:
            if isinstance(checkpoint, bool):
                checkpoint = 'model.epoch:{epoch:02d}-loss:{val_loss:.2f}.pth.tar'
            checkpoint = os.path.join(self.model_dir, checkpoint)
            checkpoint_params = dict(monitor='val_loss', save_best_only=True,
                                     temporary_filename=checkpoint + ".tmp")
            checkpoint_params.update(params_getter(
                kwargs, ("checkpoint__", "check__", "c__")))
            check_callback = TrainerCheckpoint(checkpoint, **checkpoint_params)
            callbacks += [check_callback]

        if log_path:
            log_path = os.path.join(self.model_dir, log_path)
            logger = CSVLogger(
                log_path, batch_granularity=False, separator='\t')
            callbacks += [logger]

        if tboardX:
            if isinstance(tboardX, dict):
                tboardX["logdir"] = os.path.join(self.model_dir, tboardX["logdir"])
            else:
                tboardX = {"logdir": os.path.join(self.model_dir, tboardX)}
            self.writer = SummaryWriter(
                **tboardX)  # this has a purpose, which is to access the writer from the GradFlow Callback
            callbacks += [TensorBoardLogger(self.writer)]

        return self.fit_generator(train_generator,
                                  valid_generator=valid_generator,
                                  epochs=epochs,
                                  steps_per_epoch=kwargs.get(
                                      "steps_per_epoch"),
                                  validation_steps=kwargs.get(
                                      "validation_steps"),
                                  initial_epoch=kwargs.get("initial_epoch", 1),
                                  verbose=kwargs.get("verbose", True),
                                  callbacks=callbacks)

    def _compute_metrics(self, pred_y, y):
        if isinstance(pred_y, tuple):
            pred_y = [pred.detach() for pred in pred_y]
            return np.array([float(metric(pred_y, *y)) for metric in self.metrics])

        return np.array([float(metric(pred_y.detach(), *y)) for metric in self.metrics])

    def test(self, dataset, batch_size=256):
        self.model.testing = True
        metrics_val = []
        y = [d.get_targets() for d in dataset] if isinstance(dataset, tuple) else dataset.get_targets()
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
            if isinstance(dataset, tuple):
                dataset = MTKDataset(dataset)
                loader = generator_fn(dataset, batch_size, infinite=False, shuffle=False)
            else:
                loader = batch_generator(dataset, batch_size, infinite=False, shuffle=False)
            len_data = len(dataset)
            nsteps = ceil(len_data / (batch_size * 1.0))
            if self.metrics:
                _, metrics_val, y_pred = self.evaluate_generator(loader, steps=nsteps, return_pred=True)
            else:
                _, y_pred = self.evaluate_generator(loader, steps=nsteps, return_pred=True)

        if isinstance(y, (list, tuple)) and isinstance(y_pred, (list, tuple)):
            y_pred = list(zip(*y_pred))
            y_pred = [np.concatenate(ii) for ii in y_pred]
            if torch.cuda.is_available():
                y_true = [yt.cpu().numpy() for yt in y]
            else:
                y_true = [yt.numpy() for yt in y]
        else:
            y_pred = np.concatenate(y_pred, axis=0)
            if torch.cuda.is_available():
                y_true = y.cpu().numpy()
            else:
                y_true = y.numpy() if not isinstance(y, np.ndarray) else y
        # y_pred = y_pred.squeeze()
        res = dict(auprc=auprc_score(y_pred=y_pred, y_true=y_true, average="micro"),
                   roc=roc_auc_score(y_pred=y_pred, y_true=y_true, average="micro")) if not self.metrics else dict(
            zip(self.metrics_names, metrics_val))

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
            # y = list(map(wrapped_partial(torch.unsqueeze, dim=0), y))
            y = torch.cat(y, dim=0)  # .unsqueeze(dim=1)
            yield list(zip(*x)), y
        loop += 1


def generator_fn(datasets, batch_size=32, infinite=True, shuffle=True):
    loop = 0
    datasets = datasets.args
    min_len = min([len(dataset) for dataset in datasets])
    n_steps = int(min_len / batch_size)
    batch_sizes = [int(len(dataset) / n_steps) for dataset in datasets]
    idx = [np.arange(len(dataset)) for dataset in datasets]

    while loop < 1 or infinite:
        for i in range(n_steps):
            batch_x = []
            batch_y = []
            for task_id, dataset in enumerate(datasets[:]):
                start = batch_sizes[task_id] * i
                end = min(i * batch_sizes[task_id] + batch_sizes[task_id], len(dataset))
                a = [dataset[idx[task_id][ii]] for ii in range(start, end)]
                x, y = zip(*a)
                y = list(map(wrapped_partial(torch.unsqueeze, dim=0), y))
                y = torch.cat(y, dim=0)
                # print(len(x), y.shape, len(list(zip(*x))))
                assert y.shape[0] == batch_sizes[task_id], f"task_{task_id}: wrong shape for y"
                assert len(x) == batch_sizes[task_id], "wrong number of instances fecthed!"
                assert all([len(k) == 2 for k in x]), "Must be pair of drugs"
                x = [torch.cat(list(map(wrapped_partial(torch.unsqueeze, dim=0), drugs_list)), dim=0) for drugs_list in
                     list(zip(*x))]
                assert len(x) == 2, "Must be drug a and b, not more!"
                assert x[0].shape[0] == batch_sizes[task_id], " > batch_size"
                batch_x += [x]
                # print(len(batch_x))
                batch_y += [y]
                # print(len(batch_y))
            assert len(batch_x) == len(batch_y) == len(datasets), "Must be equal to len(tasks)!"
            yield batch_x, batch_y
        loop += 1
