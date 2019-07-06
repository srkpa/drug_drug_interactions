import json
import os
import torch as th
import ivbase.nn.extractors as feat
import torch.optim as optim
from ivbase.utils.datasets.dataset import GenericDataset, DGLDataset
from pytoune.framework.metrics import get_loss_or_metric
from sklearn.utils import compute_class_weight

from side_effects.external_utils.init import *
from side_effects.external_utils.loss import Weighted_binary_cross_entropy1
from side_effects.models.model import PCNN, FCNet, DRUUD, DeepDDI, DGLGraph
from side_effects.preprocess.dataset import TDGLDataset, MyDataset
from side_effects.preprocess.transforms import *
from ivbase.nn.base import FCLayer

all_networks_dict = dict(
    pcnn=PCNN,
    deepddi=DeepDDI,
    fcnet=FCNet,
    druud=DRUUD,
    conv1d=feat.Cnn1dFeatExtractor,
    lstm=feat.LSTMFeatExtractor,
    fcfeat=FCLayer,
    dglgraph=DGLGraph
)

all_transformers_dict = dict(
    seq=sequence_transformer,
    fgp=fingerprints_transformer,
    deepddi=deepddi_transformer,
    dgl=dgl_transformer
)

all_init_fn_dict = dict(
    constant=init_constant,
    kaiming_uniform=init_kaiming_uniform,
    normal=init_normal,
    uniform=init_uniform,
    xavier=init_xavier_normal
)

all_optimizers_dict = dict(
    adadelta=optim.Adadelta,
    adagrad=optim.Adagrad,
    adam=optim.Adam,
    sparseadam=optim.SparseAdam,
    adamax=optim.Adamax,
    asgd=optim.ASGD,
    lbfgs=optim.LBFGS,
    rmsprop=optim.RMSprop,
    rprop=optim.Rprop,
    sgd=optim.SGD,
)

all_dataset_fn = dict(
    generic=GenericDataset,
    dgl=DGLDataset,
    tdgl=TDGLDataset,
    common=MyDataset
)


def compute_classes_weight(y):
    return torch.tensor([compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=target)
                         if len(np.unique(target)) > 1 else np.array([1.0, 1.0]) for target in y.T],
                        dtype=torch.float32).t()


def save(obj, filename, output_path):
    with open(os.path.join(output_path, filename), 'w') as CNF:
        json.dump(obj, CNF)


def create_dataset(dataset_fn, **kwargs):
    return dataset_fn(**kwargs)


def get_dataset(dt):
    return all_dataset_fn[dt]


def get_loss(loss, **kwargs):
    if loss == "weighted":
        y = kwargs.get("y_train")
        n_pos = kwargs.get("n_pos")
        if isinstance(y, th.Tensor):
            y = y.cpu().numpy()
        w = compute_classes_weight(y=y)
        print("weight", w.shape)
        if torch.cuda.is_available():
            w = w.cuda()
        return Weighted_binary_cross_entropy1(weights_per_targets=w, n_pos=n_pos)
    return get_loss_or_metric(loss)


def get_init_fn(init_fn):
    return all_init_fn_dict[init_fn]


def get_network(network, parameters):
    assert isinstance(network, str)
    net = network.lower()
    return all_networks_dict[net](**parameters)


def get_transformer(transformation):
    assert isinstance(transformation, str)
    return all_transformers_dict[transformation]


def get_optimizer(optimizer, module, params):
    return all_optimizers_dict[optimizer](module.parameters(), **params)
