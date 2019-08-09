import json
import os
import pandas as pd
import ivbase.nn.extractors as feat
import matplotlib.pyplot as plt
import torch.optim as optim
from ivbase.utils.datasets.dataset import GenericDataset, DGLDataset
from pytoune.framework.metrics import get_loss_or_metric
from sklearn.utils import compute_class_weight

from side_effects.inits import *
from side_effects.loss import WeightedBinaryCrossEntropy1, weighted_binary_cross_entropy3, \
    WeightedBinaryCrossEntropy5
from side_effects.models.bmn_ddi import PCNN, FCNet, BMNDDI, DeepDDI, DGLGraph
from side_effects.data.utils import TDGLDataset, MyDataset
from side_effects.data.transforms import *
from ivbase.nn.base import FCLayer

all_networks_dict = dict(
    pcnn=PCNN,
    deepddi=DeepDDI,
    fcnet=FCNet,
    druud=BMNDDI,
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


def unbalanced_classes_weights(y):
    weights = non_zeros_distribution(y)
    return torch.from_numpy(weights / y.shape[0])


def compute_labels_density(y):
    one_w = np.sum(y, axis=0)
    zero_w = y.shape[0] - one_w
    w = np.maximum(one_w, zero_w) / np.minimum(one_w, zero_w)
    return torch.from_numpy(w)


def compute_classes_weight(y, use_exp=False, exp=1):
    weights = np.array([compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=target)
                        if len(np.unique(target)) > 1 else np.array([1.0, 1.0]) for target in y.T], dtype=np.float32)
    weights = torch.from_numpy(weights).t()
    if use_exp:
        weights = exp * weights
        return torch.exp(weights)
    return weights


def compute_pos_weights(y):
    one_w = np.sum(y, axis=0)
    zero_w = [len(y) - w for w in one_w]
    pos_weights = np.array(zero_w / one_w, dtype=np.float32)
    return torch.from_numpy(pos_weights)


def non_zeros_distribution(y):
    one_w = np.sum(y, axis=0)
    zero_w = [len(y) - w for w in one_w]
    return np.array([one_w, zero_w], dtype=np.float32)


def label_distribution(y):
    return (np.sum(y, axis=0) / y.shape[0]) * 100


def save(obj, filename, output_path):
    with open(os.path.join(output_path, filename), 'w') as CNF:
        json.dump(obj, CNF)


def create_dataset(dataset_fn, **kwargs):
    return dataset_fn(**kwargs)


def get_dataset(dt):
    return all_dataset_fn[dt]


def get_loss(loss, **kwargs):
    if loss.startswith("weighted"):
        y = kwargs.get("y_train")
        if isinstance(y, th.Tensor):
            y = y.cpu().numpy()
        weights_per_label = compute_labels_density(y)
        print("weights per label", weights_per_label.shape)
        batch_weights = kwargs.get("batch_weights")
        if batch_weights:
            weigths_options = {
                "use_exp": kwargs.get("use_exp"),
                "exp": kwargs.get("exp"),
            }
            weights_per_batch_element = compute_classes_weight(y=y, **weigths_options)
            print("weights per batch", weights_per_batch_element.shape)
        else:
            weights_per_batch_element = None
            print("weights per batch", weights_per_batch_element)

        if torch.cuda.is_available():
            weights_per_label = weights_per_label.cuda()
            if weights_per_batch_element is not None:
                weights_per_batch_element = weights_per_batch_element.cuda()

        if loss == "weighted-1":
            return WeightedBinaryCrossEntropy1(weights_per_batch_element=weights_per_batch_element)
        elif loss == "weighted-3":
            return weighted_binary_cross_entropy3
        elif loss == "weighted-5":
            return WeightedBinaryCrossEntropy5(weights_per_label=weights_per_label,
                                               weights_per_batch_element=weights_per_batch_element)
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


def corr(y, output_path):
    col = [str(i) for i in range(y.shape[1])]
    data = pd.DataFrame(y, columns=col)
    corr = data.corr()
    print(corr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.savefig("{}/{}".format(output_path, "correlation.png"))
    #plt.show()
