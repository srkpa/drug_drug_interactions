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
    WeightedBinaryCrossEntropy2
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


def weighted_binary_cross_entropy1(output, target, weights_per_targets=None,
                                   reduction='elementwise_mean'):
    r"""Function that measures the Binary Cross Entropy
    between the target and the output for a multi-target binary classification.

    Args:
        output: FloatTensor of shape N * D
        target: IntTensor of the same shape as input
        weights_per_targets: FloatTensor of shape 2 * D
            The first row of this tensor represent the weights of 0 for all targets,
            The second row of this tensor represent the weights of 1 for all targets
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
    """
    assert output.shape == target.shape
    assert output.shape[1] == weights_per_targets.shape[1]
    if weights_per_targets is not None:
        assert len(weights_per_targets) == 2
        weights = torch.zeros_like(output)
        zero_w = weights_per_targets[0].unsqueeze(0).expand(*output.shape)
        weights = torch.where(target == 0, zero_w, weights)
        one_w = weights_per_targets[1].unsqueeze(0).expand(*output.shape)
        weights = torch.where(target == 1, one_w, weights)
    else:
        weights = None

    return binary_cross_entropy(output, target, weight=weights, reduction=reduction)


def weighted_binary_cross_entropy2(output, target, weights_per_targets=None,
                                   reduction='elementwise_mean'):
    r"""Function that measures the Binary Cross Entropy
    between the target and the output for a multi-target binary classification.

    Args:
        output: FloatTensor of shape N * D
        target: IntTensor of the same shape as input
        weights_per_targets: FloatTensor of shape 2 * D
            The first row of this tensor represent the weights of 0 for all targets,
            The second row of this tensor represent the weights of 1 for all targets
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
    """
    if weights_per_targets is not None:
        loss = weights_per_targets[1] * (target * torch.log(output)) + \
               weights_per_targets[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def unbalanced_classes_weights(y):
    weights = non_zeros_distribution(y)
    return torch.from_numpy(weights / y.shape[0])


class WeightedBinaryCrossEntropy2(Module):

    def __init__(self, weights_per_label=None, weights_per_batch_element=None):
        super(WeightedBinaryCrossEntropy2, self).__init__()
        self.weights_per_label = weights_per_label
        self.criterion = WeightedBinaryCrossEntropy1(weight=weights_per_batch_element,
                                                     reduction='none')

    def forward(self, input, target):
        batch_losses = self.criterion(input, target)
        assert batch_losses.shape == target.shape
        mean_per_targets = torch.mean(batch_losses, dim=0)
        assert mean_per_targets.nelement() == target.shape[1]
        return torch.sum(self.weights_per_label * mean_per_targets)


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
            return WeightedBinaryCrossEntropy1(weight=weights_per_batch_element)
        elif loss == "weighted-3":
            return weighted_binary_cross_entropy3
        elif loss == "weighted-5":
            return WeightedBinaryCrossEntropy2(weights_per_label=weights_per_label,
                                               weights_per_batch_element=weights_per_batch_element)
    return get_loss_or_metric(loss)


def weighted_binary_cross_entropy3(inputs, target):
    assert inputs.shape == target.shape
    y = target.cpu().numpy()
    weights_per_targets = compute_classes_weight(y)
    if torch.cuda.is_available():
        weights_per_targets = weights_per_targets.cuda()
    weights = torch.zeros_like(inputs)
    zero_w = weights_per_targets[0].unsqueeze(0).expand(*inputs.shape)
    weights = torch.where(target == 0, zero_w, weights)
    one_w = weights_per_targets[1].unsqueeze(0).expand(*inputs.shape)
    weights = torch.where(target == 1, one_w, weights)
    assert weights.shape == target.shape

    return binary_cross_entropy(inputs, target, weight=weights, reduction='mean')


class WeightedBinaryCrossEntropy4(Module):

    def __init__(self, weights_per_targets=None):
        super(WeightedBinaryCrossEntropy4, self).__init__()
        self.weights = weights_per_targets

    def forward(self, input, target):
        assert input.shape == target.shape
        assert input.shape[1] == self.weights.shape[1]
        # if self.weights is not None:
        assert len(self.weights) == 2
        weights = torch.zeros_like(input)
        zero_w = self.weights[0].unsqueeze(0).expand(*input.shape)
        weights = torch.where(target == 0, zero_w, weights)
        one_w = self.weights[1].unsqueeze(0).expand(*input.shape)
        weights = torch.where(target == 1, one_w, weights)

        losses = binary_cross_entropy(input, target, reduction='none')

        return torch.sum(weights * losses).mean()


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
    # plt.show()
