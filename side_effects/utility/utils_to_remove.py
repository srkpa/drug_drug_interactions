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


def label_distribution(y):
    return (np.sum(y, axis=0) / y.shape[0]) * 100


def get_loss(loss, **kwargs):
    if loss.startswith("weighted"):
        y = kwargs.get("y_train")
        if isinstance(y, torch.Tensor):
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


def test():
    import torch
    import numpy as np
    from numpy.random import binomial, uniform
    from sklearn.utils import compute_class_weight
    batch_size, nb_targets = 32, 10
    outputs = uniform(0, 1, size=(batch_size, nb_targets))
    targets = np.array([binomial(1, uniform(0.1, 0.9), size=batch_size)
                        for _ in range(nb_targets)]).T

    targets = np.array([1, 1, 0, 0, 0])
    print(compute_class_weight(class_weight=None, classes=np.array([0, 1]), y=targets))
    exit()
    # Here you should compute the class_weights using the targets for all your training data
    w = torch.tensor([
        compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=target)
        for target in targets.T], dtype=torch.float32).t()
    outputs = torch.Tensor(outputs)
    targets = torch.Tensor(targets)

    # I have a preference for 1 over 2 because 1 is numerically more stable since it use bce function from pytorch
    loss1 = weighted_binary_cross_entropy1(outputs, targets, w)
    loss2 = weighted_binary_cross_entropy2(outputs, targets, w)
    loss3 = binary_cross_entropy(outputs, targets)
    print(loss1, loss2, loss3)

def pie(file_path="/home/rogia/Téléchargements/twosides-icd11.csv"):
    data = pd.read_csv(file_path, sep=",")
    icd2se = data.groupby(["icd"])
    fig, axs = plt.subplots(5, 5)
    g = plt.colormaps()
    i = 0
    for name, group in icd2se:
        labels = list(icd2se.groups[name])
        print(len(labels), len(set(labels)))
        sizes = [1] * len(labels)
        if 0 <= i < 5:
            ax1 = axs[0, i]
        elif 5 <= i < 10:
            ax1 = axs[1, i - 5]
        elif 10 <= i < 15:
            ax1 = axs[2, i - 10]
        elif 15 <= i < 20:
            ax1 = axs[3, i - 15]
        else:
            ax1 = axs[4, i - 20]
        cm = plt.get_cmap(g[i])
        cout = cm(np.arange(len(labels)))
        ax1.pie(sizes, labels=None, startangle=90, colors=cout)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.axis('equal')
        ax1.set_title(name)
        plt.tight_layout()

        i += 1
    plt.show()


# def plot_scatter(x, colors):
#     # choose a color palette with seaborn.
#     num_classes = len(colors)
#     col = np.arange(0, num_classes)
#     palette = np.array(sns.color_palette("hls", num_classes))
#
#     # create a scatter plot.
#     f = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(aspect='equal')
#     sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[col.astype(np.int)])
#     plt.xlim(-25, 25)
#     plt.ylim(-25, 25)
#     ax.axis('off')
#     ax.axis('tight')
#
#     # add the labels for each digit corresponding to the label
#     txts = []
#
#     for i in range(num_classes):
#         print(colors[i])
#         # Position of each label at median of data points.
#         xtext, ytext = np.median(x[col == i, :], axis=0)
#         txt = ax.text(xtext, ytext, str(colors[i]), fontsize=6)
#         txt.set_path_effects([
#             PathEffects.Stroke(linewidth=5, foreground="w"),
#             PathEffects.Normal()])
#         txts.append(txt)
#     plt.show()
#     return f, ax, sc, txts
#
#
# if __name__ == '__main__':
#     import pandas as pd
#     m = load_pretrained_model(
#         "/home/rogia/Téléchargements/twosides", output_dim=964)
#     a = list(m.model.classifier.net.children())[-2].weight.data.cpu()
#     data_set = pd.read_csv("/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/twosides.csv", sep=",")
#     l = [e for l in data_set["ddi type"].values.tolist() for e in l.split(";")]
#     g = list(set(l))
#     g.sort()
#     from collections import Counter
#     ab = dict(Counter(l))
#     from sklearn.manifold import TSNE
#     m = [ab[e] for e in g]
#     s = TSNE(random_state=42, n_components=2).fit_transform(a.numpy())
#     plot_scatter(s, m)

if __name__ == '__main__':
    pie()

if __name__ == '__main__':
    # test_ddi()
    import torch
    import numpy as np
    from numpy.random import binomial, uniform
    from sklearn.utils import compute_class_weight
    from side_effects.utility.utils import compute_classes_weight

    batch_size, nb_targets = 32, 10
    outputs = uniform(0, 1, size=(batch_size, nb_targets))
    targets = np.array([binomial(1, uniform(0.1, 0.9), size=batch_size)
                        for _ in range(nb_targets)]).T

    print(compute_classes_weight(targets, use_exp=False, exp=1))
    print(compute_classes_weight(targets, use_exp=True, exp=0.5))
