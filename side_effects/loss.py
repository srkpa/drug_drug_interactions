from torch.nn import Module
from torch.distributions.categorical import Categorical
from torch.nn import Module
from torch.nn import Parameter
from torch.nn.functional import binary_cross_entropy


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


class WeightedBinaryCrossEntropy1(Module):

    def __init__(self, weight=None,
                 reduction='mean', label_density=None):
        super(WeightedBinaryCrossEntropy1, self).__init__()
        self.weights = weight
        self.label_density = label_density

    def forward(self, inputs, target):
        assert inputs.shape == target.shape
        assert inputs.shape[1] == self.weights.shape[1]
        if self.weights is None:
            self.get_weights(target)
        assert len(self.weights) == 2
        weights = torch.zeros_like(inputs)
        zero_w = self.weights[0].unsqueeze(0).expand(*inputs.shape)
        weights = torch.where(target == 0, zero_w, weights)
        one_w = self.weights[1].unsqueeze(0).expand(*inputs.shape)
        weights = torch.where(target == 1, one_w, weights)
        loss = binary_cross_entropy(inputs, target, weight=weights, reduction='none')
        if self.label_density:
            loss = (self.label_density * loss.mean(dim=0)).sum()
        else:
            loss = loss.mean()  # binary_cross_entropy(inputs, target, weight=weights, reduction='mean')
        return loss

    def get_weights(self, target):
        y = target.cpu().numpy()
        weight = compute_classes_weight(y)
        if torch.cuda.is_available():
            weight = weight.cuda()
        self.weights = weight


class BinaryCrossEntropyP(Module):

    def __init__(self, use_negative_sampling, use_weighted_loss=False):
        super(BinaryCrossEntropyP, self).__init__()
        self.use_negative_sampling = use_negative_sampling
        self.use_weighted_loss = use_weighted_loss

    def forward(self, input, target):
        assert input.shape == target.shape
        if self.use_negative_sampling and self.training:
            mask = (target == 1).float()
            for i, col in enumerate(target.t()):
                nb_pos = int(col.sum())
                if nb_pos > 0:
                    idx = Categorical((col == 0).float().flatten()).sample((nb_pos,))
                    mask[idx, i] = 1.0
            loss = (binary_cross_entropy(input, target, reduction='none') * mask).mean()

        elif self.use_weighted_loss and self.training:
            loss = WeightedBinaryCrossEntropy1()
        else:
            loss = binary_cross_entropy(input, target)
        return loss


class ULoss(Module):
    def __init__(self, nb_tasks):
        super(ULoss, self).__init__()
        self.weights = Parameter(torch.ones(nb_tasks))

    def forward(self, input, target):
        loss = binary_cross_entropy(input, target, reduction='none')
        a = torch.mean(loss, dim=0)
        b = target.sum(dim=0)
        b[b == 0] = 1
        return (a / b).sum()


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
