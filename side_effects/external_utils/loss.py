import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy


def compute_classes_weight(y):
    return torch.tensor([compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=target)
                         if len(np.unique(target)) > 1 else np.array([1.0, 1.0]) for target in y.T],
                        dtype=torch.float32).t()


class Weighted_binary_cross_entropy1(Module):

    def __init__(self, weights_per_targets=None,
                 reduction='mean', n_pos=1):
        super(Weighted_binary_cross_entropy1, self).__init__()
        self.weights = weights_per_targets
        self.reduction = reduction
        self.n_pos = n_pos

    def forward(self, input, target):
        assert input.shape == target.shape
        assert input.shape[1] == self.weights.shape[1]
        if self.weights is not None:
            assert len(self.weights) == 2
            weights = torch.zeros_like(input)
            zero_w = self.weights[0].unsqueeze(0).expand(*input.shape)
            weights = torch.where(target == 0, zero_w, weights)
            one_w = self.weights[1].unsqueeze(0).expand(*input.shape) * self.n_pos
            weights = torch.where(target == 1, one_w, weights)
        else:
            weights = None

        return binary_cross_entropy(input, target, weight=weights, reduction=self.reduction)


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


def weighted_binary_cross_entropy3(inputs, targets):
    assert inputs.shape == targets.shape
    y = targets.cpu().numpy()
    weights_per_targets = compute_classes_weight(y)
    if torch.cuda.is_available():
        weights_per_targets = weights_per_targets.cuda()
    weights = torch.zeros_like(inputs)
    zero_w = weights_per_targets[0].unsqueeze(0).expand(*inputs.shape)
    weights = torch.where(targets == 0, zero_w, weights)
    one_w = weights_per_targets[1].unsqueeze(0).expand(*inputs.shape)
    weights = torch.where(targets == 1, one_w, weights)
    assert weights.shape == targets.shape
    x = torch.sum(targets, dim=0) / targets.shape[0]
    # assert x.nelement() == targets.shape[1]
    # print(x)
    # a = [binary_cross_entropy(inputs[:, j], targets[:, j], weight=weights[:, j]) for j in range(targets.shape[1])]
    # print(a)
    return x


def test():
    import torch
    import numpy as np
    from numpy.random import binomial, uniform
    from sklearn.utils import compute_class_weight
    batch_size, nb_targets = 32, 10
    outputs = uniform(0, 1, size=(batch_size, nb_targets))
    targets = np.array([binomial(1, uniform(0.1, 0.9), size=batch_size)
                        for _ in range(nb_targets)]).T

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
    test()
