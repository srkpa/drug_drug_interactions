from torch.nn import Module
import torch
from torch.distributions.categorical import Categorical
from torch.nn import Module
from torch.nn import Parameter
from torch.nn.functional import binary_cross_entropy
from side_effects.data.loader import compute_labels_density, compute_classes_weight


class WeightedBinaryCrossEntropy1(Module):

    def __init__(self, batch_weight, batch_density, weight, density):
        super(WeightedBinaryCrossEntropy1, self).__init__()
        self.weights = weight
        self.label_density = density
        self.binary_batch_weight = batch_weight
        self.batch_weighted_labels = batch_density

    def forward(self, inputs, target):
        assert inputs.shape == target.shape
        if self.binary_batch_weight:
            self.set_weights(target)
        assert inputs.shape[1] == self.weights.shape[1]
        assert len(self.weights) == 2
        weights = torch.zeros_like(inputs)
        zero_w = self.weights[0].unsqueeze(0).expand(*inputs.shape)
        weights = torch.where(target == 0, zero_w, weights)
        one_w = self.weights[1].unsqueeze(0).expand(*inputs.shape)
        weights = torch.where(target == 1, one_w, weights)
        loss = binary_cross_entropy(inputs, target, weight=weights, reduction='none')
        if self.batch_weighted_labels:
            self.set_density(target)
        if isinstance(self.label_density, torch.Tensor):
            loss = (self.label_density * loss.mean(dim=0)).sum()
        else:
            loss = loss.mean()  # binary_cross_entropy(inputs, target, weight=weights, reduction='mean')
        return loss

    def set_weights(self, target):
        y = target.cpu().numpy()
        weight = compute_classes_weight(y)
        if torch.cuda.is_available():
            weight = weight.cuda()
        self.weights = weight

    def set_density(self, target):
        y = target.cpu().numpy()
        density = compute_labels_density(y)
        if torch.cuda.is_available():
            density = density.cuda()
        self.label_density = density


class BinaryCrossEntropyP(Module):

    def __init__(self, use_negative_sampling, weight, density,
                 use_binary_cost_per_batch, use_label_cost_per_batch):
        super(BinaryCrossEntropyP, self).__init__()
        self.use_negative_sampling = use_negative_sampling
        self.weighted_loss_params = dict(batch_weight=use_binary_cost_per_batch,
                                         batch_density=use_label_cost_per_batch,
                                         weight=weight,
                                         density=density)
        print(self.weighted_loss_params)
        self.use_weighted_loss = any(
            [isinstance(val, torch.Tensor) or val is True for val in list(self.weighted_loss_params.values())])

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
            loss = WeightedBinaryCrossEntropy1(**self.weighted_loss_params)(input, target)
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
