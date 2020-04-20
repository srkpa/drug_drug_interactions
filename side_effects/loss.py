import torch
from poutyne.framework.metrics.metrics import get_loss_or_metric
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
                 use_binary_cost_per_batch, use_label_cost_per_batch, loss='bce', neg_rate=1., use_sampling=False,
                 samp_weight=1., rescale_freq=False, is_mtk=False):
        super(BinaryCrossEntropyP, self).__init__()
        self.loss_fn = get_loss_or_metric(loss)
        self.use_negative_sampling = use_negative_sampling
        self.weighted_loss_params = dict(batch_weight=use_binary_cost_per_batch,
                                         batch_density=use_label_cost_per_batch,
                                         weight=weight,
                                         density=density)
        # print(self.weighted_loss_params, neg_rate, use_sampling, samp_weight)
        self.use_weighted_loss = any(
            [isinstance(val, torch.Tensor) or val is True for val in list(self.weighted_loss_params.values())])
        self.neg_rate = neg_rate
        self.use_sampling = use_sampling
        self.samp_weight = samp_weight
        self.rescale_freq = rescale_freq
        self.is_mtk = is_mtk
        if self.is_mtk:
            self.add_loss = get_loss_or_metric('mse')

    def forward(self, inputs, outputs):
        #print(inputs)
        target, output_scores = outputs[0], outputs[1].squeeze() if self.is_mtk else outputs,
        input, input_scores = inputs[0], inputs[1] if self.is_mtk else inputs,
       # print(input.shape, target.shape)
        assert input.shape == target.shape
        if self.use_negative_sampling and self.training:
            mask = (target == 1).float()
            for i, col in enumerate(target.t()):
                nb_pos = int(col.sum())
                if nb_pos > 0:
                    idx = Categorical((col == 0).float().flatten()).sample((int(nb_pos * self.neg_rate),))
                    mask[idx, i] = 1.0
            loss = (binary_cross_entropy(input, target, reduction='none') * mask).mean()
        elif self.use_weighted_loss and self.training:
            loss = WeightedBinaryCrossEntropy1(**self.weighted_loss_params)(input, target)
        elif self.use_sampling and self.training:
            mask = (target == 1).float()
            for i, row in enumerate(target):
                nb_neg = int(row.sum() * self.neg_rate)
                if nb_neg > 0:
                    idx = Categorical((row == 0).float().flatten()).sample((nb_neg,))
                    mask[i, idx] = 1.0
            loss = (self.samp_weight * (binary_cross_entropy(input, target, reduction='none') * mask).mean()).sum()
        elif self.rescale_freq and self.training:
            loss = (self.samp_weight * binary_cross_entropy(input, target, reduction='none').mean(dim=0)).sum()

        else:
            loss = self.loss_fn(input, target)  # hinge_embedding_loss(input, target) #

        if self.is_mtk:
            loss += self.add_loss(input_scores, output_scores)
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
