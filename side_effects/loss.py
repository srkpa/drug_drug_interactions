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

    def __init__(self, use_negative_sampling=False, weight=None, density=None,
                 use_binary_cost_per_batch=None, use_label_cost_per_batch=None, loss="bce", neg_rate=1.,
                 use_sampling=False,
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
        self.is_multioutput = is_mtk
        if self.is_multioutput:
            self.losses_fn = [self.loss_fn] + [get_loss_or_metric("mse")]

    def forward(self, pred_y, true_y):
        # print(inputs)
        #print("preds", pred_y[0].shape, pred_y[1].shape)
        #print("ouptu", true_y[0].shape, true_y[1].shape)
        #
        # print("ouptu", true_y[0].shape, true_y[1].shape)

        if self.is_multioutput:
            true_y = [y.view(-1, y.size(-1)) for y in true_y]
            masks = [(y[:, 0] == y[:, 0]) for y in true_y]
            #masks = [mask for mask in masks ]
            #print(true_y, masks)
            true_y = [true_y[i][mask, :] for i, mask in enumerate(masks) if sum(mask) > 0]
            pred_y = [pred_y[i][mask, :] for i, mask in enumerate(masks) if sum(mask) > 0]
            #print(pred_y, masks)
            #print("y_true", true_y[0].shape, true_y[1].shape)
           # print("val after mask", true_y[0], true_y[1])
          #  print("pred after mask", pred_y[0].shape, pred_y[1].shape)
            loss = sum([self.losses_fn[i](y_pred, y_true) for i, (y_pred, y_true) in enumerate(zip(pred_y, true_y))])

        elif self.use_negative_sampling and self.training:
            #mask = (true_y == 1).float()
            mask2 = (true_y == 1)
            for i, col in enumerate(true_y.t()):
                nb_pos = int(col.sum())
                if nb_pos > 0:
                    idx = Categorical((col == 0).float().flatten()).sample((int(nb_pos * self.neg_rate),))
                    #mask[idx, i] = 1.0
                    mask2[idx, i] = True

            loss = binary_cross_entropy(pred_y[mask2], true_y[mask2])
            #loss2 = (binary_cross_entropy(pred_y, true_y, reduction='none') * mask).mean()
            #print("new loss ", loss, "old loss ", loss2)

        elif self.use_weighted_loss and self.training:
            loss = WeightedBinaryCrossEntropy1(**self.weighted_loss_params)(pred_y, true_y)

        elif self.use_sampling and self.training: # a little bit conf
            mask = (true_y == 1).float()
            for i, row in enumerate(true_y):
                nb_neg = int(row.sum() * self.neg_rate)
                if nb_neg > 0:
                    idx = Categorical((row == 0).float().flatten()).sample((nb_neg,))
                    mask[i, idx] = 1.0
            loss = (self.samp_weight * (binary_cross_entropy(pred_y, true_y, reduction='none') * mask).mean()).sum()

        elif self.rescale_freq and self.training:
            loss = (self.samp_weight * binary_cross_entropy(pred_y, true_y, reduction='none').mean(dim=0)).sum()

        else:
            loss = self.loss_fn(pred_y, true_y)  # hinge_embedding_loss(input, target) #

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
