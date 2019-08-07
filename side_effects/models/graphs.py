import torch
from torch import nn
from ivbase.nn.commons import get_pooling, get_activation
from ivbase.nn.base import FCLayer


class GINConv(nn.Module):
    r"""
    This layer implements the graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{t+1}_i = NN_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x^{t}}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x^{t+1}}_j \right),

    where :math:`NN_{\mathbf{\Theta}}` denotes a neural network.

    Arguments
    ---------
        in_size: int
            Input dimension of the node feature
        kernel_size : int
            Output dim of the node feature embedding
        eps : float, optional
            (Initial) :math:`\epsilon` value. If set to None, eps will be trained
            (Default value = None)
        net : `nn.Module`, optional
            Neural network :math:`NN_{\mathbf{\Theta}}`.
            If not provided, `ivbase.nn.base.FCLayer` will be used
            (Default value = None)
        init_fn: `torch.nn.init`, optional
            Initialization function to use for the dense layer.
            The default initialization for the `nn.Linear` will be used if this is not provided.
        kwargs:
            Optional named parameters to send to the neural network
    """

    def __init__(self,  in_size, kernel_size, eps=None, net=None, init_fn=None, **kwargs):

        super(GINConv, self).__init__()
        self.in_size = in_size
        self.out_size = kernel_size
        self.init_fn = init_fn
        if 'normalize' in kwargs:
            kwargs.pop("normalize")
        self.net = (net or FCLayer)(in_size, kernel_size, **kwargs)
        self.chosen_eps = eps
        if eps is None:
            self.eps = torch.nn.Parameter(torch.Tensor([0]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Initialize weights of the models, as defined by the input initialization scheme.

        Arguments
        ----------
            init_fn (callable, optional): Function to initialize the linear weights. If it is not provided
                an attempt to use the object `self.init_fn` would be first made, before doing nothing.
                (Default value = None)

        See Also
        --------
            `nn.init` for more information
        """
        chosen_eps = self.chosen_eps or 0
        if not isinstance(self.eps, nn.Parameter):
            self.eps.data.fill_(chosen_eps)
        try:
            self.net.reset_parameters(self.init_fn)
        except:
            pass

    def forward(self, adj, nodes_features, edges_features):
        r"""
        Compute the output of the layer

        Arguments
        ----------
            adj:
            nodes_features:
            edges_features:

        Returns
        -------
        """

        temp = nodes_features.unsqueeze(0).expand(*adj.shape, nodes_features.shape[-1])
        nodes_edges = torch.cat((temp, edges_features), dim=2)
        nodes_edges = self.node_edge_net(nodes_edges)
        msg = (adj.unsqueeze(-1) * nodes_edges).sum(1)
        out = ((1 + self.eps) * self.node_net(nodes_features)) + msg
        out = self.net(out)
        return adj, out, edges_features
