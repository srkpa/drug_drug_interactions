import torch
from torch import nn
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
        node_size: int
            Input dimension of the node feature
        edge_size: int
            Input dimension of the edge feature
        kernel_size : int
            Output dim of the node feature embedding
        eps : float, optional
            (Initial) :math:`\epsilon` value. If set to None, eps will be trained
            (Default value = None)
        init_fn: `torch.nn.init`, optional
            Initialization function to use for the dense layer.
            The default initialization for the `nn.Linear` will be used if this is not provided.
        kwargs:
            Optional named parameters to send to the neural network
    """

    def __init__(self, node_size, edge_size, kernel_sizes, eps=None, init_fn=None, **kwargs):

        super(GINConv, self).__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.kernel_sizes = kernel_sizes
        self.init_fn = init_fn
        if 'normalize' in kwargs:
            kwargs.pop("normalize")
        net_layers, node_edge_net_layers, in_size = [], [], node_size
        for kernel_size in kernel_sizes:
            node_edge_net_layers.append(FCLayer(in_size + edge_size, in_size, **kwargs))
            net_layers.append(FCLayer(in_size, kernel_size, **kwargs))
            in_size = kernel_size
        self._output_dim = in_size
        self.net = nn.Sequential(*net_layers)
        self.node_edge_net = nn.Sequential(*node_edge_net_layers)
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

    def forward(self, x):
        adj, nodes_features, edges_features = x
        new_nodes_features = nodes_features
        for i, _ in enumerate(self.kernel_sizes):
            temp = new_nodes_features.unsqueeze(0).expand(*adj.shape, new_nodes_features.shape[-1])
            # print(new_nodes_features.shape, temp.shape, edges_features.shape)
            nodes_edges = torch.cat((temp, edges_features), dim=2)
            # print(nodes_edges.shape)
            nodes_edges = self.node_edge_net[i](nodes_edges)
            # print(nodes_edges.shape)
            msg = (adj.unsqueeze(-1) * nodes_edges).sum(1)
            # print(new_nodes_features.shape, msg.shape)
            out = ((1 + self.eps) * new_nodes_features) + msg
            new_nodes_features = self.net[i](out)
        return new_nodes_features


    @property
    def output_dim(self):
        return self._output_dim
