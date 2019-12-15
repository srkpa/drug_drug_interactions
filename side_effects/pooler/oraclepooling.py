import torch
from ivbase.nn.graphs.pool.base import GraphPool, _cluster_to_matrix, _get_clustering_pool


class OraclePool(GraphPool):
    """
    This layer is designed to be reused for multiple pooling level
    and not just one. You can still use it for one level though.
    
    It's a proof of concept for when the pooling path of a molecule is known in advance
    
    Please note that this implementation is not optimal and can be efficiently done using a tree by 
    taking advantage of the "K-level ancestor of a node" algorithm
    """

    def __init__(self, input_dim, pooling=None, clip=False):
        super().__init__(self, input_dim, None, None)
        self.pooling = _get_clustering_pool(pooling)
        self.clip = clip
        # Not saving the pooling path for all molecules for obvious memory reasons

    @staticmethod
    def get_cluster_at(atom, mol_path, level=1, log_all=False):
        cur_level = 0
        init_atom = atom
        pooling_path = [init_atom]
        while cur_level < level:
            ind = -1
            l_cluster = mol_path[cur_level + 1]
            for i, clist in enumerate(l_cluster):
                if atom in clist:
                    ind = i
                    break
            cur_level += 1
            if ind == -1:
                raise ValueError(
                    "Atom {} of mol {} is not included in any pooling path at level {}".format(init_atom, mol,
                                                                                               cur_level))
            atom = ind
            pooling_path.append(atom)
        if log_all:
            return pooling_path
        return ind

    @staticmethod
    def get_pooling_path(atom, mol_path):
        max_level = max(mol_path.keys())
        return OraclePool.get_cluster_at(atom, mol_path, max_level, log_all=True)

    def compute_clusters(self, *, adj, x, pooling_path, level=1, **kwargs):
        r"""
        Applies the differential pooling layer on input tensor x and adjacency matrix 

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Adjacency matrix of size (B, N, N)
            x: torch.FloatTensor 
                Node feature input tensor containing node embedding of size (B, N, F)
            pooling_path: 
                Pooling path for the molecule
            level:
                level of the pooling path to check, If larger than what is permitted for the molecule or less than 1,
                nothing will be done. Note that the adj graph in input should always be the previous adj graph from the last step
            kwargs:
                Named parameters to be passed to the function computing the clusters
        """
        # Be a bit smarter
        # compute equivalence classes     

        g_size = adj.size(-2)
        max_level = max(pooling_path.keys())
        if level < 1 or level > max_level:  # do nothing here
            return torch.eye(g_size).float()  # projection into itself

        return _cluster_to_matrix(pooling_path[level], device=x.device)

    def _loss(self, *args):
        return 0

    def compute_feats(self, *, mapper, adj, x):
        if self.pooling is not None:
            return self.pooling(mapper, x)
        return torch.matmul(mapper.transpose(-2, -1), x)

    def forward(self, adj, x, pooling_path, level=1, **kwargs):
        r"""
        Applies the Pooling layer on input tensor x and adjacency matrix 

        Arguments
        ----------
            adj: `torch.FloatTensor`
                Unormalized adjacency matrix of size (B, N, N)
            x: torch.FloatTensor 
                Node feature input tensor before embedding of size (B, N, F)
            pooling_path: 
                Pooling path for the molecule
            level:
                level of the pooling path to check, If larger than what is permitted for the molecule or less than 1,
                nothing will be done. Note that the adj graph in input should always be the previous adj graph from the last step
            kwargs:
                Named parameters to be used by the forward computation graph
        Returns
        -------
            new_adj: `torch.FloatTensor`
                New adjacency matrix, describing the reduced graph
            new_feat: `torch.FloatTensor`
                Node features resulting from the pooling operation.

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        new_adj_list = []
        new_feat_list = []
        for k in range(x.shape[0]):
            S = self.compute_clusters(adj=adj[k], x=x[k], pooling_path=pooling_path, level=level, **kwargs)
            new_feat = self.compute_feats(mapper=S, adj=adj[k], x=x[k])
            new_adj = (S.transpose(-2, -1)).matmul(adj[k]).matmul(S)
            self.cur_S = cur_S
            if self.clip:
                new_adj = torch.clamp(new_adj, max=1)
            new_adj_list.append(new_adj)
            new_feat_list.append(new_feat)
        return torch.stack(new_adj_list), torch.stack(new_feat_list)
