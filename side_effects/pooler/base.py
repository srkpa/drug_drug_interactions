from functools import partial

from ivbase.nn.graphs.pool import TopKPool, ClusterPool, AttnPool

from .diffpool import DiffPool
from .laplacianpooling import LaplacianPool


def get_graph_coarsener(**params):
    arch = params.pop("arch", None)
    if arch is None:
        return None
    if arch == "diff":
        pool = partial(DiffPool, **params)
    elif arch == "attn":
        pool = partial(AttnPool, **params)
    elif arch == "topk":
        pool = partial(TopKPool, **params)
    elif arch == "cluster":
        pool = partial(ClusterPool, algo="hierachical", gweight=0.5, **params)
    elif arch == "laplacian":
        init_params = dict(attn=1, concat=False)
        init_params.update(params)
        pool = partial(LaplacianPool, **init_params)
    return pool
