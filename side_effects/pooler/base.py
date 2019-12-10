from .laplacianpooling import LaplacianPool
from .oraclepooling import OraclePool
from .attnpool import AttnCentroidPool
from .diffpool import DiffPool
from ivbase.nn.graphs.pool import TopKPool, ClusterPool, AttnPool
from functools import partial


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
