from .bmn_ddi import BMNDDI
from .deepddi import DeepDDI
from .mlt_ddi import MLTDDI
from .lee import AutoEncoder, DNN

all_networks_dict = dict(
    deepddi=DeepDDI,
    bmnddi=BMNDDI,
    mltddi=MLTDDI,
    adnn=AutoEncoder,
    dnn=DNN
)
