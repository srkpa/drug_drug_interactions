from .deepddi import DeepDDI
from .bmn_ddi import BMNDDI, BMNDDI_with_Attention

all_networks_dict = dict(
    deepddi=DeepDDI,
    bmnddi=BMNDDI,
)