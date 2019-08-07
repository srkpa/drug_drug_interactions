from .deepddi import DeepDDI
from .bmn_ddi import BMNDDI

all_networks_dict = dict(
    deepddi=DeepDDI,
    bmnddi=BMNDDI,
)