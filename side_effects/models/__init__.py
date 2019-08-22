from .deepddi import DeepDDI
from .bmn_ddi import BMNDDI
from .decagon import DecagonModel

all_networks_dict = dict(
    deepddi=DeepDDI,
    bmnddi=BMNDDI,
    decagon=DecagonModel
)