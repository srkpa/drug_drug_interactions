from .deepddi import DeepDDI
from .bmn_ddi import BMNDDI
from .decagon import DecagonModel
from .mlt_ddi import MLTDDI
all_networks_dict = dict(
    deepddi=DeepDDI,
    bmnddi=BMNDDI,
    decagon=DecagonModel,
    mltddi=MLTDDI
)