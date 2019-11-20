import json
import seaborn as sns
from collections import OrderedDict

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from side_effects.trainer import *


def rename_state_dict_keys(source, keys_to_remove=None):
    state_dict = torch.load(source, map_location="cpu")
    print(state_dict)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        print(key)
        if key not in keys_to_remove:
            new_key = key.split("__")[-1]
            new_state_dict[new_key] = value

    torch.save(new_state_dict, source)


def load_pretrained_model(directory, delete_layers=None, output_dim=86):
    params = json.load(open(f"{directory}/configs.json"))
    model_params = params["model_params"][-1]
    model_params["network_params"].update(dict(output_dim=output_dim))
    model = Trainer(**model_params)
    keys_to_remove = []
    if delete_layers == 'last':
        model.model.classifier.net = nn.Sequential(*list(model.model.classifier.net.children())[:-2])
        keys_to_remove.extend(["classifier.net.1.weight", "classifier.net.1.bias"])
    rename_state_dict_keys(f"{directory}/weights.json", keys_to_remove)
    model.load(f"{directory}/weights.json")
    model.model.eval()
    return model


if __name__ == '__main__':
    from gensim.test.utils import datapath
    from gensim.models import KeyedVectors

    h = KeyedVectors.load_word2vec_format(datapath("/media/rogia/ROGIA/BioWordVec_PubMed_MIMICIII_d200.vec.bin"),
                                          binary=True, limit=1000000, datatype=np.float32)

    # check if the label belongs to the vocab
    print(h.index2word)

    print(h.most_similar(positive=["femur", "fracture"]))

    # Get the vector associated to this word
    print(h.get_vector('blood pressure'))
