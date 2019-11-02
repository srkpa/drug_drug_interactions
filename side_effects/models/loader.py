import json
from side_effects.trainer import *
import torch
import torch.nn as nn
from collections import OrderedDict


def rename_state_dict_keys(source):
    state_dict = torch.load(source, map_location="cpu")
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key.split("__")[-1]
        new_state_dict[new_key] = value

    torch.save(new_state_dict, source)


def load_pretrained_model(directory, delete_layers=None, output_dim=86):
    params = json.load(open(f"{directory}/configs.json"))
    model_params = params["model_params"][-1]
    model_params["network_params"].update(dict(output_dim=output_dim))
    model = Trainer(**model_params)
    if delete_layers == 'last':
        model.model.classifier.net = nn.Sequential(*list(model.model.classifier.net.children())[:-1])
    rename_state_dict_keys(f"{directory}/weights.json")
    model.load(f"{directory}/weights.json")
    model.model.eval()
    return model


if __name__ == '__main__':
    m = load_pretrained_model(
        "/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/pretrained_models/random/drugbank",
        delete_layers='last')
    print(m.model.classifier.net)

    print(m.model)
