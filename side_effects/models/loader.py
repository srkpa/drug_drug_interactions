import json
from side_effects.trainer import Trainer
import torch
from collections import OrderedDict


def rename_state_dict_keys(source):
    state_dict = torch.load(source, map_location="cpu")
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key.split("__")[-1]
        new_state_dict[new_key] = value

    torch.save(new_state_dict, source)


def load_pretrained_model(dir):
    params = json.load(open(f"{dir}/configs.json"))
    model_params = params["model_params"][-1]
    model_params["network_params"].update(dict(output_dim=model_params["network_params"].get("output_dim", 86)))
    model = Trainer(**model_params)
    rename_state_dict_keys(f"{dir}/weights.json")
    model.load(f"{dir}/weights.json")
    model.model.eval()
    return model

