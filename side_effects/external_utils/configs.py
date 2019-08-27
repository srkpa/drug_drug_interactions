import os
import json
from collections import defaultdict


def merge_config_files(config_path, output_path):
    configs_params = defaultdict(list)
    i = 0
    for file in os.listdir(config_path):
        path = os.path.join(config_path, file)
        if file.endswith(".json"):
            param = json.load(open(path, "r"))
            if param["model_params"]["network_params"]["att_hidden_dim"] is None:
                for param, val in param.items():
                    if val not in configs_params[param]:
                        configs_params[param].append(val)
                i += 1
    print("Total json files= ", i)
    json.dump(configs_params, open(os.path.join(output_path, "configs.json"), "w"), indent=2)


if __name__ == '__main__':
    merge_config_files("/home/rogia/Documents/analysis/results/graph_drug_prototype",
                       output_path="/home/rogia/Documents/git/drug_drug_interactions/expts/gdp")
