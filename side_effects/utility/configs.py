import os
import tempfile
import json
import shutil
from ivbase.utils.aws import aws_cli
from collections import defaultdict


def a(s3_paths, output_path):
    tmp_dir = tempfile.mkdtemp()
    for i, s3_path in enumerate(s3_paths):
        aws_cli(['s3', 'sync', s3_path, tmp_dir, '--quiet'])
    for out_file in os.listdir(tmp_dir):
        path = os.path.join(tmp_dir, out_file)
        if out_file.endswith(".zip"):
            print(out_file)
            shutil.unpack_archive(path, extract_dir=os.path.join(output_path, out_file))

    return output_path


def merge_config_files(config_path, output_path):
    configs_params = defaultdict(list)
    i = 0
    j = 0
    for file in os.listdir(config_path):
        path = os.path.join(config_path, file)
        if file.endswith(".json"):
            param = json.load(open(path, "r"))
            if param["model_params"]["network_params"]["att_hidden_dim"] is None:
                for param, val in param.items():
                    if val not in configs_params[param]:
                        configs_params[param].append(val)
                i += 1
            j += 1
    print("Total json files= ", i, j)
    json.dump(configs_params, open(os.path.join(output_path, "configs.json"), "w"), indent=2)


if __name__ == '__main__':
    merge_config_files("/home/rogia/Documents/analysis/results/graph_drug_prototype",
                       output_path="/home/rogia/Documents/git/drug_drug_interactions/expts/gdp_random")
