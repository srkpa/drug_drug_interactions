import json
import os
import pickle
import click
from side_effects.data.utils import *


#
# @click.command()
# @click.option('-d', '--dataset', type=str, default='twosides', help='folder to scan')
# @click.option('-a', '--algo', type=str, default='bmnddi')
# @click.option('-o', '--outfile', type=str, default='configs.json'
#
#

def save_results(filename, contents, engine='xlsxwriter'):
    writer = pd.ExcelWriter(filename, engine=engine)
    for sheet, data in contents:
        data.to_excel(writer, sheet_name=sheet)
    writer.save()


def describe_all_experiments(output_folder):
    out = []
    for c in os.listdir(output_folder):
        folder = os.path.join(output_folder, c)
        if os.path.isdir(folder):
            params, res, _, _ = _unpack_results(folder)
            out.append({**params, **res, "task_id": c})

    return pd.DataFrame(out)


def _format_dict_keys(expt_config):
    new_expt_config = {}
    for param, val in expt_config.items():
        new_key = param.split(".")[-1]
        new_expt_config[new_key] = val
    return new_expt_config


def _complete_metrics(all_metrics, keys_to_remove, y_preds, y_true):
    for i in keys_to_remove:
        del all_metrics[i]
    for n, m in all_metrics.items():
        print(n, m(y_preds, y_true))


def _unpack_results(folder):
    expt_config, expt_results, y_true, y_preds = None, None, None, None
    for f in os.listdir(folder):
        filepath = os.path.join(folder, f)
        if os.path.isfile(filepath):
            file_ext = os.path.splitext(filepath)[-1]
            if file_ext == ".pkl":
                out = pickle.load(open(filepath, "rb"))
                if f.endswith("_preds.pkl"):
                    y_preds = out
                elif f.endswith("_res.pkl"):
                    expt_results = out
                elif f.endswith("_targets.pkl"):
                    y_true = out
            elif f.endswith("_params.json"):
                expt_config = json.load(open(filepath, "rb"))

    return expt_config, expt_results, y_true, y_preds
