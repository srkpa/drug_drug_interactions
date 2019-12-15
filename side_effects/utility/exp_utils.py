import glob
import json
import os
import pickle
import shutil
import tempfile
from collections import Counter
from pickle import load

import click
import matplotlib.pyplot as plt
import pandas as pd
from ivbase.utils.aws import aws_cli

from side_effects.metrics import auprc_score, roc_auc_score

INVIVO_RESULTS = os.environ["INVIVO_RESULTS_ROOT"]


@click.group()
def cli():
    pass


def download_results(s3_paths, output_path):
    tmp_dir = tempfile.mkdtemp()
    for i, s3_path in enumerate(s3_paths):
        aws_cli(['s3', 'sync', s3_path, tmp_dir, '--quiet'])
    for out_file in os.listdir(tmp_dir):
        path = os.path.join(tmp_dir, out_file)
        if out_file.endswith(".zip"):
            print(out_file)
            shutil.unpack_archive(path, extract_dir=os.path.join(INVIVO_RESULTS, output_path, out_file))

    return output_path


def get_expt_results(output_folder, opt=1):
    # This is an old version of how i preprocess my exp results -- add some refresh
    out = []
    for task_id in os.listdir(output_folder):
        folder = os.path.join(output_folder, task_id)
        if os.path.isdir(folder):
            params, res, _, _ = _unpack_results(folder)
            if res:
                out.append({**params, **res, "task_id": task_id})
    return pd.DataFrame(out)


def _format_dict_keys(expt_config):
    new_expt_config = {}
    for param, val in expt_config.items():
        new_key = param.split(".")[-1]
        new_expt_config[new_key] = val
    return new_expt_config


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


def visualize_loss_progress(filepath):
    df = pd.read_table(filepath, sep="\t")
    plt.figure()
    df.plot(x="epoch", y=["loss", "val_loss"])
    plt.show()


def scorer(dataset_name, exp_folder=None, upper_bound=None):
    data_set = pd.read_csv(f"/home/rogia/.invivo/cache/datasets-ressources/DDI/{dataset_name}/{dataset_name}.csv",
                           sep=",")
    l = [e for l in data_set["ddi type"].values.tolist() for e in l.split(";")]
    ab = dict(Counter(l))
    print(ab)
    g = list(set(l))
    print(g)
    g.sort()
    files = os.listdir(exp_folder)
    y_preds = load(open(os.path.join(exp_folder, [f for f in files if f.endswith("_preds.pkl")][-1]),
                        "rb"))  # load(open(os.path.join(exp_folder, "predicted_labels.pkl"),"rb"))
    y_true = load(open(os.path.join(exp_folder, [f for f in files if f.endswith("_targets.pkl")][-1]),
                       "rb"))  # load(open(os.path.join(exp_folder, "true_labels.pkl"), "rb"))  #  #

    ap_scores = auprc_score(y_pred=y_preds, y_true=y_true, average=None)
    rc_scores = roc_auc_score(y_pred=y_preds, y_true=y_true, average=None)
    # first position
    pos = list(ap_scores).index(max(ap_scores))
    print(ap_scores[pos])
    print(rc_scores[pos])
    print(g[pos])
    print(ab[g[pos]])

    pos = list(ap_scores).index(min(ap_scores))
    print(ap_scores[pos])
    print(rc_scores[pos])
    print(g[pos])
    print(ab[g[pos]])

    # zoom
    x = [ab[g[i]] for i in range(len(ap_scores))]
    if upper_bound:
        x, y = [j for i, j in enumerate(x) if j <= 2000], [i for i, j in enumerate(x) if j <= 2000]
        ap_scores = [ap_scores[i] for i in y]
        rc_scores = [rc_scores[i] for i in y]

    # Plots
    # plt.figure(figsize=(12, 5))
    # plt.subplot(131)
    # plt.plot(x, ap_scores, "ro")
    # plt.legend()
    # plt.subplot(132)
    # plt.plot(x, rc_scores, "bo", )
    # plt.legend()
    # plt.subplot(133)
    plt.figure(figsize=(8, 6))
    plt.plot(x, ap_scores, "r.", label="AUC-PR")
    plt.plot(x, rc_scores, "b.", label="AUC-ROC")
    plt.xlabel('Number of samples', fontsize=10)
    plt.ylabel('Scores', fontsize=10)
    plt.legend()
    #  plt.savefig("scores_distribution_drugbank.png")
    plt.show()


@cli.command()
@click.option('-n', '--exp_name', type=str, default='test', help="Unique name for the experiment.")
@click.option('-p', '--mm', type=str, default='micro',
              help="""Position of the config file to run""")
@click.option('--keep_only', '-k', type=str, default='best',
              help="Save only task with the best mm for a given exp")
def summarize(exp_name=None, keep_only="best", mm="micro", save=True):  # Need to be fiktered by expt name
    exp_folder = f"{os.path.expanduser('~')}/expts"
    o = []
    for root in os.listdir(exp_folder):
        path = os.path.join(exp_folder, root)
        if os.path.isdir(path):
            print("Path:", path)
            config = json.load(open(glob.glob(path + "/configs/*.done")[-1], "rb")) if os.path.exists(
                path + "/configs") else json.load(open(path + "/config.json", "rb"))
            model_params = config["model_params"]["network_params"].get("drug_feature_extractor_params", None)
            exp = model_params.get("arch") if model_params else config["model_params"]["network_params"]["network_name"]
            print("Arch", exp)
            if exp == "lstm":
                exp = "BiLSTM"
            elif exp == "dglgraph":
                if not model_params["pool_arch"].get("arch"):
                    exp = "MolGraph"
                else:
                    exp = "MolGraph + LaPool"
            elif exp == "conv1d":
                exp = "CNN"
                gp = config["model_params"]["network_params"].get("graph_network_params", None)
                if gp:
                    exp = "CNN + IntGraph"
            elif exp == "deepddi":
                exp = "DeepDDI"
            else:
                exp = ""

            print("Model name", exp)
            max_mm = 0
            inst = {}
            for fp in glob.glob(f"{path}/*.pkl"):
                task_id, _ = os.path.splitext(
                    os.path.split(fp)[-1])  # need to save a pd object to handle many dataset, seed and split mode
                if task_id.endswith(
                        "_res"):  # -res case??  #dataset = config["dataset_params"].get("dataset_name") mode = config["dataset_params"].get("split_mode") seed = config["dataset_params"].get("seed")
                    params = os.path.join(path, task_id[:-3] + "params.json")
                    if os.path.exists(params):
                        res = pickle.load(open(fp, "rb"))
                        mm_val = res[f"{mm}_auprc"]
                        if mm_val > max_mm:
                            print("Config file:", params, "result file:", fp)
                            if True: #"macro_roc" not in res:
                                res["macro_roc"] = roc_auc_score(
                                    y_pred=load(open(os.path.join(path, task_id[:-3] + "preds.pkl"), "rb")),
                                    y_true=load(open(os.path.join(path, task_id[:-3] + "targets.pkl"), "rb")),
                                    average="macro")
                                res["macro_auprc"] = auprc_score(
                                    y_pred=load(open(os.path.join(path, task_id[:-3] + "preds.pkl"), "rb")),
                                    y_true=load(open(os.path.join(path, task_id[:-3] + "targets.pkl"), "rb")),
                                    average="macro")
                                res["micro_roc"] = roc_auc_score(
                                    y_pred=load(open(os.path.join(path, task_id[:-3] + "preds.pkl"), "rb")),
                                    y_true=load(open(os.path.join(path, task_id[:-3] + "targets.pkl"), "rb")),
                                    average="micro")
                                res["micro_auprc"] = auprc_score(
                                    y_pred=load(open(os.path.join(path, task_id[:-3] + "preds.pkl"), "rb")),
                                    y_true=load(open(os.path.join(path, task_id[:-3] + "targets.pkl"), "rb")),
                                    average="micro")
                            max_mm = mm_val
                            inst = res
            o.append({**inst, "model": exp})
    out = pd.DataFrame(o)
    print(out)
    if save:
        out.to_csv("test.xlsx")
    return out


if __name__ == '__main__':
    summarize()
    # # display("/home/rogia/Documents/exp_lstm/_hp_0/twosides_bmnddi_8aa095b0_log.log")
    # # display(dataset_name="drugbank", exp_folder="/home/rogia/Documents/no_e_graph/_hp_0")
    # # display(dataset_name="twosides", exp_folder="/home/rogia/Documents/exp_lstm/_hp_0/")
    # # display_H("/home/rogia/Documents/graph_mol/drugbank_bget_expt_results("/home/rogia/Documents/graph_mol/")mnddi_a556debd_log.log")
    # # display_H("/home/rogia/.invivo/result/gin_lapool/drugbank_bmnddi_12c974de_log.log")
    # # exit()
    # c = get_expt_results("/home/rogia/.invivo/result/")
    # print(c)
    # exit()
    # c.to_excel("graph_mol_as_extract.xlsx")
