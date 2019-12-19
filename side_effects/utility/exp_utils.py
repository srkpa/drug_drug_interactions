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

INVIVO_RESULTS = ""  # os.environ["INVIVO_RESULTS_ROOT"]


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


def update_model_name(exp_name, pool_arch, graph_net_params, attention_params):
    has = False
    if exp_name == "lstm":
        exp_name = "BiLSTM"
    elif exp_name == "dglgraph":
        if not pool_arch:
            exp_name = "MolGraph"
        else:
            exp_name = "MolGraph + LaPool"
            if attention_params == "attn":
                has = True
    elif exp_name == "conv1d":
        exp_name = "CNN"
        if graph_net_params:
            exp_name = "CNN + IntGraph"
        if int(attention_params) > 0:
            has = True
    elif exp_name == "deepddi":
        exp_name = "DeepDDI"
    else:
        exp_name = ""
    if has:
        exp_name += "+ attention"
    return exp_name


def __compute_metrics(y_pred, y_true):
    macro_roc = roc_auc_score(y_pred=y_pred, y_true=y_true, average="macro")
    macro_auprc = auprc_score(y_pred=y_pred, y_true=y_true, average="macro")
    micro_roc = roc_auc_score(y_pred=y_pred, y_true=y_true, average="micro")
    micro_auprc = auprc_score(y_pred=y_pred, y_true=y_true, average="micro")
    return dict(
        macro_roc=macro_roc, macro_auprc=macro_auprc, micro_roc=micro_roc, micro_auprc=micro_auprc
    )


def __loop_through_exp(path, compute_metric=True):
    outs = []
    for fp in glob.glob(f"{path}/*.pkl"):
        task_id, _ = os.path.splitext(
            os.path.split(fp)[-1])
        params_file = os.path.join(path, task_id[:-3] + "params.json")
        if os.path.exists(params_file):
            config = json.load(open(params_file, "rb"))
            exp = config.get("model_params.network_params.drug_feature_extractor_params.arch", None)
            exp = exp if exp else config.get("model_params.network_params.network_name", None)
            pool_arch = config.get("model_params.network_params.drug_feature_extractor_params.pool_arch.arch", None)
            graph_net = config.get("model_params.network_params.graph_network_params.kernel_sizes", None)
            att_params = config.get("model_params.network_params.drug_feature_extractor_params.use_self_attention",
                                    None)
            att_params = config.get("model_params.network_params.drug_feature_extractor_params.gather",
                                    None) if att_params is None else att_params
            att_params = config.get(
                "model_params.network_params.att_hidden_dim", 0) if att_params is None else att_params
            att_params = 0 if att_params is None else att_params
            exp_name = update_model_name(exp, pool_arch, graph_net, att_params)
            dataset_name = config["dataset_params.dataset_name"]
            mode = config["dataset_params.split_mode"]
            seed = config["dataset_params.seed"]
            res = 0
            if task_id.endswith("_res"):
                print("Config file:", params_file, "result file:", fp, "exp:", exp_name)
                out = pickle.load(open(fp, "rb"))
                if compute_metric:
                    y_pred = load(open(os.path.join(path, task_id[:-3] + "preds.pkl"), "rb"))
                    y_true = load(open(os.path.join(path, task_id[:-3] + "targets.pkl"), "rb"))
                    out = __compute_metrics(y_true=y_true, y_pred=y_pred)
                res = (
                    dict(model_name=exp_name, dataset_name=dataset_name, split_mode=mode, seed=seed, task_id=task_id,
                         path=params_file,
                         **out),)
            if mode == "leave_drugs_out" and os.path.exists(os.path.join(path, task_id[:-4] + "-preds.pkl")):
                print("Config file:", params_file, "result file:", os.path.join(path, task_id[:-4] + "-preds.pkl"),
                      "exp:", exp_name)
                out = pickle.load(open(os.path.join(path, task_id[:-4] + "-res.pkl"), "rb"))
                if compute_metric:
                    y_pred = load(open(os.path.join(path, task_id[:-4] + "-preds.pkl"), "rb"))
                    y_true = load(open(os.path.join(path, task_id[:-4] + "-targets.pkl"), "rb"))
                    out = __compute_metrics(y_true=y_true, y_pred=y_pred)
                res += (
                    dict(model_name=exp_name, dataset_name=dataset_name, split_mode=mode, seed=seed, task_id=task_id,
                         path=params_file,
                         **out),)
            outs.append(res)
    return outs


def get_best_model_params(outs, criteria="micro"):
    max_mm = 0
    inst = {}
    for elem in outs:
        # elem = elem[0]
        mm_val = elem[f"{criteria}_auprc"]
        if mm_val > max_mm:
            max_mm = mm_val
            inst = elem
    return inst


@cli.command()
@click.option('-p', '--mm', type=str, default='micro',
              help="""Position of the config file to run""")
@click.option('--return_best_config/--no-return_best_config', default=True,
              help="Save only task with the best mm for a given exp")
def summarize(return_best_config=False, mm="micro", save=True):
    exp_folder = f"{os.path.expanduser('~')}/expts"
    rand = []
    hidsk = []
    allhd = []
    for i, root in enumerate(os.listdir(exp_folder)):
        path = os.path.join(exp_folder, root)
        if os.path.isdir(path):
            print("__Path__: ", path)
            res = __loop_through_exp(path, compute_metric=True)
            res_2 = [x[0] for x in res if len(x) > 1]
            res_1 = [x for x in res if len(x) <= 1]
            print("Length check:", len(res_1), len(res_2))
            if len(res_2) > 0:
                l1, l2 = list(zip(*res_2))
                hidsk.extend(l1)
                allhd.extend(l2)
            rand.extend(res_1)


    # if return_best_config:
    #             best_hp = get_best_model_params(res_1, criteria=mm)
    #             rand.append(best_hp)
    #         else:
    #
    #
    # Format and save results   # output stats after??
    if rand:
        out = pd.DataFrame(rand)
        out.to_csv("random.xlsx")
    if hidsk and allhd:
        out1 = pd.DataFrame(hidsk)
        out2 = pd.DataFrame(allhd)
        out1.to_csv("leave_drugs_out_1.xlsx")
        out2.to_csv("leave_drugs_out_2.xlsx")


def get_exp_stats(fp):
    # o = []
    # for l in open(fp, "r"):
    #     a = l.split(',"{')[-1]
    #     c = {}
    #     for i in a.split(","):
    #         k = i.split(":")[0].strip("\n").strip().strip("'")
    #         v = i.split(":")[-1].strip("\n").strip('}"').strip().strip("'")
    #         if k.startswith("micro") or k.startswith("macro"):
    #             v = float(v)
    #         print(k, v)
    #         c[k] = v
    #
    #     o.append(c)
    # o = o[1:]
    #
    # df = pd.DataFrame(o)
    # print(df)
    # exit()
    df = pd.read_csv(fp)
    t = df.groupby(['dataset_name', 'model_name'], as_index=True)[
        ["macro_roc", "macro_auprc", "micro_roc", "micro_auprc"]].mean()
    print(t)


if __name__ == '__main__':
    # get_exp_stats("/home/rogia/Bureau/result/leave_drugs_out_1.xlsx")
    # exit()
    #visualize_loss_progress("/home/rogia/Bureau/drugbank_bmnddi_27e75f3c_log.log")
    # exit()
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
