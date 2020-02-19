import glob
import json
import multiprocessing
import operator
import os
import pickle
import shutil
import tempfile
from collections import Counter, defaultdict
from pickle import load
from statistics import mean

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ivbase.utils.aws import aws_cli
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from tqdm import tqdm

from side_effects.data.loader import get_data_partitions
from side_effects.expts_routines import run_experiment
from side_effects.metrics import auprc_score, roc_auc_score
from side_effects.trainer import wrapped_partial

INVIVO_RESULTS = ""  # os.environ["INVIVO_RESULTS_ROOT"]
num_cores = multiprocessing.cpu_count()


def find_nlargest_elements(list1, N):
    final_list = []
    out = []
    for i in range(0, N):
        max1 = 0
        idx = 0
        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]
                idx = j
        list1.remove(max1)
        final_list.append(idx)
        out.append(max1)

    return out, final_list


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


def visualize_test_perf(fp="../../results/temp.csv", model_name="CNN", **kwargs):
    output = defaultdict(dict)
    bt = pd.read_csv(fp)
    c = bt[bt["model_name"] == model_name][['dataset_name', 'task_id', 'split_mode', 'path']].values
    n = len(c) // 2
    print(len(c), n)
    # fig, ax = plt.subplots(nrows=n - 1, ncols=n, figsize=(15, 9))
    # i = 0
    on_test = kwargs.get('eval_on_test')
    for dataset, task_id, split, path in c:
        output[dataset][split] = {}
        res = scorer(dataset_name=dataset, task_path=path[:-12], split=split, f1_score=True, **kwargs)
        if on_test:
            y_true, ap, roc = res
            output[dataset][split].update(dict(dens=y_true, ap=ap, roc=roc))
        else:
            n_samples, x, ap, roc, f1 = res
            output[dataset][split].update(dict(total=n_samples, x=x, roc=roc, ap=ap, f1=f1))
    return output
    # for row in ax:
    #     for col in row:
    #         dataset, task_id, split, path = c[i]
    #         print(split)
    #         scorer(dataset_name=dataset, task_path=path[:-12], split=split, f1_score=True, ax=col)
    #         i += 1
    # plt.show()
    # plt.savefig(f"../../results/scores.png")


# plt.show()


# /media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json -- best result
def scorer(dataset_name, task_path=None, upper_bound=None, f1_score=True, split="random", ax=None, annotate=False,
           req=None, eval_on_train=False, eval_on_test=False):
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    y_preds = load(open(task_path + "_preds.pkl", "rb")) if split != "leave_drugs_out (test_set 2)" else load(
        open(task_path + "-preds.pkl", "rb"))
    y_true = load(open(task_path + "_targets.pkl", "rb")) if split != "leave_drugs_out (test_set 2)" else load(
        open(task_path + "-targets.pkl", "rb"))
    if eval_on_test:
        ind = [i for i in range(len(y_true)) if len(np.unique(y_true[i, :])) == 2]
        auprc = list(map(wrapped_partial(auprc_score, average='micro'), y_preds[ind, :], y_true[ind, :]))
        auroc = list(map(wrapped_partial(roc_auc_score, average='micro'), y_preds[ind, :], y_true[ind, :]))
        dens = np.sum(y_true, axis=1).tolist()
        dens = [dens[i] for i in ind]
        assert len(ind) == len(auroc) == len(auprc) == len(dens)
        return dens, auprc, auroc

    ap_scores = auprc_score(y_pred=y_preds, y_true=y_true, average=None)
    rc_scores = roc_auc_score(y_pred=y_preds, y_true=y_true, average=None)
    if eval_on_train:
        task_id = task_path + "_params.json"
        train, valid, test1, test2 = __collect_data__(task_id, return_config=False)
        g = train.labels_vectorizer.classes_
        train_freq = torch.sum(train.get_targets(), dim=0)
        print(len(train.samples), train.get_targets().shape)
        n_samples = dict(zip(g, list(map(int, train_freq.tolist()))))
        total = len(train.samples)

    else:
        data_set = pd.read_csv(f"{cach_path}/datasets-ressources/DDI/{dataset_name}/{dataset_name}.csv",
                               sep=",")
        ddi_types = [e for l in data_set["ddi type"].values.tolist() for e in l.split(";")]
        n_samples = dict(Counter(ddi_types))
        g = list(set(ddi_types))
        g.sort()
        total = 63472 if dataset_name == "twosides" else 191878

    # Label ranking
    # ap_scores_ranked = sorted(zip(g, ap_scores), key=operator.itemgetter(1))
    # rc_scores_dict = dict(zip(g, list(rc_scores)))
    # btm_5_ap_scores = ap_scores_ranked[:10]
    # btm_5_ap_rc_scores = [(l, v, rc_scores_dict[l]) for l, v in btm_5_ap_scores]
    # top_5_ap_scores = ap_scores_ranked[::-1][:10]
    # top_5_ap_rc_scores = [(l, v, rc_scores_dict[l]) for l, v in top_5_ap_scores]
    #
    # df = pd.DataFrame(top_5_ap_rc_scores, columns=["Best performing DDI types", "AUPRC", "AUROC"])
    # df1 = pd.DataFrame(btm_5_ap_rc_scores, columns=["Worst performing DDI types", "AUPRC", "AUROC"])
    # prefix = os.path.basename(task_path)
    # df.to_csv(f"../../results/{prefix}_{split}_top.csv")
    # df1.to_csv(f"../../results/{prefix}_{split}_bottom.csv")
    # print(df)
    # print(df1)
    #
    if split == "random":
        title = split
    elif split == "leave_drugs_out (test_set 2)":
        title = "hard early"
    else:
        title = "soft early"
    # zoom
    x = [n_samples[g[i]] for i in range(len(ap_scores))]
    if upper_bound:
        x, y = [j for i, j in enumerate(x) if j <= 2000], [i for i, j in enumerate(x) if j <= 2000]
        ap_scores = [ap_scores[i] for i in y]
        rc_scores = [rc_scores[i] for i in y]

    # f1 score instead
    if f1_score:
        f1_scores = list(map(lambda prec, rec: (2 * prec * rec) / (prec + rec), ap_scores, rc_scores))
    # print(f1_scores)

    output = {}
    if req:
        for side_effect in req:
            freq = (100 * n_samples[side_effect.lower()]) / 63472
            pos = g.index(side_effect.lower())
            ap, rc, f1 = ap_scores[pos], rc_scores[pos], f1_scores[pos]
            output[side_effect] = ap, rc, f1, freq
        return output
    return total, x, ap_scores, rc_scores, f1_scores
    #     if ax is None:
    #         plt.figure(figsize=(8, 6))
    #         plt.scatter(x, f1_scores, 40, marker="+", color='xkcd:lightish blue', alpha=0.7, zorder=1, label="F1 score")
    #         plt.axhline(y=0.5, color="gray", linestyle="--", zorder=2, alpha=0.6)
    #     else:
    #         ax.scatter(x, f1_scores, 40, marker="+", color='xkcd:lightish blue', alpha=0.7, zorder=1, label="F1 score")
    #         ax.axhline(y=0.5, color="gray", linestyle="--", zorder=2, alpha=0.6)
    #         ax.set_xlabel('Number of positives samples', fontsize=10)
    #         ax.set_ylabel('F1 Scores', fontsize=10)
    #
    # else:
    #     plt.scatter(x, ap_scores, 40, marker="o", color='xkcd:lightish blue', alpha=0.7, zorder=1, label="AUPRC")
    #     plt.scatter(x, rc_scores, 40, marker="o", color='xkcd:lightish red', alpha=0.7, zorder=1, label="AUROC")
    #     plt.axhline(y=0.5, color="gray", linestyle="--", zorder=2, alpha=0.6)
    #     plt.text(max(x), 0.5, "mean = 0.5", horizontalalignment="right", size=12, family='sherif', color="r")

    # ax.set_title(tile, fontsize=10)

    # # plt.gca().spines['top'].set_visible(False)
    # # plt.gca().spines['right'].set_visible(False)
    # plt.title(f"{split}-{dataset_name}")
    # plt.legend()
    # plt.savefig(f"../../results/{prefix}_{save}_scores.png")


# plt.show()


def set_model_name(exp_name, pool_arch, graph_net_params, attention_params):
    has = False
    if exp_name == "lstm":
        exp_name = "BLSTM"
    elif exp_name == "dglgraph":
        if not pool_arch:
            exp_name = "GIN"
        else:
            exp_name = "GIN + LaPool"
            # if attention_params == "attn":
            #     has = True
    elif exp_name == "conv1d":
        exp_name = "CNN"
        if graph_net_params:
            exp_name = "CNN + GIN"
        if int(attention_params) > 0:
            has = True
    elif exp_name == "deepddi":
        exp_name = "DeepDDI"
    else:
        exp_name = ""
    if has:
        exp_name += ""
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
            exp_name = set_model_name(exp, pool_arch, graph_net, att_params)
            dataset_name = config["dataset_params.dataset_name"]
            mode = config["dataset_params.split_mode"]
            # print(mode)
            seed = config["dataset_params.seed"]
            fmode = config.get("model_params.network_params.mode", None)
            test_fold = config.get("dataset_params.test_fold", None)
            if task_id.endswith("_res") and os.path.exists(os.path.join(path, task_id[:-3] + "preds.pkl")):
                i_mode = ""
                print("Config file:", params_file, "result file:", fp, "exp:", exp_name)
                out = pickle.load(open(fp, "rb"))
                if mode == "leave_drugs_out":
                    i_mode = mode + " (test_set 1)"
                else:
                    i_mode = mode
                if compute_metric:
                    y_pred = load(open(os.path.join(path, task_id[:-3] + "preds.pkl"), "rb"))
                    y_true = load(open(os.path.join(path, task_id[:-3] + "targets.pkl"), "rb"))
                    out = __compute_metrics(y_true=y_true, y_pred=y_pred)

                res = dict(model_name=exp_name, dataset_name=dataset_name, split_mode=i_mode, seed=seed,
                           task_id=task_id,
                           path=params_file, fmode=fmode, test_fold=test_fold,
                           **out)
                print(i_mode)
                outs.append(res)

            if mode == "leave_drugs_out" and os.path.exists(os.path.join(path, task_id[:-4] + "-res.pkl")):
                print("Config file:", params_file, "result file:", os.path.join(path, task_id[:-4] + "-preds.pkl"),
                      "exp:", exp_name)
                i_mode = mode + " (test_set 2)"
                print(i_mode)
                out = pickle.load(open(os.path.join(path, task_id[:-4] + "-res.pkl"), "rb"))
                if compute_metric:
                    y_pred = load(open(os.path.join(path, task_id[:-4] + "-preds.pkl"), "rb"))
                    y_true = load(open(os.path.join(path, task_id[:-4] + "-targets.pkl"), "rb"))

                    out = __compute_metrics(y_true=y_true, y_pred=y_pred)
                res = dict(model_name=exp_name, dataset_name=dataset_name, split_mode=i_mode, seed=seed,
                           task_id=task_id, test_fold=test_fold,
                           path=params_file, fmode=fmode,
                           **out)
                outs.append(res)

            # outs.append(res)
    return outs


def summarize_experiments(main_dir=None, cm=True):
    main_dir = f"{os.path.expanduser('~')}/expts" if main_dir is None else main_dir
    rand = []
    for i, root in enumerate(os.listdir(main_dir)):
        path = os.path.join(main_dir, root)
        if os.path.isdir(path):
            print("__Path__: ", path)
            res = __loop_through_exp(path, compute_metric=cm)
            print(res)
            if len(res) > 0:
                rand.extend(res)
    print(rand)
    if rand:
        out = pd.DataFrame(rand)
        out.to_csv("../../results/all_raw-exp-res.csv")
        modes = out["split_mode"].unique()
        out["split_mode"].fillna(inplace=True, value="null")
        out["fmode"].fillna(inplace=True, value="null")
        print("modes", modes)
        for mod in modes:
            df = out[out["split_mode"] == mod]
            t = df.groupby(['dataset_name', 'model_name', 'split_mode', "fmode"], as_index=True)
            print("t", t)
            t_mean = t[["micro_roc", "micro_auprc"]].mean()
            print("mean", t_mean)
            t_std = t[["micro_roc", "micro_auprc"]].std()
            print("std", t_std)
            t_mean['micro_roc'] = t_mean[["micro_roc"]].round(3).astype(str) + " ± " + t_std[["micro_roc"]].round(
                3).astype(
                str)
            t_mean['micro_auprc'] = t_mean[["micro_auprc"]].round(3).astype(str) + " ± " + t_std[["micro_auprc"]].round(
                3).astype(str)
            t_mean.to_csv(f"../../results/{mod}.csv")


def __collect_data__(config_file, return_config=False):
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    config = json.load(open(config_file, "rb"))
    dataset_params = {param.split(".")[-1]: val for param, val in config.items() if param.startswith("dataset_params")}
    print(config["model_params.network_params.network_name"])
    dataset_params = {key: val for key, val in dataset_params.items() if
                      key in get_data_partitions.__code__.co_varnames}
    dataset_name = dataset_params["dataset_name"]
    input_path = f"{cach_path}/datasets-ressources/DDI/{dataset_name}"
    dataset_params["input_path"] = input_path
    train, valid, test1, test2 = get_data_partitions(**dataset_params)
    if return_config:
        return dataset_params, train, valid, test1, test2

    return train, valid, test1, test2


@click.group()
def cli():
    pass


#
# @cli.command()
# @click.option('--path', '-r',
#               help="Model path.")
def reload(path):
    for fp in tqdm(glob.glob(f"{path}/*.pth")):
        train_params = defaultdict(dict)
        config = json.load(open(fp[:-15] + "_params.json", "rb"))
        for param, val in config.items():
            keys = param.split(".")
            if 1 < len(keys) <= 2:
                mk, sk = keys
                train_params[mk][sk] = val
            elif len(keys) == 3:
                mk, sk, skk = keys
                if sk not in train_params[mk]:
                    train_params[mk][sk] = {}
                train_params[mk][sk][skk] = val
            elif len(keys) == 4:
                mk, sk, skk, skkk = keys
                if sk in train_params[mk]:
                    if skk not in train_params[mk][sk]:
                        train_params[mk][sk][skk] = {}
                train_params[mk][sk][skk][skkk] = val

            elif len(keys) >= 5:
                mk, sk, skk, skkk, sk5 = keys
                if sk in train_params[mk]:
                    if skk in train_params[mk][sk]:
                        if skkk not in train_params[mk][sk][skk]:
                            train_params[mk][sk][skk][skkk] = {}
                train_params[mk][sk][skk][skkk][sk5] = val
            else:
                print(f"None of my selected keys, {param}")
        print(list(train_params.keys()))
        run_experiment(**train_params, restore_path=fp, checkpoint_path=fp, input_path=None,
                       output_path="/home/rogia/Documents/exps_results/nw_bin")

    exit()
    raw_data = pd.read_csv(f"/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/3003377s-twosides.tsv",
                           sep="\t")
    main_backup = raw_data[['stitch_id1', 'stitch_id2', 'event_name', 'confidence']]
    data_dict = {(did1, did2, name.lower()): conf for did1, did2, name, conf in
                 tqdm(main_backup.values.tolist())}

    # train, valid, test1, test2 = __collect_data__(fp)
    # print(test1.feeding_insts)
    # exit()
    # e = []
    # req = pd.read_csv(fp, sep=",", header=None)
    # for index, row in tqdm(req.iterrows()):
    #     k = row[0], row[1], row[2]
    #     r = row[3]
    #     e.append(k)
    #     p = data_dict.get(k, None)
    #     y_pred += [r]
    #     if p is None:
    #         y_true.append(0)
    #     else:
    #         y_true.append(1)
    # print(Counter(y_true))
    # for k, j in dict(Counter(e)).items():
    #     if j > 1:
    #         print(k, j)
    # print(skm.average_precision_score(np.array(y_true), np.array(y_pred), average='micro'))
    # print(Counter(y_true))

    # exit()
    # train, valid, test1, test2 = __collect_data__(path)
    # y_pred = load(open(path[:-12] + "_preds.pkl", "rb"))
    # y_true = test1.get_targets()
    # samples = test1.samples
    # labels = test1.labels_vectorizer.classes_
    # output = [(*samples[r][:2], labels[c], y_pred[r, c]) for (r, c) in (y_true == 1).nonzero()]
    # print("done")
    # conf = [data_dict[(drug1, drug2, lab.lower())] for drug1, drug2, lab, _ in tqdm(output)]
    # probs = [probability for did1, did2, lab, probability in tqdm(output)]
    #
    # b = defaultdict(list)
    # for i, j in enumerate(conf):
    #     b[j].append(probs[i])
    # for k, v in b.items():
    #     print(k, sum(v) / len(v))


def analyse_kfold_predictions(mod="random", config="CNN", label=1, sup=0.1, inf=0., selected_pairs=None):
    output = []
    res = pd.read_csv("../../results/all_raw-exp-res.csv", index_col=0)
    res = res[res["split_mode"] == mod]
    t = res.groupby(["test_fold"], as_index=True)
    pred_files = {name: group["path"].values.tolist() for name, group in t}
    task_config_files = dict(res[res["model_name"] == config][["test_fold", "path"]].values.tolist())
    dp = pd.read_csv("/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/twosides-drugs-all.csv")
    drugs_dict = dict(zip(dp["drug_id"], dp["drug_name"]))
    preds = {}
    targets = defaultdict(tuple)
    # collect the ground truth per test fold
    for fold, path in tqdm(task_config_files.items()):
        train, valid, test1, test2 = __collect_data__(path)
        y_true, raw_samples, raw_labels = test1.get_targets(), test1.samples, test1.labels_vectorizer.classes_
        targets[fold] = y_true, raw_samples, raw_labels

    # get the mean performances
    for fold, pfs in tqdm(pred_files.items()):
        y_preds = list(map(lambda task_id: load(open(task_id[:-12] + "_preds.pkl", "rb")), pfs))
        y_pred = np.mean(np.stack(y_preds, axis=0), axis=0)
        preds[fold] = y_pred
        y_true = targets[fold][0]
        raw_samples = targets[fold][1]
        raw_labels = list(targets[fold][2])
        if selected_pairs:
            annoted_samples = [operator.itemgetter(*r[:2])(drugs_dict) for r in raw_samples]
            samples_indices = list(
                map(lambda x: annoted_samples.index(x[:2]) if x[:2] in annoted_samples else None, selected_pairs))
            label_indices = list(
                map(lambda x: raw_labels.index(x[-1]), selected_pairs))
            out = list(zip(samples_indices, label_indices))
            for i, (sample_id, label_id) in enumerate(out):
                if sample_id is not None and label_id is not None:
                    row = (mod, fold, *selected_pairs[i], y_pred[sample_id, label_id])
                    # output += [row]
                    print(row)
        else:
            chosen_pairs = [(*operator.itemgetter(*raw_samples[r][:2])(drugs_dict), raw_labels[c], y_pred[r, c]) for
                            (r, c)
                            in (y_true == label).nonzero() if
                            inf < y_pred[r, c] <= sup]
            output += [chosen_pairs]
    if output:  # chosen_pairs
        res = pd.DataFrame(output, columns=["drug_1", "drug_2", "side effect", "probability"])
        res.to_csv(f"../../results/{mod}_kfold_{label}_sis.csv")

    assert len(preds) == len(targets), "Not the same length!"


def analyze_models_predictions(fp="../../results/temp.csv", split_mode="random", dataset_name="twosides", false_pos=0.1,
                               false_neg=0.7, model='CNN'):
    bt = pd.read_csv(fp)
    items = bt[(bt["dataset_name"] == dataset_name) & (bt["split_mode"] == split_mode) & (bt["model_name"] == model)][
        ['dataset_name', "model_name", 'task_id', 'split_mode', 'path']].values
    print(len(items))
    i = 1
    for dataset, mn, task_id, split, path in items:  # [items[::-1][1]]
        print("no ", i, "task_path ", task_id, mn, path)
        __analyse_model_predictions__(task_id=path[:-12], threshold=false_pos, label=1)
        print("False Positives: done !!!")
        __analyse_model_predictions__(task_id=path[:-12], threshold=false_neg, label=0)
        print("False negatives: done !!!")
        i += 1
    assert i == len(items)


def __analyse_model_predictions__(task_id, threshold=0.1, label=1):
    print(label)
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    pref = os.path.basename(task_id)
    y_pred = load(open(task_id + "_preds.pkl", "rb"))
    y_true = load(open(task_id + "_targets.pkl", "rb"))
    config = json.load(open(task_id + "_params.json", "rb"))
    dataset_params = {param.split(".")[-1]: val for param, val in config.items() if param.startswith("dataset_params")}
    print(config["model_params.network_params.network_name"])
    dataset_params = {key: val for key, val in dataset_params.items() if
                      key in get_data_partitions.__code__.co_varnames}
    dataset_name = dataset_params["dataset_name"]
    input_path = f"{cach_path}/datasets-ressources/DDI/{dataset_name}"
    dp = pd.read_csv(f"{input_path}/{dataset_name}-drugs-all.csv")
    dp = dp.applymap(str.lower)
    drugs_dict = dict(zip(dp["drug_id"], dp["drug_name"]))
    dataset_params["input_path"] = input_path
    _, _, test1, _ = get_data_partitions(**dataset_params)
    labels = test1.labels_vectorizer.classes_
    samples = test1.samples
    pos = [[(i, g) for g, h in enumerate(y_true[i, :]) if h == label] for i in range(len(y_true))]
    ddi_types = [[(samples[i][0].lower(), samples[i][1].lower(), labels[y]) for _, y in l] for i, l in
                 enumerate(pos)]

    prob = [[y_pred[xy] for xy in l] for _, l in enumerate(pos)]
    assert len(pos) == len(prob)
    if label == 1:
        print("Case: Positive but predicted as Negative")
        pairs = [[ddi_types[k][i] + (j,) for i, j in enumerate(m) if j < threshold] for k, m in enumerate(prob)]
    else:
        print("Case: Negative but predicted as positive")
        pairs = [[ddi_types[k][i] + (j,) for i, j in enumerate(m) if j >= threshold] for k, m in enumerate(prob)]
    pairs = [l for i in pairs for l in i]

    # check it , i don't want to lie
    raw_data = pd.read_csv(f"{input_path}/3003377s-twosides.tsv", sep="\t")
    main_backup = raw_data[['stitch_id1', 'stitch_id2', 'event_name']]
    main_backup = main_backup.applymap(str.lower)
    main_backup["confidence"] = raw_data['confidence']
    main_backup = [tuple(r) for r in main_backup.values]
    pos_bk = [x[:3] for x in main_backup]
    neg_bk = [y[:2] for y in main_backup]

    if label == 1:
        #  check if the pair exist really
        print(f"check if all pairs exist (Positive pairs) => {len(pairs)}!")
        pos_bk = set(pos_bk)
        print(len(pos_bk), list(pos_bk)[0])
        pos_pairs = set([p[:3] for p in pairs])
        print(len(pos_pairs), list(pos_pairs)[0])
        common = pos_pairs.intersection(pos_bk)
        print(len(common), len(pos_pairs))
        print(len(pos_pairs - common))
        is_ok = len(pos_pairs - common) == 0
    else:
        print(f"check if all pairs exist (Negative pairs)=> {len(pairs)}!")
        neg_bk = set(neg_bk)  # all the drugs pai"/rs that really exists
        neg_pairs = set([p[:3] for p in pairs])  # my negative pairs
        neg_drugs_comb = set([p[:2] for p in pairs])
        common = set(pos_bk).intersection(neg_pairs)
        print(len(neg_pairs), len(common))
        is_ok = len(common) == 0 and len(neg_drugs_comb - neg_bk) == 0
    assert is_ok == True, "Everything is not ok!"
    print("Done!")

    # save it
    pairs = [(drugs_dict[p[0]],) + (drugs_dict[p[1]],) + p[2:] for p in pairs]
    res = pd.DataFrame(pairs, columns=["drug_1", "drug_2", "side effect", "probability"])
    print(res)
    res.to_csv(f"../../results/{pref}_{label}_sis.csv")


def __check_if_overlaps(data, subset):
    i, j, k = 0, 0, 0
    for (drug1, drug2) in subset:
        if drug1 in data and drug2 in data:
            i += 1
        elif drug1 in data or drug2 in data:
            j += 1
        else:
            k += 1
    return i, j, k


def __get_dataset_stats__(task_id, level='pair'):
    test_stats = []
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    config = json.load(open(task_id, "rb"))
    print(config)
    dataset_params = {param.split(".")[-1]: val for param, val in config.items() if param.startswith("dataset_params")}
    dataset_params = {key: val for key, val in dataset_params.items() if
                      key in get_data_partitions.__code__.co_varnames}
    dataset_name = dataset_params["dataset_name"]
    split_mode = dataset_params["split_mode"]
    detailed_split_scheme = split_mode if split_mode == "random" else "soft early"
    columns = ["dataset_name", "seed", "splitting scheme", "drugs pairs in test set",
               "Both of the drugs are seen",
               "One of the drug is seen",
               "None of the drugs were seen "] if level == "pair" else ["dataset_name", "seed", "splitting scheme",
                                                                        "no.drugs (train)",
                                                                        "no. drugs (valid)", "no. drugs (test drugs)",
                                                                        "train ∩ valid", "train ∩ test", "valid ∩ test"]
    seed = dataset_params["seed"]
    input_path = f"{cach_path}/datasets-ressources/DDI/{dataset_name}"
    dataset_params["input_path"] = input_path
    train, valid, test1, test2 = get_data_partitions(**dataset_params)  # test2

    train_samples, test1_samples, test2_samples = train.samples, test1.samples, test2.samples
    test = [(d1, d2) for (d1, d2, _) in test1_samples]
    train_drugs = set([d1 for (d1, _, _) in train_samples] + [d2 for (_, d2, _) in train_samples])
    test1_drugs = set([d1 for (d1, _, _) in test1_samples] + [d2 for (_, d2, _) in test1_samples])
    valid_drugs = set([d1 for (d1, _, _) in valid.samples] + [d2 for (_, d2, _) in valid.samples])

    if level == "drugs":
        b = train_drugs.intersection(valid_drugs)
        l = train_drugs.intersection(test1_drugs)
        g = valid_drugs.intersection(test1_drugs)
        prec = dataset_name, seed, detailed_split_scheme, len(train_drugs), len(valid_drugs), len(test1_drugs), len(
            b), len(
            l), len(g)
    else:
        i, j, k = __check_if_overlaps(train_drugs, test)
        prec = dataset_name, seed, detailed_split_scheme, len(set(test)), i, j, k
    test_stats += [prec]

    print("Number of drugs pairs - test 1 ", len(test))
    print("Number of train drugs", len(set(train_drugs)))
    print("Number of test set 1 drugs ", len(set(test1_drugs)))

    if split_mode == "leave_drugs_out":
        test = [(d1, d2) for (d1, d2, _) in test2_samples]
        test2_drugs = set([d1 for (d1, _, _) in test2_samples] + [d2 for (_, d2, _) in test2_samples])
        print("Number of drugs pairs - test 2 ", len(test))
        print("Number of test set 2 drugs", len(set(test2_drugs)))
        assert len(set(train_drugs).intersection(test2_drugs)) == 0
        if level == "drugs":
            b = train_drugs.intersection(valid_drugs)
            l = train_drugs.intersection(test2_drugs)
            g = valid_drugs.intersection(test2_drugs)
            prec = dataset_name, seed, "hard early", len(train_drugs), len(valid_drugs), len(
                test2_drugs), len(b), len(
                l), len(g)
        else:
            i, j, k = __check_if_overlaps(train_drugs, test)
            prec = dataset_name, seed, "hard early", len(set(test)), i, j, k
        test_stats += [prec]

    df = pd.DataFrame(test_stats, columns=columns)
    return df


def get_dataset_stats(task_ids=None, exp_path=None, split=True, level='pair'):
    if not task_ids:
        assert exp_path is not None
        task_ids = [fp for fp in glob.glob(f"{exp_path}/*.json")]
    print(task_ids)
    df = pd.concat(list(map(wrapped_partial(__get_dataset_stats__, level=level), task_ids)), ignore_index=True)
    df.to_csv(f"../../results/{level}_dataset_params_stats.csv")
    print(df)
    mean_sp = df.groupby(["dataset_name", "splitting scheme"]).mean().round()
    mean_std = df.groupby(["dataset_name", "splitting scheme"]).std().round(3)
    overall_summary = mean_sp.astype(str) + " ± " + mean_std.astype(str)
    overall_summary.to_csv(f"../../results/{level}_all-modes_dataset_params_stats.csv")
    if split:
        modes = df["splitting scheme"].unique()
        df["splitting scheme"].fillna(inplace=True, value="null")
        for mod in modes:
            res = df[df["splitting scheme"] == mod]
            res.to_csv(f"../../results/{level}_{mod}_dataset_params_stats.csv")


def get_best_hp():
    res = pd.read_csv("../../results/all_raw-exp-res.csv", index_col=0)
    config_hp = res.groupby(["dataset_name", "model_name", "split_mode"], as_index=False)["micro_auprc"].max()
    print(list(config_hp))
    best_exp_hps = []
    for index, row in config_hp.iterrows():
        best_exp_hps.append(
            res.loc[(res["dataset_name"] == row["dataset_name"]) & (res["model_name"] == row["model_name"]) & (
                    res["split_mode"] == row["split_mode"]) & (res["micro_auprc"] == row["micro_auprc"])])
    out = pd.concat(best_exp_hps)
    print(out)
    out.to_csv("../../results/temp.csv")


def __collect_link_pred_res__(lst=None):
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    lst = load(open("/home/rogia/Documents/exps_results/kfrgcn/twosides_RGCN_c1ce1cae_res.pkl", "rb"))
    config = json.load(open("/home/rogia/Documents/exps_results/kfrgcn/twosides_RGCN_c1ce1cae_params.json", "rb"))
    dataset_params = {param.split(".")[-1]: val for param, val in config.items() if param.startswith("dataset_params")}
    print(config["model_params.network_params.network_name"])
    dataset_params = {key: val for key, val in dataset_params.items() if
                      key in get_data_partitions.__code__.co_varnames}
    dataset_name = dataset_params["dataset_name"]
    input_path = f"{cach_path}/datasets-ressources/DDI/{dataset_name}"
    dataset_params["input_path"] = input_path
    train, valid, test1, test2 = get_data_partitions(**dataset_params)
    labels = test1.labels_vectorizer.classes_.tolist()
    samples = [(did1, did2) for did1, did2, _ in test1.samples]
    y_pred = np.zeros((len(test1.samples), len(labels)))
    y_true = test1.get_targets()

    for k, v, pb, _ in tqdm(lst):
        row, col = samples.index(k), labels.index(v)
        y_pred[row, col] = pb
    out = __compute_metrics(y_true=y_true, y_pred=y_pred)
    print(out)


def find_pairs(fp, mod, model="CNN", label=0, req_pairs=None, inf=None, sup=None, save_filename=None):
    output = []
    dp = pd.read_csv("/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/twosides-drugs-all.csv")
    drugs_dict = dict(zip(dp["drug_id"], dp["drug_name"]))
    res = pd.read_csv(fp, index_col=0)
    res = res[(res["split_mode"] == mod) & (res['model_name'] == model) & (res['dataset_name'] == "twosides")]
    task_config_files = res[["path"]].values.tolist()
    print(task_config_files)
    import torch
    task_config_files = [["/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json"]]
    for path in tqdm(task_config_files):
        cfg, train, valid, test1, test2 = __collect_data__(path[-1], return_config=True)
        y_pred = load(open(path[-1][:-12] + "_preds.pkl", "rb")) if mod != "leave_drugs_out (test_set 2)" else load(
            open(path[-1][:-12] + "-preds.pkl", "rb"))
        test = test1 if mod != "leave_drugs_out (test_set 2)" else test2
        y_true, raw_samples, raw_labels = test.get_targets(), test.samples, list(test.labels_vectorizer.classes_)
        annoted_samples = [operator.itemgetter(*r[:2])(drugs_dict) for r in raw_samples]
        annoted_samples = [(*p, raw_samples[r][-1]) for r, p in enumerate(annoted_samples)]
        samples_indices = list(
            map(lambda x: annoted_samples.index(x[:2]) if x[:2] in annoted_samples else None, req_pairs))
        label_indices = list(
            map(lambda x: raw_labels.index(x[-1]), req_pairs))
        out = list(zip(samples_indices, label_indices))
        if sup is not None:
            res = [(r, c) for (r, c) in (torch.tensor(y_true) == label).nonzero() if
                   inf < y_pred[r, c] <= sup]  # (y_[:, id] >= sup).nonzero()
            res = [(r, c) for (r, c) in res for id in label_indices if c == id]
            n_1 = len(res)
            if label == 1:
                res = [(mod, cfg['seed'], *annoted_samples[r][:2], raw_labels[c], y_pred[r, c]) for r, c in res if
                       raw_labels[c] in annoted_samples[r][-1]]
            else:
                res = [(mod, cfg['seed'], *annoted_samples[r][:2], raw_labels[c], y_pred[r, c]) for r, c in res if
                       raw_labels[c] not in annoted_samples[r][-1]]
            print(len(res), n_1)
            assert len(res) == n_1
            output = res
        else:
            for i, (sample_id, label_id) in enumerate(out):
                if sample_id is not None and label_id is not None:
                    row = (mod, cfg['seed'], *req_pairs[i], y_pred[sample_id, label_id])
                    output += [row]
                    print(row)
    if output:
        rs = pd.DataFrame(output,
                          columns=["Evaluation schemes", "seed", "drug_1", "drug_2", "side effect", "probability"])
        rs.to_csv(f"../../results/{save_filename}.csv")


def get_similarity(pairs, task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json", ref=None,
                   label=1):
    did1, did2, se = pairs
    train, valid, test1, test2 = __collect_data__(task_id, return_config=False)
    # pd.read_csv("/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides//3003377s-twosides.tsv",
    #                                 sep="\t")
    # twosides_database = twosides_database[["stitch_id1", "stitch_id2", 'event_name']].applymap(str.lower)
    dp = pd.read_csv("/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/drugs.csv", sep=";")
    dp["drug_id"] = dp[["drug_id"]].applymap(str.lower)
    drugs_dict = pickle.load(open("/home/rogia/Documents/projects/drug_drug_interactions/expts/save.p", "rb"))
    drugs_dict = {x.lower(): v for x, v in drugs_dict.items()}
    ds_dict = dict(zip(dp["drug_name"], dp["drug_id"]))
    if ref:
        selected_pairs = pd.read_csv(ref,
                                     sep=",")[["drug_1", "drug_2", "side effect"]]

        selected_pairs = [operator.itemgetter(*r[:2])(ds_dict) for r in selected_pairs.values if r[-1] == se]

    else:
        if label == 1:
            selected_pairs = [(drug1id.lower(), drug2id.lower()) for (drug1id, drug2id, labels) in train.samples if
                              se in labels]
        else:
            selected_pairs = [(drug1id.lower(), drug2id.lower()) for (drug1id, drug2id, labels) in train.samples if
                              se not in labels]
        print(se, len(selected_pairs))
    ref_smiles = [operator.itemgetter(*r)(drugs_dict) for r in selected_pairs]
    did1, did2 = ds_dict[did1], ds_dict[did2]
    smi1, smi2 = drugs_dict[did1], drugs_dict[did2]

    def compute_sim(ref, smi1a, smi1b):
        smi2a, smi2b = ref
        fps = [FingerprintMols.FingerprintMol(x) for x in [smi1a, smi1b, smi2a, smi2b]]
        k = [max(DataStructs.FingerprintSimilarity(fps[0], fps[2]), DataStructs.FingerprintSimilarity(fps[0], fps[3])),
             max(DataStructs.FingerprintSimilarity(fps[1], fps[2]), DataStructs.FingerprintSimilarity(fps[1], fps[3]))]
        return mean(k) - 0.001

    # ref_smiles = ref_smiles[:2]
    output = list(map(wrapped_partial(compute_sim, smi1a=smi1, smi1b=smi2), tqdm(ref_smiles)))
    assert len(output) == len(ref_smiles)
    output = sorted(output, reverse=True)[:100]
    print(len(output), output)
    return output


if __name__ == '__main__':
    # summarize_experiments("/home/rogia/Documents/exps_results/kfolds", cm=False)
    # # ('cid000003345', 'cid000005076', 'malignant melanoma')
    # input_path = f"/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/3003377s-twosides.tsv"
    # data = pd.read_csv(input_path, sep="\t")
    # print(list(data))
    # l = data[(data['drug1'].str.lower() == 'fenofibrate') & (data['drug2'].str.lower() == 'valsartan') & (data['event_name'].str.lower() == 'hepatitis c')]
    # print(l)
    #
    #
    # exit()
    # request_pairs = [("fenofibrate", "valsartan", "hepatitis c"), ("verapamil", "fluvoxamine", 'hepatitis c'),
    #                  ("bupropion", "fluoxetine", 'hepatitis a'), ("bupropion", "fluoxetine", "hiv disease"),
    #                  ("paroxetine", "risperidone", "hiv disease"), ("mupirocin", "sertraline", "drug withdrawal"),
    #                  ("amlodipine", "cerivastatin", "flu"), ("metoclopramide", "minoxidil", "road traffic accident"),
    #                  ("n-acetylcysteine", "cefuroxime", "herpes simplex"),
    #                  ("clonazepam", "salmeterol", "adverse drug effect")]
    # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=1,
    #        req_pairs=request_pairs, inf=0.6, sup=1.,  save_filename="tab9") # goog predictions
    # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='leave_drugs_out (test_set 1)', label=1,
    #            req_pairs=request_pairs)
    # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='leave_drugs_out (test_set 2)', label=1,
    #            req_pairs=request_pairs)

    request_pairs = [("bupropion", "benazepril", "heart attack"), ("didanosine", "stavudine", "acidosis"),
                     ("bupropion", "fluoxetine", "panic attack"), ("bupropion", "orphenadrine", "muscle spasm"),
                     ("tramadol", "zolpidem", "diaphragmatic hernia"), ("paroxetine", "fluticasone", "muscle spasm"),
                     ("paroxetine", "fluticasone", "fatigue"), ("fluoxetine", "methadone", "pain"),
                     ("carboplatin", "cisplatin", "blood sodium decreased"),
                     ("chlorthalidone", "fluticasone", "high blood pressure")]

    # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=0,
    #            req_pairs=request_pairs, inf=0.0, sup=0.1, save_filename="tab-10")
    # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=0,
    #            req_pairs=request_pairs, inf=0.7, sup=1., save_filename="bad_pred")
    find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=1,
               req_pairs=request_pairs, inf=0.7, sup=1, save_filename="1_as_1")
    find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=1,
               req_pairs=request_pairs, inf=0., sup=0.1, save_filename="1_as_0")
    find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=0,
               req_pairs=request_pairs, inf=0., sup=0.1, save_filename="0_as_0")
    find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=0,
               req_pairs=request_pairs, inf=0.7, sup=1., save_filename="0_as_1")
    exit()
    # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='leave_drugs_out (test_set 2)', label=0,
    #            req_pairs=request_pairs)

    # # analyze_models_predictions(fp="/home/rogia/Documents/projects/temp-2/temp.csv", split_mode="leave_drugs_out (test_set 1)", false_pos=1., false_neg=0.)
    # # analyze_models_predictions(fp="/home/rogia/Documents/projects/temp-2/temp.csv",
    # #                            split_mode="leave_drugs_out (test_set 2)",false_pos=1., false_neg=0)
    # # # # cli()
    # # #
    # # #  reload(path="/home/rogia/Documents/exps_results/binary/cl_gin")
    # #exit()
    # # # summarize_experiments("/media/rogia/5123-CDC3/SOC", cm=True)
    # # # get_best_hp()
    # # # exit()
    # #
    # # # brou("/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json")
    # # # exit()
    # # # raw_data = pd.read_csv(f"/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/3003377s-twosides.tsv", sep="\t")
    # # # print(list(raw_data))
    # # # print(raw_data[(raw_data["drug1"].str.lower() == "aprepitant")  & (raw_data["drug2"].str.lower() == "cisplatin")  & (raw_data["event_name"].str.lower() =="hepatitis c" )][["drug2", "event_name"]])  #
    # # #
    # # # exit()
    # # # summarize_experiments(main_dir="/home/rogia/Documents/exps_results")
    # analyse_kfold_predictions(mod="random", config="CNN", label=1, sup=0.1, inf=0.)
    # analyse_kfold_predictions(mod="random", config="CNN", label=0, sup=1., inf=0., selected_pairs=request_pairs)
    # analyse_kfold_predictions(mod="leave_drugs_out (test_set 1)", config="CNN", label=0, sup=1., inf=0., selected_pairs=request_pairs)
    # # # analyse_kfold_predictions(mod="leave_drugs_out (test_set 1)", config="CNN", label=1, sup=0.1, inf=0.)
    # # # analyse_kfold_predictions(mod="leave_drugs_out (test_set 1)", config="CNN", label=0, sup=1., inf=0.7)
    # #
    # res = pd.read_csv("/home/rogia/Documents/projects/temp-SOC/all_raw-exp-res.csv", index_col=0)
    # # y1 = res[res["split_mode"] == "random"]
    # # y1 = y1.groupby(["test_fold", "model_name"], as_index=True)[["micro_roc", "micro_auprc"]].max()
    # # y1.to_csv("../../results/random_exp_grouped_by_test_fold.csv")
    # # y2 = res[res["split_mode"] == "leave_drugs_out (test_set 1)"]
    # # y2 = y2.groupby(["test_fold", "model_name"], as_index=True)[
    # #     ["micro_roc", "micro_auprc"]].max()
    # # y2.to_csv("../../results/early_exp_grouped_by_test_fold.csv")
    # # exit()
    # # visualize_loss_progress("../../../twosides_bmnddi_6b85a591_log.log")
    # # exit()
    # # #
    # # # exit()
    # # #
    # # #
    # # # exit()
    # # visualize_loss_progress("/home/rogia/Documents/projects/twosides_deepddi_7a8da2c0_log.log")
    # #
    # # exit()
    # #
    # # exit()
    # # get_dataset_stats(exp_path="/media/rogia/CLé USB/expts/DeepDDI", level='pair')
    # # get_dataset_stats(exp_path="/media/rogia/CLé USB/expts/DeepDDI", level='drugs')
    # # exit()
    # # df = __get_dataset_stats__("/home/rogia/.invivo/result/cl_deepddi/twosides_deepddi_4ebbb144_params.json")
    # # print(df)
    # # exit()
    # # # summarize_experiments("/media/rogia/CLé USB/expts/")
    # # visualize_test_perf()
    # #
    # # exit()
    # # # bt = pd.read_csv("../../results/best_hp.csv")
    # # # # c = bt[bt["model_name"] == 'CNN'][['dataset_name', 'task_id', 'split_mode', 'path']].values
    # # # # print(c)
    # # # c = bt[(bt["dataset_name"] == 'twosides') & (bt["split_mode"] == "random")][
    # # #     ['dataset_name', "model_name", 'task_id', 'split_mode', 'path']].values
    # # # analyze_models_predictions([c[::-1][1]])
    # #
    # # # analysis_test_perf(c)
    # # exit()
    # # get_dataset_stats(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json")
    # # exit()
    # # # get_misclassified_labels(
    # # #     "/home/rogia/Documents/projects/drug_drug_interactions/results/twosides_bmnddi_6936e1f9_1_sis.csv", items=10)
    # # # exit()
    # # # combine(exp_path="/media/rogia/CLé USB/expts/DeepDDI", split=False)
    # # exit()
    # # analyse_model_predictions(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", label=1, threshold=0.1)
    # # analyse_model_predictions(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", label=0, threshold=0.7)
    # # choose o.1 as threshold for false positive
    # # combine(["/home/rogia/Documents/projects/twosides_bmnddi_390811c9",
    # #          "/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9"])
    #
    # exit()
    # exp_folder = "/home/rogia/Documents/exps_results/IMB"  # /media/rogia/CLé USB/expts"  # f"{os.path.expanduser('~')}/expts"
    # # summarize(exp_folder)
    # # Scorer twosides random split
    # scorer("twosides", task_path="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", save="random")
    # # Scorer twosides early split stage
    # scorer("twosides", task_path="/home/rogia/Documents/projects/twosides_bmnddi_390811c9", save="early+stage")
    # exit()
    # scorer("twosides", task_path="/home/rogia/Documents/exps_results/imb/twosides_bmnddi_48d052b6", save="cnn+NS")
    # scorer("twosides", task_path="/home/rogia/Documents/exps_results/imb/twosides_bmnddi_5cbedb4a", save="cnn+CS")
    #
    # # Scorer drugbank random split
    # scorer("drugbank", task_path="/media/rogia/CLé USB/expts/CNN/drugbank_bmnddi_54c9f5bd", save="drugbank")
    # # Scorer drugbank early split
    # exit()
    #
    # f = __loop_through_exp(exp_folder + "/CNN")
    # # # # # d = __loop_through_exp(exp_folder + "/BiLSTM")
    # # # # # a = __loop_through_exp(exp_folder + "/DeepDDI")
    # # # # # b = __loop_through_exp(exp_folder + "/Lapool")
    # # # # # c = __loop_through_exp(exp_folder + "/MolGraph")
    # # # # #
    # # # # # e = a + b + c + d
    # out = pd.DataFrame(f)
    # print(out)
    # out.to_csv("temp.csv")
    # t = out.groupby(['dataset_name', 'model_name', "split_mode"], as_index=True)[
    #     ["micro_roc", "micro_auprc"]].mean()
    # print(t)
    # # out.to_csv("sum-exp.xlsx")
    # # get_exp_stats("sum-exp.xlsx")
    # # exit()
    # visualize_loss_progress("/home/rogia/Documents/lee/drugbank_adnn_bc985d58_log.log")
    # exit()
    # summarize()
    # # display("/home/rogia/Documents/cl_lstm/_hp_0/twosides_bmnddi_8aa095b0_log.log")
    # # display(dataset_name="drugbank", exp_folder="/home/rogia/Documents/no_e_graph/_hp_0")
    # # display(dataset_name="twosides", exp_folder="/home/rogia/Documents/cl_lstm/_hp_0/")
    # # display_H("/home/rogia/Documents/graph_mol/drugbank_bget_expt_results("/home/rogia/Documents/graph_mol/")mnddi_a556debd_log.log")
    # # display_H("/home/rogia/.invivo/result/cl_lapool/drugbank_bmnddi_12c974de_log.log")
    # # exit()
    # c = get_expt_results("/home/rogia/.invivo/result/")
    # print(c)
    # exit()
    # c.to_excel("graph_mol_as_extract.xlsx")
