import datetime
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ivbase.utils.aws import aws_cli
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from tqdm import tqdm

from side_effects.data.loader import get_data_partitions
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


def visualize_loss_progress(filepath, n_epochs=20, title="", save_as=""):
    df = pd.read_table(filepath, sep="\t")
    seconds = df['time'][0]
    total_time = str(datetime.timedelta(seconds=seconds * n_epochs))
    print(f"Time per epoch: {str(datetime.timedelta(seconds=seconds))}")
    print(f"Total time needeed over {n_epochs}: ", total_time)
    plt.figure()
    df.plot(x="epoch", y=["loss", "val_loss"])
    # plt.savefig(f"/home/rogia/Bureau/figs/{save_as}")
    plt.show()
    # plt.title(f"dropout rate = {str(title)}")
    # # plt.show()
    # plt.savefig(f"dropout rate = {str(title)}" + ".png")


def load_learning_curves(log_files, ylim=0.4):
    result = []
    fig, ax = plt.subplots(1, 2, figsize=(8, 5), sharey='row')

    t = np.arange(100)

    for filepath in log_files:
        df = pd.read_table(filepath, sep="\t")
        epoch = df["epoch"]
        loss, val_loss = df["loss"], df["val_loss"]
        result += [(epoch, loss, val_loss)]
    plt.ylim(0.15, 0.7)

    i = 0
    for row in ax:
        row.plot(result[i][0], result[i][1], result[i][2])
        row.set(xlabel='epoch')
        i += 1

    ax[0].set_title("Feat. Extractor Freezed")
    ax[0].set_ylabel("loss")
    ax[1].set_title('Feat Extractor not freezed')
    # plt.show()
    plt.savefig("transfer_learning.png")


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
        print(task_id)
        res = scorer(dataset_name=dataset, task_path=path[:-12], split=split, f1_score=True, **kwargs)
        if on_test:
            y_true, ap, roc = res
            output[dataset][split].update(dict(dens=y_true, ap=ap, roc=roc))
        else:
            n_samples, x, ap, roc, f1 = res
            output[dataset][split].update(dict(total=n_samples, x=x, roc=roc, ap=ap, f1=f1))
    return output


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

    if np.isnan(y_preds).any():
        filter = np.isnan(y_preds).any(axis=1)
        y_preds = y_preds[~filter]
        y_true = y_true[~filter]

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
        print("Je suis ici total", total)

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


def set_model_name(exp_name, pool_arch="", graph_net_params="", attention_params=0):
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


def __loop_through_exp(path, compute_metric=True, param_names=None):
    outs = []
    for fp in glob.glob(f"{path}/*.pkl"):
        task_id, _ = os.path.splitext(
            os.path.split(fp)[-1])
        params_file = os.path.join(path, task_id[:-3] + "params.json")
        if os.path.exists(params_file):
            config = json.load(open(params_file, "rb"))
            exp = config.get("model_params.network_params.drug_feature_extractor_params.arch", None)
            exp = exp if exp else config.get("model_params.network_params.network_name", None)
            dataset_name = config["dataset_params.dataset_name"]
            mode = config["dataset_params.split_mode"]
            if isinstance(mode, list):
                mode, _ = mode

            if isinstance(dataset_name, list):
                dataset_name, _ = dataset_name
            # print(mode)
            seed = config["dataset_params.seed"]
            fmode = config.get("model_params.network_params.mode", None)
            config_params = dict(model_name=exp, dataset_name=dataset_name, seed=seed, task_id=task_id,
                                 path=params_file, fmode=fmode)

            if param_names is None:
                neg_rate = config.get("model_params.loss_params.neg_rate", None)
                use_weight_rescaling = config.get("model_params.loss_params.rescale_freq", None)
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
                test_fold = config.get("dataset_params.test_fold", None)
                config_params.update(
                    dict(
                        neg_rate=neg_rate, rescale=use_weight_rescaling, test_fold=test_fold, model_name=exp_name
                    )
                )
            else:
                config_params.update({met: config.get(met, None) for met in param_names})

            if task_id.endswith("_res") and os.path.exists(os.path.join(path, task_id[:-3] + "preds.pkl")):
                i_mode = ""
                print("Config file:", params_file, "result file:", fp, "exp:", config_params.get("exp_name"))
                out = pickle.load(open(fp, "rb"))
                if mode == "leave_drugs_out":
                    i_mode = mode + " (test_set 1)"
                else:
                    i_mode = mode
                if compute_metric:
                    y_pred = load(open(os.path.join(path, task_id[:-3] + "preds.pkl"), "rb"))
                    y_true = load(open(os.path.join(path, task_id[:-3] + "targets.pkl"), "rb"))
                    out = __compute_metrics(y_true=y_true, y_pred=y_pred)

                res = dict(split_mode=i_mode,
                           **out, **config_params)
                print(i_mode)
                outs.append(res)

            if mode == "leave_drugs_out" and os.path.exists(os.path.join(path, task_id[:-4] + "-res.pkl")):
                print("Config file:", params_file, "result file:", os.path.join(path, task_id[:-4] + "-preds.pkl"),
                      "exp:", config_params.get("exp_name"))
                i_mode = mode + " (test_set 2)"
                print(i_mode)
                out = pickle.load(open(os.path.join(path, task_id[:-4] + "-res.pkl"), "rb"))
                if compute_metric:
                    y_pred = load(open(os.path.join(path, task_id[:-4] + "-preds.pkl"), "rb"))
                    y_true = load(open(os.path.join(path, task_id[:-4] + "-targets.pkl"), "rb"))

                    out = __compute_metrics(y_true=y_true, y_pred=y_pred)
                res = dict(split_mode=i_mode,
                           **out, **config_params)
                outs.append(res)

            # outs.append(res)
    return outs


def summarize_experiments(main_dir=None, cm=True, fp="", param_names=None, save_as="", metric_names=None):
    out = None
    default_metric_names = ['macro_roc', 'macro_auprc', "micro_roc", "micro_auprc"]
    if metric_names is not None:
        default_metric_names = metric_names

    if os.path.isfile(fp):
        out = pd.read_csv(fp, index_col=0)
        print(list(out))

    else:
        main_dir = f"{os.path.expanduser('~')}/expts" if main_dir is None else main_dir
        rand = []
        for i, root in enumerate(os.listdir(main_dir)):
            path = os.path.join(main_dir, root)
            print(path)
            if os.path.isdir(path):
                print("__Path__: ", path)
                res = __loop_through_exp(path, compute_metric=cm, param_names=param_names)
                if len(res) > 0:
                    rand.extend(res)

        if rand:
            out = pd.DataFrame(rand)
        else:
            out = pd.DataFrame(__loop_through_exp(main_dir, compute_metric=cm, param_names=param_names))

        out.to_csv(f"../../results/{save_as}-all_raw-exp-res.csv")

    modes = out["split_mode"].unique()
    out["split_mode"].fillna(inplace=True, value="null")
    out["fmode"].fillna(inplace=True, value="null")
    print("modes", modes)
    for mod in modes:
        df = out[out["split_mode"] == mod]
        t = df.groupby(['dataset_name', 'model_name', 'split_mode', "fmode"] + param_names, as_index=True)
        t_mean = t[default_metric_names].mean()
        t_std = t[default_metric_names].std()

        for met in default_metric_names:
            t_mean[met] = t_mean[[met]].round(3).astype(str) + " ± " + t_std[[met]].round(
                3).astype(
                str)
        # t_mean['micro_auprc'] = t_mean[["micro_auprc"]].round(3).astype(str) + " ± " + t_std[["micro_auprc"]].round(
        #     3).astype(str)
        # t_mean['macro_roc'] = t_mean[["macro_roc"]].round(3).astype(str) + " ± " + t_std[["macro_roc"]].round(
        #     3).astype(
        #     str)
        # t_mean['macro_auprc'] = t_mean[["macro_auprc"]].round(3).astype(str) + " ± " + t_std[["macro_auprc"]].round(
        #     3).astype(str)

        t_mean.to_csv(f"../../results/{save_as}-{mod}.csv")
    # return rand


def __collect_data__(config_file, return_config=False):
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    config = json.load(open(config_file, "rb"))
    dataset_params = {param.split(".")[-1]: val for param, val in config.items() if param.startswith("dataset_params")}
    print(config["model_params.network_params.network_name"])
    dataset_params = {key: val for key, val in dataset_params.items() if
                      key in get_data_partitions.__code__.co_varnames}
    dataset_name = dataset_params["dataset_name"]
    input_path = f"{cach_path}/datasets-ressources/DDI/"  # {dataset_name}
    dataset_params["input_path"] = input_path
    train, valid, test1, test2 = get_data_partitions(**dataset_params)
    if return_config:
        return dataset_params, train, valid, test1, test2

    return train, valid, test1, test2


def reload(path):
    # for fp in tqdm(glob.glob(f"{path}/*.pth")):
    train_params = defaultdict(dict)
    config = json.load(open(path, "rb"))
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

    return train_params
    # run_experiment(**train_params, restore_path=fp, checkpoint_path=fp, input_path=None,
    #                output_path="/home/rogia/Documents/exps_results/nw_bin")


# exit()
# raw_data = pd.read_csv(f"/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/3003377s-twosides.tsv",
#                        sep="\t")
# main_backup = raw_data[['stitch_id1', 'stitch_id2', 'event_name', 'confidence']]
# data_dict = {(did1, did2, name.lower()): conf for did1, did2, name, conf in
#              tqdm(main_backup.values.tolist())}
#
# # train, valid, test1, test2 = __collect_data__(fp)
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

    drugs_dict = pickle.load(open("/home/rogia/Documents/projects/drug_drug_interactions/expts/save.p", "rb"))
    drugs_dict = {x: v for x, v in drugs_dict.items()}

    if level == "drugs":
        b = train_drugs.intersection(valid_drugs)
        l = train_drugs.intersection(test1_drugs)
        g = valid_drugs.intersection(test1_drugs)
        prec = dataset_name, seed, detailed_split_scheme, len(train_drugs), len(valid_drugs), len(test1_drugs), len(
            b), len(
            l), len(g)

        d1 = test1_drugs.difference(train_drugs)
        print(len(d1))

        fps = [FingerprintMols.FingerprintMol(x) for x in [drugs_dict[i] for i in train_drugs]]
        train_sim_scores = [DataStructs.FingerprintSimilarity(fgp, me) for fgp in fps for me in fps]
        print(train_sim_scores)
        print(len(train_sim_scores))
        fpss = [FingerprintMols.FingerprintMol(x) for x in [drugs_dict[i] for i in d1]]
        d1_sim_scores = [max([DataStructs.FingerprintSimilarity(fgp, me) for me in fps]) for fgp in fpss]
        print(d1_sim_scores)
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.distplot(train_sim_scores, color=sns.xkcd_rgb['denim blue'], hist_kws={"alpha": 1}, kde=False,
                     norm_hist=True, label="train")
        sns.distplot(d1_sim_scores, color=sns.xkcd_rgb['pastel orange'], hist_kws={"alpha": 1}, kde=False,
                     norm_hist=True,
                     label="test")
        plt.show()
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
            d2 = test2_drugs.difference(train_drugs)
            print(len(d2))
            fpss = [FingerprintMols.FingerprintMol(x) for x in [drugs_dict[i] for i in d2]]
            d2_sim_scores = [max([DataStructs.FingerprintSimilarity(fgp, me) for me in fps]) for fgp in fpss]
            print(d2_sim_scores)
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.distplot(train_sim_scores, color=sns.xkcd_rgb['denim blue'], hist_kws={"alpha": 1}, kde=False,
                         norm_hist=True, label="train")
            sns.distplot(d2_sim_scores, color=sns.xkcd_rgb['pastel orange'], hist_kws={"alpha": 1}, kde=False,
                         norm_hist=True,
                         label="test")
            plt.show()
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
        return mean(k)  # - 0.001

    # ref_smiles = ref_smiles[:2]
    output = list(map(wrapped_partial(compute_sim, smi1a=smi1, smi1b=smi2), tqdm(ref_smiles)))
    assert len(output) == len(ref_smiles)
    output = sorted(output, reverse=True)[:100]
    print(len(output), output)
    return output


def umap_plot(train_feats, test_feats, split_mode, save_as, title, **kwargs):
    import seaborn as sns
    import umap
    sns.set_context("paper", font_scale=2.5)
    sns.set_style("ticks")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fit1 = umap.UMAP(random_state=42, n_components=3, min_dist=0, **kwargs)
    u1 = fit1.fit(train_feats)
    u1 = u1.embedding_
    # plt.scatter(u1[:, 0], u1[:, 1], label="train")
    ax.scatter(u1[:, 0], u1[:, 1], u1[:, 2], label="training")

    if len(test_feats) != 0:
        try:
            fit2 = umap.UMAP(random_state=42, n_components=3, min_dist=0, **kwargs)
            u2 = fit2.fit(test_feats)
            u2 = u2.embedding_
            # plt.scatter(u2[:, 0], u2[:, 1], label="test")
            ax.scatter(u2[:, 0], u2[:, 1], u2[:, 2], label="testing")
        except:
            pass

    if split_mode == "random":
        plt.legend()
    plt.title(f"split = {title}")
    plt.tight_layout()
    plt.savefig(f"/home/rogia/Bureau/figs/{save_as}.png")
    plt.show()


def ssp(filepath, dataset_name, model, split_mode, save_as="", title="", extract=False, **kwargs):
    from side_effects.data.loader import load_smiles, Chem, AllChem, get_data_partitions

    cach_path = str(os.environ["HOME"]) + "/.invivo/cache" + "/datasets-ressources/DDI/"

    result = pd.read_csv(filepath, index_col=0)
    result = result.loc[(result["dataset_name"] == dataset_name) & (result["model_name"] == model) & (
            result["split_mode"] == split_mode)]
    config_file = result["path"].values[-1]
    print("Config: ", config_file)
    params = reload(config_file)
    print(params)

    ds = params["dataset_params"]
    ds["seed"] = 42
    train_data, valid_data, test_data, unseen_test_data = get_data_partitions(**ds,
                                                                              input_path=cach_path, debug=False)
    if split_mode == "leave_drugs_out (test_set 2)":
        test_data = unseen_test_data

    train_drugs = list(set([d1 for (d1, _, _) in train_data.samples] + [d2 for (_, d2, _) in train_data.samples]))
    test_drugs = list(set([d1 for (d1, _, _) in test_data.samples] + [d2 for (_, d2, _) in test_data.samples]))
    # train_test_drugs = list(set(train_drugs).intersection(set(test_drugs)))

    catalog = {}
    path = f"{cach_path}/{dataset_name}/{dataset_name}-drugs-all.csv"
    smiles, idx2name = load_smiles(fname=path)
    molecules = [Chem.MolFromSmiles(X) for X in list(smiles.values())]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 4) for mol in molecules]
    ssp = [[DataStructs.FingerprintSimilarity(drg, fingerprints[i]) for i in range(len(fingerprints))] for
           ids, drg in enumerate(fingerprints)]
    catalog1 = dict(zip(list(smiles.keys()), ssp))
    if kwargs.get("return_all"):
        return catalog1, train_drugs, test_drugs

    return catalog1


def extract_features(filepath, dataset_name, model, split_mode, inputs={}, level="drug"):
    from side_effects.data.loader import load_smiles, get_data_partitions, to_tensor, all_transformers_dict
    from side_effects.trainer import Trainer

    iprefix = model
    cach_path = str(os.environ["HOME"]) + "/.invivo/cache" + "/datasets-ressources/DDI/"

    result = pd.read_csv(filepath, index_col=0)
    result = result.loc[(result["dataset_name"] == dataset_name) & (result["model_name"] == model) & (
            result["split_mode"] == split_mode)]
    config_file = result["path"].values[-1]
    print("Config: ", config_file)
    params = reload(config_file)
    print(params)
    nb_side_effects = 964 if dataset_name == "twosides" else 86
    partitions = {}

    if not inputs:
        ds = params["dataset_params"]
        ds["seed"] = 42
        train_data, valid_data, test_data, unseen_data = get_data_partitions(**ds,
                                                                             input_path=cach_path, debug=False)
        if split_mode == "leave_drugs_out (test_set 2)":
            test_data = unseen_data

        partitions["train"] = train_data
        partitions["test"] = test_data
        partitions["valid"] = valid_data

        path = f"{cach_path}/{dataset_name}/{dataset_name}-drugs-all.csv"
        smiles, idx2name = load_smiles(fname=path)
    else:
        print("OUTSIDE DATASET..", len(inputs))
        smiles = inputs
        transformer_fn = params["dataset_params"]["transformer"]
        transformer = all_transformers_dict.get(transformer_fn)
        smiles = transformer(drugs=list(smiles.keys())
                             , smiles=list(smiles.values()))
        print("Nb SMILES..", len(smiles))

    print(params["model_params"]["network_params"])

    model_params = params["model_params"]
    model_params['network_params'].update(
        dict(nb_side_effects=nb_side_effects))

    if "att_hidden_dim" in model_params["network_params"]:
        del (model_params["network_params"]["att_hidden_dim"])

    if "auxnet_params" in model_params["network_params"]:
        del (model_params["network_params"]["auxnet_params"])

    model = Trainer(network_params=model_params["network_params"],
                    checkpoint_path=config_file[:-12] + ".checkpoint.pth")
    model.model.eval()
    if level == "drug":
        fc = model.model.drug_feature_extractor
        print(fc)
        seq = []
        if partitions["train"] is not None:
            seq = [partitions["train"].drug_to_smiles[d] for d, _ in smiles.items()]

        else:
            seq = list(smiles.values())

        if isinstance(seq[0], (torch.Tensor, np.ndarray)):
            seq = to_tensor(seq, gpu=False, dtype=torch.LongTensor)
            print("Seq. Shape:", seq.shape)

        catalog2 = dict(zip(list(smiles.keys()), fc(seq).detach().numpy()))

        out = {}
        for k in partitions:
            a, b, _ = list(zip(*partitions[k].samples))
            c = list(set(a + b))
            print(k, len(c))
            feats = np.stack([catalog2[kk] for kk in c])
            print("c", feats.shape)
            out[k] = feats

            with open(f'{dataset_name}_{split_mode}_{iprefix}_{k}_drug.feats.npy', 'wb') as f:
                np.save(f, feats)

        return catalog2

    else:
        import torch.nn as nn
        fc = model.model
        fc.classifier.net = nn.Sequential(*list(model.model.classifier.net.children())[:-2])
        transf_dict = partitions["train"].drug_to_smiles
        out = {}
        print(fc)

        for k in partitions:
            a, b, _ = list(zip(*partitions[k].samples))
            a, b = [transf_dict[d] for d in a], [transf_dict[d] for d in b]

            if isinstance(a, (list, torch.Tensor, np.ndarray)):
                a = to_tensor(a, gpu=False, dtype=torch.LongTensor)
                b = to_tensor(b, gpu=False, dtype=torch.LongTensor)
                print("Seq. Shape:", a.shape, b.shape)
                assert a.shape == b.shape

            # let use minibatches to reduce memory use
            batch_size = 1
            lens = a.shape[0]
            n_minibatches = int(lens / batch_size)
            output = []
            for i in range(n_minibatches):
                start = batch_size * i
                ends = min(i * batch_size + batch_size, lens)
                feats = fc([a[start:ends, :], b[start:ends, :]])
                print("Loop ", i, feats.shape)
                output += [feats.detach().numpy()]
            feats = np.concatenate(output, axis=0)
            print("OUT", feats.shape, a.shape)
            assert feats.shape[0] == a.shape[0]

            with open(f'{dataset_name}_{split_mode}_{iprefix}_{k}.feats.npy', 'wb') as f:
                np.save(f, feats)
            out[k] = feats


def rbf_kernel(x, y, gamma=1.0):
    x_y_distance = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    if isinstance(x_y_distance, torch.Tensor):
        res = torch.exp(-1 * x_y_distance / gamma)
    else:
        res = np.exp(-1 * x_y_distance / gamma)
    return res


def __compute_mmd__(x, y, gamma=1.0):
    x_kernel = rbf_kernel(x, x, gamma)
    y_kernel = rbf_kernel(y, y, gamma)
    xy_kernel = rbf_kernel(x, y, gamma)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()


def compute_mmd(path, dataset, mode, model, n_repeats=100, level="drug"):
    if level == "drug":
        train = np.load(f"{path}/{dataset}_{mode}_{model}_train_drug.feats.npy")
        test = np.load(f"{path}/{dataset}_{mode}_{model}_test_drug.feats.npy")
        valid = np.load(f"{path}/{dataset}_{mode}_{model}_valid_drug.feats.npy")
    else:
        train = np.load(f"{path}/{dataset}_{mode}_{model}_train.feats.npy")
        test = np.load(f"{path}/{dataset}_{mode}_{model}_test.feats.npy")
        valid = np.load(f"{path}/{dataset}_{mode}_{model}_valid.feats.npy")

    print(train.shape, test.shape, valid.shape)
    a1, b1, c1 = [], [], []

    def mmd(X, Y, n=1000):
        #
        # x_indexes = np.random.choice(len(X), n)
        # y_indexes = np.random.choice(len(Y), n)
        # x = X[x_indexes, :]
        # y = Y[y_indexes, :]
        #gamma = np.std(np.concatenate([x, y]))

        gamma = np.std(np.concatenate([X, Y]))
       # print("gamma ", gamma)
        #return __compute_mmd__(x, y, gamma=float(gamma))
        return __compute_mmd__(X, Y, gamma=float(gamma))

    n_repeats = 1
    for i in range(n_repeats):
        a, b, c = mmd(train, valid), mmd(train, test), mmd(valid, test)
        a1 += [a]
        b1 += [b]
        c1 += [c]

    mmda1 = str(mean(a1).round(6)) + " ± " + str(np.std(a1).round(3))
    mmdb1 = str(mean(b1).round(6)) + " ± " + str(np.std(b1).round(3))
    mmdc1 = str(mean(c1).round(6)) + " ± " + str(np.std(c1).round(3))
    return mmda1, mmdb1, mmdc1


def plot_umap(path, dataset, mode, model, n_components=2, level="drug"):
    from mpl_toolkits.mplot3d import Axes3D
    if level == "drug":
        train = np.load(f"{path}/{dataset}_{mode}_{model}_train_drug.feats.npy")
        test = np.load(f"{path}/{dataset}_{mode}_{model}_test_drug.feats.npy")
        valid = np.load(f"{path}/{dataset}_{mode}_{model}_valid_drug.feats.npy")
    else:
        train = np.load(f"{path}/{dataset}_{mode}_{model}_train.feats.npy")
        test = np.load(f"{path}/{dataset}_{mode}_{model}_test.feats.npy")
        valid = np.load(f"{path}/{dataset}_{mode}_{model}_valid.feats.npy")

    labels = [0] * len(train)  + [1] * len (valid) + [2] * len(test)

    import umap
    import seaborn as sns
    reducer = umap.UMAP(n_components=n_components)
    ck = np.concatenate([train, valid, test])
    print(ck.shape)
    embedding = reducer.fit_transform(ck)
    print(embedding.shape)

    fig = plt.figure()

    if n_components == 2:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=1, c=labels, cmap='Spectral')
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(4) - 0.5).set_ticks(np.arange(3))
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap='Spectral')

    plt.title(f"{split_mode} | {dataset}")
    plt.savefig(f"{dataset}_{split_mode}_{level}.png")
    plt.show()



def mean_similarity(filepath, dataset_name, model, save_as="", title="", extract=False, **kwargs):
    import seaborn as sns
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.8, "axes.labelsize": 25})
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif': 'Liberation Sans'})

    output = []

    for split_mode in ["random", "leave_drugs_out (test_set 1)", "leave_drugs_out (test_set 2)"]:
        fmri = pd.DataFrame()
        result, train, test = ssp(filepath, dataset_name, model, split_mode, save_as="", title="", extract=False,
                                  return_all=True)
        result = {drug: np.array(sp).std() for drug, sp in result.items()}
        if split_mode == "random":
            test = [result[drug] for drug in test]
        else:
            test = [result[drug] for drug in test if drug not in train]
        print(dataset_name, split_mode, len(test))
        train = [result[drug] for drug in train]

        fmri.loc[:, "mean SSP"] = train + test
        fmri.loc[:, "Partition"] = ["Train"] * len(train) + ["Test"] * len(test)
        fmri.loc[:, "split"] = [split_mode] * (len(train) + len(test))
        output.append(fmri)

    output = pd.concat(output)
    output.replace({"random": "Random", "leave_drugs_out (test_set 1)": "One-unseen",
                    "leave_drugs_out (test_set 2)": "Both-unseen"}, inplace=True)

    print(output)
    g = sns.catplot(y="mean SSP",
                    x="split", hue="Partition",
                    data=output, kind="box",
                    saturation=.5,
                    sharex=False, margin_titles=False)
    plt.show()


def cosine_sim(inputs, split, dataset, **kwargs):
    from sklearn.decomposition import PCA

    dim = min([len(list(drug_features.values())[0]) for method, drug_features in inputs.items()])
    print("Min. len ", dim)

    dim = 64
    pca = PCA(n_components=dim)

    out = {}
    for method, drug_features in inputs.items():
        ot = pca.fit_transform(np.array(list(drug_features.values()))).astype(np.float32)
        out[method] = ot

    from scipy import spatial
    keys = [i for i in list(out.keys()) if i != "ssp"]

    result = pd.DataFrame()
    tmp = []
    label = []
    # for k in keys:
    output = list(map(lambda x, y: 1 - spatial.distance.cosine(x, y), out[keys[0]], out[keys[1]]))
    tmp.extend(output)
    # label.extend([k] * len(output))

    # result.loc[:, "Model"] = label
    result.loc[:, "Cosine similarity"] = tmp
    result.loc[:, "split"] = split
    result.loc[:, "D"] = [dataset] * len(tmp)

    return result


def visualize_drug_features():
    import seaborn as sns

    sns.set_context("paper", rc={"lines.linewidth": 2.8, "axes.labelsize": 20})
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif':
                       'Liberation Sans'})
    file_name = f"/home/rogia/Bureau/figs/figure_comparison_rn_&_gcn_nt_features.png"

    def _sub_features(dataset):
        output = []
        for mod in ["random", "leave_drugs_out (test_set 1)", "leave_drugs_out (test_set 2)"]:
            ref = ssp(dataset_name=dataset,
                      filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
                      split_mode=mod, model="BiLSTM")
            bilstm = extract_features(dataset_name=dataset,
                                      filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
                                      model="BiLSTM", split_mode=mod)

            molgraph = extract_features(dataset_name="twosides",
                                        filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
                                        model="MolGraph", split_mode=mod)

            inputs = dict(
                ssp=ref, bilstm=bilstm, molgraph=molgraph
            )

            res = cosine_sim(inputs, mod, dataset)
            output.append(res)
        data = pd.concat(output)
        return data

    result = pd.concat([_sub_features(d) for d in ["drugbank", "twosides"]])
    result.replace({"random": "Random", "leave_drugs_out (test_set 1)": "One-unseen",
                    "leave_drugs_out (test_set 2)": "Both-unseen"}, inplace=True)
    g = sns.catplot(x="split", y="Cosine similarity",
                    data=result, kind="violin", col="D",
                    saturation=.5, margin_titles=False, legend=True)
    sns.despine(top=False, right=False)

    for ax in g.axes.flat:
        ax.set_ylabel(ax.get_ylabel(), fontsize="xx-large")
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


def train_and_validate_using_ML(model, dataset, mod):
    from rdkit.Chem import PandasTools
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier

    def __run__(X, y):
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        y_values = []
        predictions = []
        probas = []
        for train, test in kf.split(X, y):
            clf = RandomForestClassifier(n_estimators=500, random_state=0)
            clf.fit(X[train], y[train])
            predictions.append(clf.predict(X[test]))
            probas.append(clf.predict_proba(X[test]).T[1])  # )  # Probabilities for class 1
            y_values.append(y[test])
            del clf
        return y_values, probas

    def __rep__(y_values, probas):
        from sklearn.metrics import roc_auc_score, average_precision_score
        aucs = [roc_auc_score(y, proba, average="micro") for y, proba in zip(y_values, probas)]
        aps = [average_precision_score(y, proba, average="micro") for y, proba in zip(y_values, probas)]
        return (np.mean(aucs), np.std(aucs)), (np.mean(aps), np.std(aps))

    df = PandasTools.LoadSDF('../../data/ames.sdf.txt', smilesName='SMILES')
    print(df["ID"].unique())
    print(len(df["ID"].unique()))
    print(len(df))
    smiles_dict = dict(zip(df["ID"], df["SMILES"]))
    feats = extract_features(dataset_name=dataset,
                             filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
                             model=model, split_mode=mod, outside=True, inputs=smiles_dict)

    X = np.array(list(feats.values()))
    y = np.array(df['class'].astype(int))

    print("X ", X.shape, "Y ", y.shape)

    y_val, probs = __run__(X, y)
    (a1, a2), (b1, b2) = __rep__(y_val, probs)
    return (a1, a2), (b1, b2)


def main():
    result = []
    modes = ["random", "leave_drugs_out (test_set 1)", "leave_drugs_out (test_set 2)"]
    datasets = ["drugbank", "twosides"]
    models = ["MolGraph", "BiLSTM"]
    p = [(mode, dat, model) for mode in modes for dat in datasets for model in models]
    for mode, dataset, model in p:
        (a1, a2), (b1, b2) = train_and_validate_using_ML(model, dataset, mode)
        line = [dataset, mode, model, str(a1.round(3)) + " ± " + str(a2.round(3)),
                str(b1.round(3)) + " ± " + str(b2.round(3))]
        result.append(line)
        print(line)
    result = pd.DataFrame(result, columns=["Dataset", "Split", "Model", "micro_roc", "micro_auprc"])
    result.replace({"random": "Random", "leave_drugs_out (test_set 1)": "One-unseen",
                    "leave_drugs_out (test_set 2)": "Both-unseen"}, inplace=True)
    result.to_csv("ames_result.csv")


def reformatter(file, ref="/home/rogia/Documents/exps_results/temp-2/all_raw-exp-res.csv", save_as=""):
    # keep a place to save x smiles
    raw_data = []
    ref = pd.read_csv(ref, index_col=0)
    outptut = pd.read_csv(file, index_col=0)
    print(list(outptut))
    for i in outptut.itertuples():
        i = split, model, seed, dataset, micro_roc, micro_auprc, n_smiles = i.split_mode, set_model_name(
            i.model_name), i.seed, i.dataset_name, i.micro_roc, i.micro_auprc, i._12
        if model == "GIN":
            model = "MolGraph"
        old = ref[(ref["split_mode"] == split) & (ref["model_name"] == model) & (ref["seed"] == seed) & (
                ref["dataset_name"] == dataset)]
        old = old["split_mode"].values[-1], old["model_name"].values[-1], old["seed"].values[-1], \
              old["dataset_name"].values[-1], old["micro_roc"].values[-1], old["micro_auprc"].values[-1], n_smiles
        # print(i)
        # print(old)
        # print("----------------")
        raw_data.append(["ref"] + list(old))
        raw_data.append(["Rand. SMILES"] + list(i))

    raw_data = pd.DataFrame(raw_data,
                            columns=["exp", "split", "Model", "seed", "dataset", "micro_roc", "micro_auprc", "rand"])
    raw_data.to_csv(f"{save_as}_comparing_to_ref_all_raw_data.csv")

    # |Random - One-unseen|, |Random - Both-unseen|, |One-unseen - Both-unseen|
    statistiq = []
    df = raw_data.groupby(["rand"], as_index=False)

    for name, group in df:
        for expname in ["ref", "Rand. SMILES"]:
            stats = pd.DataFrame()
            sb1 = group[group["exp"] == expname]
            sb1.set_index(["split", "dataset"], inplace=True)

            stats.loc["twosides", "|Random - One-unseen|"] = sb1.loc[
                                                                 (sb1.index.get_level_values("split") == "random") & (
                                                                         sb1.index.get_level_values(
                                                                             "dataset") == "twosides"), "micro_auprc"].values - \
                                                             sb1.loc[
                                                                 (sb1.index.get_level_values(
                                                                     "split") == "leave_drugs_out (test_set 1)") & (
                                                                         sb1.index.get_level_values(
                                                                             "dataset") == "twosides"), "micro_auprc"].values
            a = sb1.loc[
                (sb1.index.get_level_values("split") == "random") & (
                        sb1.index.get_level_values(
                            "dataset") == "drugbank"), "micro_auprc"].values
            if len(a) == 0:
                a = [np.nan]  # C'est just je n,ai pas encore les resultats de drugbank

            stats.loc["drugbank", "|Random - One-unseen|"] = a - \
                                                             sb1.loc[
                                                                 (sb1.index.get_level_values(
                                                                     "split") == "leave_drugs_out (test_set 1)") & (
                                                                         sb1.index.get_level_values(
                                                                             "dataset") == "drugbank"), "micro_auprc"].values

            stats.loc["twosides", "|Random - Both-unseen|"] = sb1.loc[
                                                                  (sb1.index.get_level_values("split") == "random") & (
                                                                          sb1.index.get_level_values(
                                                                              "dataset") == "twosides"), "micro_auprc"].values - \
                                                              sb1.loc[
                                                                  (sb1.index.get_level_values(
                                                                      "split") == "leave_drugs_out (test_set 2)") & (
                                                                          sb1.index.get_level_values(
                                                                              "dataset") == "twosides"), "micro_auprc"].values
            stats.loc["drugbank", "|Random - Both-unseen|"] = a - \
                                                              sb1.loc[
                                                                  (sb1.index.get_level_values(
                                                                      "split") == "leave_drugs_out (test_set 2)") & (
                                                                          sb1.index.get_level_values(
                                                                              "dataset") == "drugbank"), "micro_auprc"].values

            stats.loc["twosides", "|One-unseen - Both-unseen|"] = sb1.loc[
                                                                      (sb1.index.get_level_values(
                                                                          "split") == "leave_drugs_out (test_set 1)") & (
                                                                              sb1.index.get_level_values(
                                                                                  "dataset") == "twosides"), "micro_auprc"].values - \
                                                                  sb1.loc[
                                                                      (sb1.index.get_level_values(
                                                                          "split") == "leave_drugs_out (test_set 2)") & (
                                                                              sb1.index.get_level_values(
                                                                                  "dataset") == "twosides"), "micro_auprc"].values
            stats.loc["drugbank", "|One-unseen - Both-unseen|"] = sb1.loc[
                                                                      (sb1.index.get_level_values(
                                                                          "split") == "leave_drugs_out (test_set 1)") & (
                                                                              sb1.index.get_level_values(
                                                                                  "dataset") == "drugbank"), "micro_auprc"].values - \
                                                                  sb1.loc[
                                                                      (sb1.index.get_level_values(
                                                                          "split") == "leave_drugs_out (test_set 2)") & (
                                                                              sb1.index.get_level_values(
                                                                                  "dataset") == "drugbank"), "micro_auprc"].values

            stats.loc[:, "expname"] = expname
            stats.loc[:, "X"] = name
            print(stats)
            statistiq.append(stats)

    output = pd.concat(statistiq)
    print("op", output)
    output.to_csv(f"{save_as}-statistiq.csv")
    # def __main__umaps(model):
    #     for t in ["twosides", "drugbank"]:
    #             e = True
    #             print(f"{t} - {str(e)}")
    #             visualize_drug_features(dataset_name=t,
    #                                     filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                                     model=model, split_mode="random",
    #                                     save_as=f"figure_{t}_{model}_fp_{str(not e)}_feat_distribution_random",
    #                                     title="Random", extract=e)
    #             visualize_drug_features(dataset_name=t, filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                                     model=model, split_mode="leave_drugs_out (test_set 1)",
    #                                     save_as=f"figure_{t}_{model}_fp_{str(not e)}_feat_distribution_one_unseen",
    #                                     title="One-unseen", extract=e)
    #
    #             visualize_drug_features(dataset_name=t,
    #                                     filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                                     model=model, split_mode="leave_drugs_out (test_set 2)",
    #                                     save_as=f"figure_{t}_{model}_fp_{str(not e)}_feat_distribution_both_unseen",
    #                                     title="Both-unseen", extract=e)
    #


if __name__ == '__main__':
    summarize_experiments(main_dir="/home/rogia/.invivo/result/rand", cm=False,
                                                 param_names=["dataset_params.randomized_smiles", "dataset_params.random_type", "dataset_params.isomeric"], save_as="test",
                                                )

    exit()

    outs = []
    for split_mode in ["random", "leave_drugs_out (test_set 1)", "leave_drugs_out (test_set 2)"]:
           # dataset = "twosides"
            #print(dataset, split_mode)
        for dataset in ["twosides", "drugbank"]:

           a, b, c = compute_mmd(path="/home/rogia/Documents/exps_results/smi_1/", model="BiLSTM", mode=split_mode, dataset=dataset, n_repeats=10, level="drug")
           print(dataset, split_mode, a, b, c)
           outs.append([dataset, split_mode, a, b, c])
           # plot_umap(path="/home/rogia/Documents/exps_results/smi_1/", model="BiLSTM", mode=split_mode, dataset=dataset,
                #     n_components=2)


    result = pd.DataFrame(outs, columns=["dataset", "split", "train_valid", "train_test", "test_valid"])

    result.to_csv("single_mol_mmd_no_bootstraping.csv")


    exit()
    # for split_mode in ["random", "leave_drugs_out (test_set 1)", "leave_drugs_out (test_set 2)"]:
    #     extract_features(dataset_name="twosides",
    #                      filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                      model="BiLSTM", split_mode=split_mode, level="pair")

    for split_mode in ["random", "leave_drugs_out (test_set 1)", "leave_drugs_out (test_set 2)"]:
        extract_features(dataset_name="twosides",
                         filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
                         model="BiLSTM", split_mode=split_mode, level="drug")
    exit()

    exit()
    # main()
    # exit()
    # visualize_drug_features()
    #
    # exit()
    # reformatter("../../results/GH_100_RD-all_raw-exp-res.csv", save_as="GH_100_RD")
    #
    reformatter("../../results/RN_100_RD-all_raw-exp-res.csv", save_as="RN_100_RD")

    exit()
    # mean_similarity(dataset_name="drugbank",
    #                 filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                 model="BiLSTM")
    #
    # mean_similarity(dataset_name="twosides",
    #                 filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                 model="BiLSTM")
    #
    # exit()
    # visualize_drug_features()
    # exit()
    #
    # visualize_loss_progress("/media/rogia/CLé USB/expts/BiLSTM/drugbank_bmnddi_c1ac805a_log.log", n_epochs=100)
    # exit()
    # __main__umaps("BiLSTM")
    # __main__umaps("MolGraph")
    # e = False
    # model = "BiLSTM"
    # for t in ["twosides", "drugbank"]:
    #     print(f"{t} - {str(e)}")
    #     visualize_drug_features(dataset_name=t,
    #                             filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                             model=model, split_mode="random",
    #                             save_as=f"figure_{t}_{model}_fp_{str(not e)}_feat_distribution_random",
    #                             title="Random", extract=e)
    #     visualize_drug_features(dataset_name=t, filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                             model=model, split_mode="leave_drugs_out (test_set 1)",
    #                             save_as=f"figure_{t}_{model}_fp_{str(not e)}_feat_distribution_one_unseen",
    #                             title="One-unseen", extract=e)
    #
    #     visualize_drug_features(dataset_name=t,
    #                             filepath="/home/rogia/Documents/exps_results/temp-2/temp.csv",
    #                             model=model, split_mode="leave_drugs_out (test_set 2)",
    #                             save_as=f"figure_{t}_{model}_fp_{str(not e)}_feat_distribution_both_unseen",
    #                             title="Both-unseen", extract=e)

    # # exit()
    # # visualize_drug_features(dataset_name="drugbank")
    #
    # exit()
    # visualize_loss_progress("/media/rogia/CLé USB/expts/BiLSTM/twosides_bmnddi_a556e5b3_log.log")
    # exit()
    # summarize_experiments(main_dir="/home/rogia/.invivo/result/NegS", cm=False,
    #                       param_names=["model_params.loss_params.use_negative_sampling"], save_as="negative_samp",
    #                      )

    # summarize_experiments(main_dir="/home/rogia/.invivo/result/GIN_MLT", cm=False, param_names=["model_params.weight_decay"], save_as="graph_multitask", metric_names=["miroc", "miaup", "mse", "maroc", "maaup"])
    # summarize_experiments(fp="/home/rogia/Documents/exps_results/temp-2/all_raw-exp-res.csv")
    # exit()
    # summarize_experiments(main_dir="/home/rogia/.invivo/result/GH_RAND_SMI_V1_ALL_GRAND_BATCH", cm=False,
    #                       param_names=["dataset_params.randomized_smiles"], save_as="GH_100_RD")
    # #
    exit()
    summarize_experiments(main_dir="/home/rogia/.invivo/result/RN_RAND_SMI_V1_ALL_GRAND_BATCH", cm=False,
                          param_names=["dataset_params.randomized_smiles"], save_as="RN_100_RD")

    exit()

    # path = "/home/rogia/.invivo/result/RN_RAND_SMI_V1_ALL_GRAND_BATCH"
    # for fp in glob.glob(f"{path}/*.log"):
    #     print(fp)
    #     visualize_loss_progress(os.path.join(path, fp))

    exit()

    visualize_loss_progress("/home/rogia/.invivo/result/randomized_smi_cnn/twosides_bmnddi_7982c9b9_log.log")
    visualize_loss_progress("/home/rogia/.invivo/result/randomized_smi_cnn/twosides_bmnddi_b812d329_log.log")

    exit()
    visualize_loss_progress("/home/rogia/.invivo/result/GIN_MLT/drugbank_L1000_bmnddi_195ea68d_log.log",
                            save_as="gin_drugbank_mlt.png")
    visualize_loss_progress("/home/rogia/.invivo/result/GIN_MLT/twosides_L1000_bmnddi_06e744f2_log.log",
                            save_as="twosides_drugbank_mlt.png")
    exit()
    visualize_loss_progress("/home/rogia/.invivo/result/NegS/twosides_bmnddi_dd7a577f_log.log")
    #
    # exit()
    #
    # summarize_experiments("/home/rogia/.invivo/result/res", cm=False)
    # get_best_hp()
    # exit()
    # -5
    # visualize_loss_progress("/home/rogia/.invivo/result/same_dataset_LO/twosides_L1000_bmnddi_3c0355b2_log.log")
    # -6
    # visualize_loss_progress("/home/rogia/.invivo/result/same_dataset_LO/twosides_L1000_bmnddi_d60506d4_log.log")
    # -4

    exit()
    visualize_loss_progress("/home/rogia/.invivo/result/same_dataset_LO/twosides_L1000_bmnddi_e0f8ef99_log.log",
                            n_epochs=20, title="C1-No Reg.")
    import pickle as pk

    print(pk.load(open("/home/rogia/.invivo/result/same_dataset_LO/twosides_L1000_bmnddi_e0f8ef99_res.pkl", "rb")))
    print(pk.load(open("/home/rogia/.invivo/result/same_dataset_LO/twosides_L1000_bmnddi_e0f8ef99-res.pkl", "rb")))
    # exit()
    # load_learning_curves(["/home/rogia/.invivo/result/Pretraining_F1/twosides_bmnddi_7c0f7c7b_log.log",
    #                       "/home/rogia/.invivo/result/Pretraining_F2/twosides_bmnddi_32b0f70d_log.log"])

    visualize_loss_progress("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_3f971e92_log.log",
                            n_epochs=20, title=json.load(
            open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_3f971e92_params.json"))[
            "model_params.network_params.dropout"])

    print(json.load(
        open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_3f971e92_params.json"))[
              "model_params.network_params.dropout"])
    print(pk.load(open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_3f971e92_res.pkl", "rb")))
    print(pk.load(open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_3f971e92-res.pkl", "rb")))
    visualize_loss_progress("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_06cffa40_log.log",
                            n_epochs=20, title=json.load(
            open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_06cffa40_params.json"))[
            "model_params.network_params.dropout"])
    print(json.load(
        open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_06cffa40_params.json"))[
              "model_params.network_params.dropout"])
    print(pk.load(open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_06cffa40_res.pkl", "rb")))
    print(pk.load(open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_06cffa40-res.pkl", "rb")))
    visualize_loss_progress("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_c98a7ab3_log.log",
                            n_epochs=20, title=json.load(
            open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_c98a7ab3_params.json"))[
            "model_params.network_params.dropout"])
    print(json.load(
        open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_c98a7ab3_params.json"))[
              "model_params.network_params.dropout"])
    print(pk.load(open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_c98a7ab3_res.pkl", "rb")))
    print(pk.load(open("/home/rogia/.invivo/result/RegT/twosides_L1000_bmnddi_c98a7ab3-res.pkl", "rb")))

    # load_learning_curves(["/home/rogia/.invivo/result/on_multi_dataset/twosides_bmnddi_d6ee066d_log.log",
    #                       "/home/rogia/.invivo/result/same_dataset/twosides_L1000_bmnddi_ea7b8d1f_log.log"])
    # visualize_loss_progress("/home/rogia/Téléchargements/twosides_bmnddi_d6ee066d_log.log", n_epochs=100)
    # visualize_loss_progress("/home/rogia/.invivo/result/same_dataset/twosides_L1000_bmnddi_ea7b8d1f_log.log", n_epochs=20)
    # print(pk.load(open("/home/rogia/.invivo/result/same_dataset/twosides_L1000_bmnddi_ea7b8d1f_res.pkl", "rb")))
    # visualize_loss_progress("/home/rogia/Téléchargements/RegT/twosides_L1000_bmnddi_48c6cca1_log.log", n_epochs=100)
    # visualize_loss_progress("/home/rogia/Téléchargements/cmap_pretrain/L1000_bmnddi_2bb10187_log.log",
    #                         n_epochs=100)
    # visualize_loss_progress("/home/rogia/Téléchargements/cmap_pretrain/L1000_bmnddi_f3f9d5de_log.log",
    #                         n_epochs=100)
    # visualize_loss_progress("/home/rogia/Téléchargements/cmap_pretrain/L1000_bmnddi_8d931fff_log.log",
    #                        n_epochs=200)
    # exit()
    # visualize_loss_progress("/home/rogia/.invivo/result/Pretraining_F1/twosides_bmnddi_7c0f7c7b_log.log")
    # visualize_loss_progress("/home/rogia/.invivo/result/Pretraining_F2/twosides_bmnddi_32b0f70d_log.log", n_epochs=76)
    # /home/rogia/Téléchargements/same_dataset
    # visualize_loss_progress("/home/rogia/Téléchargements/twosides_bmnddi_d6ee066d_log.log", n_epochs=10)
    # exit()
    #    # get_dataset_stats(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json")
    #    # __get_dataset_stats__("/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_b01dd329_params.json", level="drugs")
    # #exit()
    #     #summarize_experiments("/media/rogia/5123-CDC3/NOISE/", cm=False)
    #     #exit()
    #     #get_best_hp()
    #     #exit()
    #     # # ('cid000003345', 'cid000005076', 'malignant melanoma')
    #     # input_path = f"/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/3003377s-twosides.tsv"
    #     # data = pd.read_csv(input_path, sep="\t")
    #     # print(list(data))
    #     # l = data[(data['drug1'].str.lower() == 'fenofibrate') & (data['drug2'].str.lower() == 'valsartan') & (data['event_name'].str.lower() == 'hepatitis c')]
    #     # print(l)
    #     #
    #     #
    #     # exit()
    #     # request_pairs = [("fenofibrate", "valsartan", "hepatitis c"), ("verapamil", "fluvoxamine", 'hepatitis c'),
    #     #                  ("bupropion", "fluoxetine", 'hepatitis a'), ("bupropion", "fluoxetine", "hiv disease"),
    #     #                  ("paroxetine", "risperidone", "hiv disease"), ("mupirocin", "sertraline", "drug withdrawal"),
    #     #                  ("amlodipine", "cerivastatin", "flu"), ("metoclopramide", "minoxidil", "road traffic accident"),
    #     #                  ("n-acetylcysteine", "cefuroxime", "herpes simplex"),
    #     #                  ("clonazepam", "salmeterol", "adverse drug effect")]
    #     # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=1,
    #     #        req_pairs=request_pairs, inf=0.6, sup=1.,  save_filename="tab9") # goog predictions
    #     # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='leave_drugs_out (test_set 1)', label=1,
    #     #            req_pairs=request_pairs)
    #     # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='leave_drugs_out (test_set 2)', label=1,
    #     #            req_pairs=request_pairs)
    #
    #     request_pairs = [("bupropion", "benazepril", "heart attack"), ("didanosine", "stavudine", "acidosis"),
    #                      ("bupropion", "fluoxetine", "panic attack"), ("bupropion", "orphenadrine", "muscle spasm"),
    #                      ("tramadol", "zolpidem", "diaphragmatic hernia"), ("paroxetine", "fluticasone", "muscle spasm"),
    #                      ("paroxetine", "fluticasone", "fatigue"), ("fluoxetine", "methadone", "pain"),
    #                      ("carboplatin", "cisplatin", "blood sodium decreased"),
    #                      ("chlorthalidone", "fluticasone", "high blood pressure")]
    #
    #     # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=0,
    #     #            req_pairs=request_pairs, inf=0.0, sup=0.1, save_filename="tab-10")
    #     # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=0,
    #     #            req_pairs=request_pairs, inf=0.7, sup=1., save_filename="bad_pred")
    #     find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=1,
    #                req_pairs=request_pairs, inf=0.7, sup=1, save_filename="1_as_1")
    #     find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=1,
    #                req_pairs=request_pairs, inf=0., sup=0.1, save_filename="1_as_0")
    #     find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=0,
    #                req_pairs=request_pairs, inf=0., sup=0.1, save_filename="0_as_0")
    #     find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='random', label=0,
    #                req_pairs=request_pairs, inf=0.7, sup=1., save_filename="0_as_1")
    #     exit()
    #     # find_pairs("/home/rogia/Documents/projects/temp-2/all_raw-exp-res.csv", mod='leave_drugs_out (test_set 2)', label=0,
    #     #            req_pairs=request_pairs)
    #
    #     # # analyze_models_predictions(fp="/home/rogia/Documents/projects/temp-2/temp.csv", split_mode="leave_drugs_out (test_set 1)", false_pos=1., false_neg=0.)
    #     # # analyze_models_predictions(fp="/home/rogia/Documents/projects/temp-2/temp.csv",
    #     # #                            split_mode="leave_drugs_out (test_set 2)",false_pos=1., false_neg=0)
    #     # # # # cli()
    #     # # #
    #     # # #  reload(path="/home/rogia/Documents/exps_results/binary/cl_gin")
    #     # #exit()
    #     # # # summarize_experiments("/media/rogia/5123-CDC3/SOC", cm=True)
    #     # # # get_best_hp()
    #     # # # exit()
    #     # #
    #     # # # brou("/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json")
    #     # # # exit()
    #     # # # raw_data = pd.read_csv(f"/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/3003377s-twosides.tsv", sep="\t")
    #     # # # print(list(raw_data))
    #     # # # print(raw_data[(raw_data["drug1"].str.lower() == "aprepitant")  & (raw_data["drug2"].str.lower() == "cisplatin")  & (raw_data["event_name"].str.lower() =="hepatitis c" )][["drug2", "event_name"]])  #
    #     # # #
    #     # # # exit()
    #     # # # summarize_experiments(main_dir="/home/rogia/Documents/exps_results")
    #     # analyse_kfold_predictions(mod="random", config="CNN", label=1, sup=0.1, inf=0.)
    #     # analyse_kfold_predictions(mod="random", config="CNN", label=0, sup=1., inf=0., selected_pairs=request_pairs)
    #     # analyse_kfold_predictions(mod="leave_drugs_out (test_set 1)", config="CNN", label=0, sup=1., inf=0., selected_pairs=request_pairs)
    #     # # # analyse_kfold_predictions(mod="leave_drugs_out (test_set 1)", config="CNN", label=1, sup=0.1, inf=0.)
    #     # # # analyse_kfold_predictions(mod="leave_drugs_out (test_set 1)", config="CNN", label=0, sup=1., inf=0.7)
    #     # #
    #     # res = pd.read_csv("/home/rogia/Documents/projects/temp-SOC/all_raw-exp-res.csv", index_col=0)
    #     # # y1 = res[res["split_mode"] == "random"]
    #     # # y1 = y1.groupby(["test_fold", "model_name"], as_index=True)[["micro_roc", "micro_auprc"]].max()
    #     # # y1.to_csv("../../results/random_exp_grouped_by_test_fold.csv")
    #     # # y2 = res[res["split_mode"] == "leave_drugs_out (test_set 1)"]
    #     # # y2 = y2.groupby(["test_fold", "model_name"], as_index=True)[
    #     # #     ["micro_roc", "micro_auprc"]].max()
    #     # # y2.to_csv("../../results/early_exp_grouped_by_test_fold.csv")
    #     # # exit()
    #     # # visualize_loss_progress("../../../twosides_bmnddi_6b85a591_log.log")
    #     # # exit()
    #     # # #
    #     # # # exit()
    #     # # #
    #     # # #
    #     # # # exit()
    #     # # visualize_loss_progress("/home/rogia/Documents/projects/twosides_deepddi_7a8da2c0_log.log")
    #     # #
    #     # # exit()
    #     # #
    #     # # exit()
    #     # # get_dataset_stats(exp_path="/media/rogia/CLé USB/expts/DeepDDI", level='pair')
    #     # # get_dataset_stats(exp_path="/media/rogia/CLé USB/expts/DeepDDI", level='drugs')
    #     # # exit()
    #     # # df = __get_dataset_stats__("/home/rogia/.invivo/result/cl_deepddi/twosides_deepddi_4ebbb144_params.json")
    #     # # print(df)
    #     # # exit()
    #     # # # summarize_experiments("/media/rogia/CLé USB/expts/")
    #     # # visualize_test_perf()
    #     # #
    #     # # exit()
    #     # # # bt = pd.read_csv("../../results/best_hp.csv")
    #     # # # # c = bt[bt["model_name"] == 'CNN'][['dataset_name', 'task_id', 'split_mode', 'path']].values
    #     # # # # print(c)
    #     # # # c = bt[(bt["dataset_name"] == 'twosides') & (bt["split_mode"] == "random")][
    #     # # #     ['dataset_name', "model_name", 'task_id', 'split_mode', 'path']].values
    #     # # # analyze_models_predictions([c[::-1][1]])
    #     # #
    #     # # # analysis_test_perf(c)
    #     # # exit()
    #     # # get_dataset_stats(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json")
    #     # # exit()
    #     # # # get_misclassified_labels(
    #     # # #     "/home/rogia/Documents/projects/drug_drug_interactions/results/twosides_bmnddi_6936e1f9_1_sis.csv", items=10)
    #     # # # exit()
    #     # # # combine(exp_path="/media/rogia/CLé USB/expts/DeepDDI", split=False)
    #     # # exit()
    #     # # analyse_model_predictions(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", label=1, threshold=0.1)
    #     # # analyse_model_predictions(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", label=0, threshold=0.7)
    #     # # choose o.1 as threshold for false positive
    #     # # combine(["/home/rogia/Documents/projects/twosides_bmnddi_390811c9",
    #     # #          "/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9"])
    #     #
    #     # exit()
    #     # exp_folder = "/home/rogia/Documents/exps_results/IMB"  # /media/rogia/CLé USB/expts"  # f"{os.path.expanduser('~')}/expts"
    #     # # summarize(exp_folder)
    #     # # Scorer twosides random split
    #     # scorer("twosides", task_path="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", save="random")
    #     # # Scorer twosides early split stage
    #     # scorer("twosides", task_path="/home/rogia/Documents/projects/twosides_bmnddi_390811c9", save="early+stage")
    #     # exit()
    #     # scorer("twosides", task_path="/home/rogia/Documents/exps_results/imb/twosides_bmnddi_48d052b6", save="cnn+NS")
    #     # scorer("twosides", task_path="/home/rogia/Documents/exps_results/imb/twosides_bmnddi_5cbedb4a", save="cnn+CS")
    #     #
    #     # # Scorer drugbank random split
    #     # scorer("drugbank", task_path="/media/rogia/CLé USB/expts/CNN/drugbank_bmnddi_54c9f5bd", save="drugbank")
    #     # # Scorer drugbank early split
    #     # exit()
    #     #
    #     # f = __loop_through_exp(exp_folder + "/CNN")
    #     # # # # # d = __loop_through_exp(exp_folder + "/BiLSTM")
    #     # # # # # a = __loop_through_exp(exp_folder + "/DeepDDI")
    #     # # # # # b = __loop_through_exp(exp_folder + "/Lapool")
    #     # # # # # c = __loop_through_exp(exp_folder + "/MolGraph")
    #     # # # # #
    #     # # # # # e = a + b + c + d
    #     # out = pd.DataFrame(f)
    #     # print(out)
    #     # out.to_csv("temp.csv")
    #     # t = out.groupby(['dataset_name', 'model_name', "split_mode"], as_index=True)[
    #     #     ["micro_roc", "micro_auprc"]].mean()
    #     # print(t)
    #     # # out.to_csv("sum-exp.xlsx")
    #     # # get_exp_stats("sum-exp.xlsx")
    #     # # exit()
    #     # visualize_loss_progress("/home/rogia/Documents/lee/drugbank_adnn_bc985d58_log.log")
    #     # exit()
    #     # summarize()
    #     # # display("/home/rogia/Documents/cl_lstm/_hp_0/twosides_bmnddi_8aa095b0_log.log")
    #     # # display(dataset_name="drugbank", exp_folder="/home/rogia/Documents/no_e_graph/_hp_0")
    #     # # display(dataset_name="twosides", exp_folder="/home/rogia/Documents/cl_lstm/_hp_0/")
    #     # # display_H("/home/rogia/Documents/graph_mol/drugbank_bget_expt_results("/home/rogia/Documents/graph_mol/")mnddi_a556debd_log.log")
    #     # # display_H("/home/rogia/.invivo/result/cl_lapool/drugbank_bmnddi_12c974de_log.log")
    #     # # exit()
    #     # c = get_expt_results("/home/rogia/.invivo/result/")
    #     # print(c)
    #     # exit()
    #     # c.to_excel("graph_mol_as_extract.xlsx")
