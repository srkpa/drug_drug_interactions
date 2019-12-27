import glob
import json
import operator
import os
import pickle
import shutil
import tempfile
from collections import Counter
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ivbase.utils.aws import aws_cli

from side_effects.data.loader import get_data_partitions
from side_effects.metrics import auprc_score, roc_auc_score

INVIVO_RESULTS = ""  # os.environ["INVIVO_RESULTS_ROOT"]


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


# /media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json -- best result
def scorer(dataset_name, task_path=None, upper_bound=None, save=None):
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    data_set = pd.read_csv(f"{cach_path}/datasets-ressources/DDI/{dataset_name}/{dataset_name}.csv",
                           sep=",")
    ddi_types = [e for l in data_set["ddi type"].values.tolist() for e in l.split(";")]
    n_samples = dict(Counter(ddi_types))
    g = list(set(ddi_types))
    g.sort()
    y_preds = load(open(task_path + "_preds.pkl", "rb"))
    y_true = load(open(task_path + "_targets.pkl", "rb"))
    ap_scores = auprc_score(y_pred=y_preds, y_true=y_true, average=None)
    rc_scores = roc_auc_score(y_pred=y_preds, y_true=y_true, average=None)
    # Label ranking
    ap_scores_ranked = sorted(zip(g, ap_scores), key=operator.itemgetter(1))
    rc_scores_dict = dict(zip(g, list(rc_scores)))
    btm_5_ap_scores = ap_scores_ranked[:10]
    btm_5_ap_rc_scores = [(l, v, rc_scores_dict[l]) for l, v in btm_5_ap_scores]
    top_5_ap_scores = ap_scores_ranked[::-1][:10]
    top_5_ap_rc_scores = [(l, v, rc_scores_dict[l]) for l, v in top_5_ap_scores]

    df = pd.DataFrame(top_5_ap_rc_scores, columns=["Best performing DDI types", "AUPRC", "AUROC"])
    df1 = pd.DataFrame(btm_5_ap_rc_scores, columns=["Worst performing DDI types", "AUPRC", "AUROC"])
    prefix = os.path.basename(task_path)
    df.to_csv(f"../../results/{prefix}_{save}_top.csv")
    df1.to_csv(f"../../results/{prefix}_{save}_bottom.csv")
    print(df)
    print(df1)
    # zoom
    x = [n_samples[g[i]] for i in range(len(ap_scores))]
    if upper_bound:
        x, y = [j for i, j in enumerate(x) if j <= 2000], [i for i, j in enumerate(x) if j <= 2000]
        ap_scores = [ap_scores[i] for i in y]
        rc_scores = [rc_scores[i] for i in y]
    plt.figure(figsize=(8, 6))
    s = np.array([ap_scores, rc_scores])
    s = np.mean(s, axis=0).tolist()
    print(s)
    plt.scatter(x, ap_scores, 40, marker="o", color='xkcd:lightish blue', alpha=0.7, zorder=1, label="AUPRC")
    plt.scatter(x, rc_scores, 40, marker="o", color='xkcd:lightish red', alpha=0.7, zorder=1, label="AUROC")
    plt.axhline(y=0.5, color="gray", linestyle="--", zorder=2, alpha=0.6)
    plt.text(max(x), 0.5, "roc=aup=0.5", horizontalalignment="right", size=12, family='sherif', color="r")
    plt.xlabel('Number of positives samples', fontsize=10)
    plt.ylabel('Scores', fontsize=10)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    plt.title(save)
    plt.legend()
    plt.savefig(f"../../results/{prefix}_{save}_scores.png")
    # plt.show()


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
            exp_name = "CNN + GIN"
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
            fmode = config.get("model_params.network_params.mode", None)
            if task_id.endswith("_res"):
                i_mode = ""
                print("Config file:", params_file, "result file:", fp, "exp:", exp_name)
                out = pickle.load(open(fp, "rb"))
                if compute_metric:
                    y_pred = load(open(os.path.join(path, task_id[:-3] + "preds.pkl"), "rb"))
                    y_true = load(open(os.path.join(path, task_id[:-3] + "targets.pkl"), "rb"))
                    out = __compute_metrics(y_true=y_true, y_pred=y_pred)

                    if mode == "leave_drugs_out":
                        i_mode = mode + " (test_set 1)"
                    else:
                        i_mode = mode
                res = dict(model_name=exp_name, dataset_name=dataset_name, split_mode=i_mode, seed=seed,
                           task_id=task_id,
                           path=params_file, fmode=fmode,
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
                           task_id=task_id,
                           path=params_file, fmode=fmode,
                           **out)
                outs.append(res)

            # outs.append(res)
    return outs


def summarize(exp_folder=None):
    exp_folder = f"{os.path.expanduser('~')}/expts" if exp_folder is None else exp_folder
    rand = []
    for i, root in enumerate(os.listdir(exp_folder)):
        path = os.path.join(exp_folder, root)
        if os.path.isdir(path):
            print("__Path__: ", path)
            res = __loop_through_exp(path, compute_metric=True)
            print(res)
            if len(res) > 0:
                rand.extend(res)
    print(rand)
    if rand:
        out = pd.DataFrame(rand)
        out.to_csv("../../results/raw-exp-res.csv")
        modes = out["split_mode"].unique()
        out["split_mode"].fillna(inplace=True, value="null")
        for mod in modes:
            df = out[out["split_mode"] == mod]
            t = df.groupby(['dataset_name', 'model_name', 'split_mode', "fmode"], as_index=True)
            t_mean = t[["micro_roc", "micro_auprc"]].mean()
            t_std = t[["micro_roc", "micro_auprc"]].std()
            print(t_std)
            t_mean['micro_roc'] = t_mean[["micro_roc"]].round(3).astype(str) + " ± " + t_std[["micro_roc"]].round(
                3).astype(
                str)
            t_mean['micro_auprc'] = t_mean[["micro_auprc"]].round(3).astype(str) + " ± " + t_std[["micro_auprc"]].round(
                3).astype(str)
            t_mean.to_csv(f"../../results/{mod}.csv")


def analyse_model_predictions(task_id, threshold=0.1, label=1):
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    pref = os.path.basename(task_id)
    y_pred = load(open(task_id + "-preds.pkl", "rb"))
    y_true = load(open(task_id + "-targets.pkl", "rb"))
    config = json.load(open(task_id + "_params.json", "rb"))
    print(config)
    dataset_params = {param.split(".")[-1]: val for param, val in config.items() if param.startswith("dataset_params")}
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
        print(list(pos_bk)[0])
        pos_pairs = set([p[:3] for p in pairs])
        print(list(pos_pairs)[0])
        common = pos_pairs.intersection(pos_bk)
        print(len(common), len(pos_pairs))
        print(pos_pairs - common)
        is_ok = len(pos_pairs - common) == 0
    else:
        print(f"check if all pairs exist (Negative pairs)=> {len(pairs)}!")
        neg_bk = set(neg_bk)  # all the drugs pairs that really exists
        neg_pairs = set([p[:3] for p in pairs])  # my negative pairs
        neg_drugs_comb = set([p[:2] for p in pairs])
        common = set(pos_bk).intersection(neg_pairs)
        print(len(neg_pairs), len(common))
        is_ok = len(common) == 0 and len(neg_drugs_comb - neg_bk) == 0
    print("Done!")
    assert is_ok == True, "Everything is not ok!"
    # save it
    pairs = [(drugs_dict[p[0]],) + (drugs_dict[p[1]],) + p[2:] for p in pairs]
    res = pd.DataFrame(pairs, columns=["drug_1", "drug_2", "side effect", "probability"])
    print(res)
    res.to_csv(f"../../results/{pref}_{label}_sis.csv")


if __name__ == '__main__':
    # analyse_model_predictions(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", label=1, threshold=0.1)
    # analyse_model_predictions(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", label=0, threshold=0.7)
    # choose o.1 as threshold for false positive
    # exit()
    exp_folder = "/home/rogia/Documents/exps_results/IMB"  # /media/rogia/CLé USB/expts"  # f"{os.path.expanduser('~')}/expts"
    # summarize(exp_folder)
    # scorer("twosides", task_path="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", save="cnn")
    scorer("twosides", task_path="/home/rogia/Documents/exps_results/imb/twosides_bmnddi_48d052b6", save="cnn+NS")
    scorer("twosides", task_path="/home/rogia/Documents/exps_results/imb/twosides_bmnddi_5cbedb4a", save="cnn+CS")
    # scorer("drugbank", task_path="/media/rogia/CLé USB/expts/CNN/drugbank_bmnddi_54c9f5bd", save="drugbank")
    exit()

    f = __loop_through_exp(exp_folder + "/CNN")
    # # # # d = __loop_through_exp(exp_folder + "/BiLSTM")
    # # # # a = __loop_through_exp(exp_folder + "/DeepDDI")
    # # # # b = __loop_through_exp(exp_folder + "/Lapool")
    # # # # c = __loop_through_exp(exp_folder + "/MolGraph")
    # # # #
    # # # # e = a + b + c + d
    out = pd.DataFrame(f)
    print(out)
    out.to_csv("temp.csv")
    t = out.groupby(['dataset_name', 'model_name', "split_mode"], as_index=True)[
        ["micro_roc", "micro_auprc"]].mean()
    print(t)
    # # out.to_csv("sum-exp.xlsx")
    # # get_exp_stats("sum-exp.xlsx")
    # # exit()
    # visualize_loss_progress("/home/rogia/Documents/lee/drugbank_adnn_bc985d58_log.log")
    # exit()
    # summarize()
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
