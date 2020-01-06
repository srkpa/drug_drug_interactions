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


def visualize_test_perf(fp="../../results/temp.csv", model_name="CNN"):
    bt = pd.read_csv(fp)
    c = bt[bt["model_name"] == model_name][['dataset_name', 'task_id', 'split_mode', 'path']].values
    n = len(c) // 2
    print(len(c), n)
    fig, ax = plt.subplots(nrows=n - 1, ncols=n, figsize=(15, 9))
    i = 0
    for row in ax:
        for col in row:
            dataset, task_id, split, path = c[i]
            print(split)
            scorer(dataset_name=dataset, task_path=path[:-12], split=split, f1_score=True, ax=col)
            i += 1
    plt.savefig(f"../../results/scores.png")


# plt.show()


# /media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json -- best result
def scorer(dataset_name, task_path=None, upper_bound=None, f1_score=True, split="random", ax=None, annotate=False):
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    data_set = pd.read_csv(f"{cach_path}/datasets-ressources/DDI/{dataset_name}/{dataset_name}.csv",
                           sep=",")
    ddi_types = [e for l in data_set["ddi type"].values.tolist() for e in l.split(";")]
    n_samples = dict(Counter(ddi_types))
    g = list(set(ddi_types))
    g.sort()
    y_preds = load(open(task_path + "_preds.pkl", "rb")) if split != "leave_drugs_out (test_set 2)" else load(
        open(task_path + "-preds.pkl", "rb"))
    y_true = load(open(task_path + "_targets.pkl", "rb")) if split != "leave_drugs_out (test_set 2)" else load(
        open(task_path + "-targets.pkl", "rb"))
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
    df.to_csv(f"../../results/{prefix}_{split}_top.csv")
    df1.to_csv(f"../../results/{prefix}_{split}_bottom.csv")
    print(df)
    print(df1)
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
        print(f1_scores)
        if ax is None:
            plt.figure(figsize=(8, 6))
            plt.scatter(x, f1_scores, 40, marker="+", color='xkcd:lightish blue', alpha=0.7, zorder=1, label="F1 score")
            plt.axhline(y=0.5, color="gray", linestyle="--", zorder=2, alpha=0.6)
        else:
            ax.scatter(x, f1_scores, 40, marker="+", color='xkcd:lightish blue', alpha=0.7, zorder=1, label="F1 score")
            ax.axhline(y=0.5, color="gray", linestyle="--", zorder=2, alpha=0.6)
            ax.set_xlabel('Number of positives samples', fontsize=10)
            ax.set_ylabel('F1 Scores', fontsize=10)

    else:
        plt.scatter(x, ap_scores, 40, marker="o", color='xkcd:lightish blue', alpha=0.7, zorder=1, label="AUPRC")
        plt.scatter(x, rc_scores, 40, marker="o", color='xkcd:lightish red', alpha=0.7, zorder=1, label="AUROC")
        plt.axhline(y=0.5, color="gray", linestyle="--", zorder=2, alpha=0.6)
        plt.text(max(x), 0.5, "mean = 0.5", horizontalalignment="right", size=12, family='sherif', color="r")

    # ax.set_title(tile, fontsize=10)

    # # plt.gca().spines['top'].set_visible(False)
    # # plt.gca().spines['right'].set_visible(False)
    # plt.title(f"{split}-{dataset_name}")
    # plt.legend()
    # plt.savefig(f"../../results/{prefix}_{save}_scores.png")


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


def summarize_experiments(main_dir=None):
    main_dir = f"{os.path.expanduser('~')}/expts" if main_dir is None else main_dir
    rand = []
    for i, root in enumerate(os.listdir(main_dir)):
        path = os.path.join(main_dir, root)
        if os.path.isdir(path):
            print("__Path__: ", path)
            res = __loop_through_exp(path, compute_metric=True)
            print(res)
            if len(res) > 0:
                rand.extend(res)
    print(rand)
    if rand:
        out = pd.DataFrame(rand)
        best_hp = out.groupby(["dataset_name", "model_name", "split_mode"], as_index=True).max()
        print("best hp", best_hp)
        best_hp.to_csv("../../results/best_hp.csv")
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


def analyze_models_predictions(fp="../../results/temp.csv", split_mode="random", dataset_name="twosides"):
    bt = pd.read_csv(fp)
    items = bt[(bt["dataset_name"] == dataset_name) & (bt["split_mode"] == split_mode)][
        ['dataset_name', "model_name", 'task_id', 'split_mode', 'path']].values
    print(len(items))
    i = 1
    for dataset, mn, task_id, split, path in [items[::-1][1]]:
        print("no ", i, "task_path ", task_id, mn)
        __analyse_model_predictions__(task_id=path[:-12], threshold=0.1, label=1)
        print("False Positives: done !!!")
        __analyse_model_predictions__(task_id=path[:-12], threshold=0.7, label=0)
        print("False negatives: done !!!")
        i += 1
    assert i == len(items)


def __analyse_model_predictions__(task_id, threshold=0.1, label=1):
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    pref = os.path.basename(task_id)
    y_pred = load(open(task_id + "-preds.pkl", "rb"))
    y_true = load(open(task_id + "-targets.pkl", "rb"))
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
        neg_bk = set(neg_bk)  # all the drugs pai"/rs that really exists
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


def __get_dataset_stats__(task_id):
    test_stats = []
    cach_path = os.getenv("INVIVO_CACHE_ROOT", "/home/rogia/.invivo/cache")
    config = json.load(open(task_id, "rb"))
    print(config)
    dataset_params = {param.split(".")[-1]: val for param, val in config.items() if param.startswith("dataset_params")}
    dataset_params = {key: val for key, val in dataset_params.items() if
                      key in get_data_partitions.__code__.co_varnames}
    dataset_name = dataset_params["dataset_name"]
    split_mode = dataset_params["split_mode"]
    seed = dataset_params["seed"]
    input_path = f"{cach_path}/datasets-ressources/DDI/{dataset_name}"
    dataset_params["input_path"] = input_path
    train, _, test1 = get_data_partitions(**dataset_params) #test2

    train_samples, test1_samples = train.samples, test1.samples#, test2.samples
    test2_samples = []
    test = [(d1, d2) for (d1, d2, _) in test1_samples]
    print("Number of drugs pairs - test 1 ", len(test))
    train_drugs = set([d1 for (d1, _, _) in train_samples] + [d2 for (_, d2, _) in train_samples])
    test1_drugs = set([d1 for (d1, _, _) in test1_samples] + [d2 for (_, d2, _) in test1_samples])
    print("Number of train drugs", len(set(train_drugs)))
    print("Number of test set 1 drugs ", len(set(test1_drugs)))
    i, j, k = __check_if_overlaps(train_drugs, test)

    if split_mode == "leave_drugs_out":
        test_stats.append((dataset_name, seed, "early split-test set 1", len(set(test)), i, j, k))
        test = [(d1, d2) for (d1, d2, _) in test2_samples]
        test2_drugs = set([d1 for (d1, _, _) in test2_samples] + [d2 for (_, d2, _) in test2_samples])
        print("Number of drugs pairs - test 2 ", len(test))
        print("Number of test set 2 drugs", len(set(test2_drugs)))
        assert len(set(train_drugs).intersection(test2_drugs)) == 0
        i, j, k = __check_if_overlaps(train_drugs, test)
        test_stats.append((dataset_name, seed, "early split-test set 2", len(set(test)), i, j, k))
    else:
        test_stats.append((dataset_name, seed, "Random ", len(set(test)), i, j, k))

    df = pd.DataFrame(test_stats,
                      columns=["dataset_name", "seed", "splitting scheme", "drugs pairs in test set",
                               "Both of the drugs are seen",
                               "One of the drug is seen",
                               "None of the drugs were seen "])
    return df


def get_dataset_stats(task_ids=None, exp_path=None, split=True):
    if not task_ids:
        assert exp_path is not None
        task_ids = [fp for fp in glob.glob(f"{exp_path}/*.json")]
    print(task_ids)
    df = pd.concat(list(map(__get_dataset_stats__, task_ids)), ignore_index=True)
    df.to_csv("../../results/dataset_params_stats.csv")
    print(df)
    overall_summary = df.groupby(["dataset_name", "splitting scheme"]).mean().round()
    overall_summary.to_csv(f"../../results/all-modes_dataset_params_stats.csv")
    if split:
        modes = df["splitting scheme"].unique()
        df["splitting scheme"].fillna(inplace=True, value="null")
        for mod in modes:
            res = df[df["splitting scheme"] == mod]
            res.to_csv(f"../../results/{mod}_dataset_params_stats.csv")


def get_misclassified_labels(path, label=0, items=10):
    i = 1
    out = []
    h = []
    c = ['road traffic accident', 'drug withdrawal', 'adverse drug effect', 'herpes simplex', 'hepatitis c',
         'hepatitis a', 'diabetes', 'flu', 'hiv disease']
    for fp in glob.glob(f"{path}/*_{label}_sis.csv"):
        # Ranked misclassified side effects
        print(fp)
        data = pd.read_csv(fp, index_col=0)
        # data = data.nsmallest(10, 'probability')
        e = data["side effect"].value_counts(normalize=True) * 100
        print(e)
        f = data["side effect"].values
        # print(f)
        # e = e.round(2).nlargest(items)
        # Most commons pairs misclassified
        l = set([tuple(i[:3]) for i in data.values])
        # print(l)
        h.append(l)
        i += 1
        out.extend(f)
    g = dict(Counter(out))
    g = sorted(g.items(), key=operator.itemgetter(1))
    d = sum([i[-1] for i in g])
    print("d", d)
    print("g", len(g))
    # k = {e: (dict(Counter(out))[e]/d) * 100 for e in c}
    print(g)
    # print(sum([j for i, j in k.items()]))
    print(h[1].intersection(*h[2:]))


# many files path - one per model
# stats about recurrences
# False positive vs False negatives

def get_best_hp():
    dt = pd.read_csv("../../results/all_raw-exp-res.csv", index_col=0)
    d = dt.groupby(["dataset_name", "model_name", "split_mode"], as_index=False)["micro_auprc"].max()
    print(list(d))
    c = []
    for index, row in d.iterrows():
        c.append(dt.loc[(dt["dataset_name"] == row["dataset_name"]) & (dt["model_name"] == row["model_name"]) & (
                dt["split_mode"] == row["split_mode"]) & (dt["micro_auprc"] == row["micro_auprc"])])
    f = pd.concat(c)
    print(f)
    f.to_csv("../../results/temp.csv")


if __name__ == '__main__':
    df = __get_dataset_stats__("/home/rogia/.invivo/result/deepddi/twosides_deepddi_4ebbb144_params.json")
    print(df)
    exit()
    # summarize_experiments("/media/rogia/CLé USB/expts/")
    # visualize_test_perf()
    # analyze_models_predictions()
    get_misclassified_labels(path="../../results/", label=0)
    exit()
    # bt = pd.read_csv("../../results/best_hp.csv")
    # # c = bt[bt["model_name"] == 'CNN'][['dataset_name', 'task_id', 'split_mode', 'path']].values
    # # print(c)
    # c = bt[(bt["dataset_name"] == 'twosides') & (bt["split_mode"] == "random")][
    #     ['dataset_name', "model_name", 'task_id', 'split_mode', 'path']].values
    analyze_models_predictions([c[::-1][1]])

    # analysis_test_perf(c)
    exit()
    get_dataset_stats(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json")
    exit()
    # get_misclassified_labels(
    #     "/home/rogia/Documents/projects/drug_drug_interactions/results/twosides_bmnddi_6936e1f9_1_sis.csv", items=10)
    # exit()
    # combine(exp_path="/media/rogia/CLé USB/expts/DeepDDI", split=False)
    exit()
    # analyse_model_predictions(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", label=1, threshold=0.1)
    # analyse_model_predictions(task_id="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", label=0, threshold=0.7)
    # choose o.1 as threshold for false positive
    # combine(["/home/rogia/Documents/projects/twosides_bmnddi_390811c9",
    #          "/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9"])

    exit()
    exp_folder = "/home/rogia/Documents/exps_results/IMB"  # /media/rogia/CLé USB/expts"  # f"{os.path.expanduser('~')}/expts"
    # summarize(exp_folder)
    # Scorer twosides random split
    scorer("twosides", task_path="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", save="random")
    # Scorer twosides early split stage
    scorer("twosides", task_path="/home/rogia/Documents/projects/twosides_bmnddi_390811c9", save="early+stage")
    exit()
    scorer("twosides", task_path="/home/rogia/Documents/exps_results/imb/twosides_bmnddi_48d052b6", save="cnn+NS")
    scorer("twosides", task_path="/home/rogia/Documents/exps_results/imb/twosides_bmnddi_5cbedb4a", save="cnn+CS")

    # Scorer drugbank random split
    scorer("drugbank", task_path="/media/rogia/CLé USB/expts/CNN/drugbank_bmnddi_54c9f5bd", save="drugbank")
    # Scorer drugbank early split
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
