import json
import os
from collections import Counter
from collections import defaultdict
from itertools import chain

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import jaccard_score
from tqdm import tqdm

from side_effects.data.loader import load_ddis_combinations, load_side_effect_mapping, relabel
from side_effects.utility.exp_utils import visualize_test_perf, __collect_data__, get_similarity, wrapped_partial

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20})
csfont = {'fontname': 'FreeSerif', 'size': 20}


# rc('text', usetex=True)


def plot_distribution(dist, title="", x_label="", y_label="", file_name=None):
    plt.figure(figsize=(8, 6))
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    sns.distplot(dist, kde=False, color=sns.xkcd_rgb['denim blue'], hist_kws={"alpha": 1})
    # sns.kdeplot(dist, shade=True, color=sns.xkcd_rgb['denim blue'])
    plt.xlabel(x_label)
    plt.title(title)
    # plt.gcf().subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.ylabel(y_label)
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_barplot(dist, title="", x_label="", y_label="", file_name=None):
    plt.figure(figsize=(8, 6))
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    sns.distplot(dist, kde=False, color=sns.xkcd_rgb['denim blue'], bins=20, hist_kws={"alpha": 1}, norm_hist=False)
    plt.xlabel(x_label)
    plt.title(title)
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.ylabel(y_label)
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()


def get_se_counter(se_map):
    side_effects = []
    drugs = []
    for drug in se_map:
        side_effects += list(set(se_map[drug]))
        drugs.extend(list(drug))
    return len(set(drugs)), Counter(side_effects)


def plot_percentage_hist(data, title="", x_label="", y_label="", file_name=None):
    fig, ax = plt.subplots()
    ax.hist(data, bins=20, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(data)))
    # ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(data)))
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_data_distribution(filepath="/home/rogia/.invivo/cache/datasets-ressources/DDI"):
    def data_info(dataset_name, filepath="/home/rogia/.invivo/cache/datasets-ressources/DDI", smothing=False, syn=True):
        filepath += f"/{dataset_name}/{dataset_name}.csv"
        data = load_ddis_combinations(filepath, header=True, dataset_name=dataset_name)
        if smothing:
            filepath = os.path.dirname(filepath)
            dataset_name += "-SOC"
            side_effects_mapping = load_side_effect_mapping(filepath, 'SOC')
            data = relabel(data, side_effects_mapping)
        elif not syn:
            filepath = os.path.dirname(filepath)
            output = defaultdict(set)
            with open(f"{filepath}/twosides_umls_fin.json", "r") as cf:
                cls = json.load(cf)
                for par, chld in cls.items():
                    for ld in chld:
                        output[ld].add(par)
            for pair, lab in data.items():
                nw = []
                for phen in lab:
                    if phen in output:
                        nw.extend(list(output[phen]))
                    else:
                        nw += [phen]
                data[pair] = nw
        nb_drugs, combo_counter = get_se_counter(data)
        per_combo_counter = {side_effect: round(100 * (count / len(data))) for side_effect, count in
                             combo_counter.items()}
        distribution_combos = [len(data[combo]) for combo in data]
        ld = sum([x / len(per_combo_counter) for x in distribution_combos]) / len(distribution_combos)
        print(" m = ", len(data), "\n", "|y|= ", len(combo_counter), "\n", "Nb of uniq drugs = ", nb_drugs, "\n",
              "Median of |y| = ", np.mean(distribution_combos), "\n", f"LD = {ld}")
        return per_combo_counter, dataset_name, len(data), len(combo_counter), nb_drugs, np.mean(distribution_combos)

    out0 = data_info("twosides", smothing=False, syn=False)
    exit()
    out1 = data_info("drugbank", smothing=False)
    out2 = data_info("twosides", smothing=False)
    out3 = data_info("twosides", smothing=True)

    filename = "/home/rogia/Bureau/figs/figure_phenotype_hist.png"
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 12))
    sns.set_context("paper", font_scale=3)
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'Times New Roman'})

    ax1 = sns.distplot(list(zip(*out1[0].items()))[1], ax=ax1, kde=False, color=sns.xkcd_rgb['denim blue'],
                       hist_kws={"alpha": 1})
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif':
                       'Liberation Sans'})
    ax2 = sns.distplot(list(zip(*out2[0].items()))[1], color=sns.xkcd_rgb['denim blue'], hist_kws={"alpha": 1}, ax=ax2,
                       kde=False)
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif':
                       'Liberation Sans'})
    ax3 = sns.distplot(list(zip(*out3[0].items()))[1], color=sns.xkcd_rgb['denim blue'], hist_kws={"alpha": 1}, ax=ax3,
                       kde=False, bins=10)

    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=out1[3], decimals=0, symbol=""))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=out2[3], decimals=0, symbol=""))
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=out3[3], decimals=0, symbol=""))

    ax1.set_title("Drugbank\n", fontsize=44)
    ax2.set_title("Twosides\n", fontsize=44)
    ax3.set_title("Twosides-SOC\n", fontsize=44)
    ax1.set_xlabel("")
    ax2.set_xlabel("\nPhenotype Frequency (%)", fontsize=44)
    ax3.set_xlabel("")
    ax1.set_ylabel("Percentage of phenotype (%)\n", fontsize=44)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)

    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    if filename:
        plt.tight_layout()
        plt.savefig(filename)
    else:
        plt.show()


def plot_results(params, title="", file_name="", leg=True, smothing=False):
    result_file, dataset = params
    file_name = f"/home/rogia/Bureau/figs/figure_all_{dataset}.png"
    dist = pd.read_csv(result_file, index_col=0)
    print(list(dist))
    dist = dist[dist["dataset"] == dataset]
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 15))
    sns.set_context("paper", font_scale=5)
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif':
                       'Liberation Sans'})
    ax1 = sns.barplot(x='Evaluation schemes', y='micro_roc', hue='Model name', data=dist,
                      palette=sns.color_palette("Paired"),
                      capsize=0.05,
                      saturation=8,
                      order=['Random', 'One-unseen', 'Both-unseen'],
                      errcolor='gray', errwidth=2,
                      ci='sd', ax=axs[0]
                      )

    ax1.set_ylabel("AUROC", fontsize=40)
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif':
                       'Liberation Sans'})
    ax2 = sns.barplot(x='Evaluation schemes', y='micro_auprc', hue='Model name', data=dist,
                      palette=sns.color_palette("Paired"),
                      capsize=0.05,
                      saturation=8,
                      errcolor='gray', errwidth=2,
                      order=['Random', 'One-unseen', 'Both-unseen'],
                      ci='sd', ax=axs[1]
                      )
    ax2.get_legend().remove()
    if dataset == "drugbank":
        ax1.set_ylim(0.90, 1)
    else:
        if smothing:
            ax1.set_ylim(0.75, 1)
            ax2.set_ylim(0.75, 1)
        else:
            ax1.set_ylim(0.7, 1)
    if not leg:
        ax1.get_legend().remove()
    else:
        plt.setp(ax1.get_legend().get_texts(), fontsize=35)
        plt.setp(ax1.get_legend().get_title(), fontsize=35)
    ax2.set_ylabel("AUPRC", fontsize=40)
    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position("right")
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(38)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(38)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    plt.title(title)
    plt.tight_layout()
    # plt.gcf().subplots_adjust(left=0.2, right=0.3, top=0.3, bottom=0.2)
    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)
    else:
        plt.show()


def plot_test_dens(fp, leg=False):
    sns.set_context("paper", font_scale=2.5)
    sns.set_style("ticks")
    output = visualize_test_perf(fp, eval_on_train=False, eval_on_test=True)
    for dataset, res_dict in output.items():
        file_name = f"/home/rogia/Bureau/figs/figure_drug_pair_n_positive_{dataset}.png"
        out = []
        density = {}
        for split, sp_dict in res_dict.items():
            df = pd.DataFrame()
            dens, ya, yr = sp_dict['dens'], sp_dict['ap'], sp_dict['roc']
            density[split] = dens
            print("Max dens", max(dens))
            print(len(dens), len(ya), len(yr))
            df.loc[:, "split"] = [split] * (len(ya) + len(yr))
            df.loc[:, "Metric name"] = ["AUPRC"] * len(ya) + ["AUROC"] * len(
                yr)
            df.loc[:, "Scores"] = ya + yr
            df.loc[:, "Density"] = dens * 2
            out += [df]
        df = pd.concat(out, ignore_index=True)
        k = list(set(sorted(density['random'])))
        tck = [k[0], k[len(k) // 2], k[-1]]
        tck = [0.0] + tck[0:2] if tck[1] == tck[2] else tck
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))
        ax1 = sns.boxplot(x='Density', y='Scores', hue='Metric name', data=df[df['split'] == 'random'],
                          ax=ax1
                          )
        ax1.xaxis.set_major_locator(plt.LinearLocator(3))
        ax1.set_xticklabels(tck)
        k = list(set(sorted(density['leave_drugs_out (test_set 1)'])))
        tck = [k[0], k[len(k) // 2], k[-1]]
        tck = [0.0] + tck[0:2] if tck[1] == tck[2] else tck
        ax2 = sns.boxplot(x='Density', y='Scores', hue='Metric name',
                          data=df[df['split'] == 'leave_drugs_out (test_set 1)'],
                          ax=ax2
                          )
        ax2.xaxis.set_major_locator(plt.LinearLocator(3))
        ax2.set_xticklabels(tck)
        k = list(set(sorted(density['leave_drugs_out (test_set 2)'])))
        tck = [k[0], k[len(k) // 2], k[-1]]
        tck = [0.0] + tck[0:2] if tck[1] == tck[2] else tck
        ax3 = sns.boxplot(x='Density', y='Scores', hue='Metric name',
                          data=df[df['split'] == 'leave_drugs_out (test_set 2)'],
                          ax=ax3
                          )
        ax3.xaxis.set_major_locator(plt.LinearLocator(3))
        ax3.set_xticklabels(tck)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        for tick in ax3.yaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        for tick in ax3.xaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        ax2.get_legend().remove()  # '#fb9a99' rose
        ax3.get_legend().remove()
        if leg:
            ax1.get_legend().remove()
        ax1.set_title("Random", fontsize=35)
        ax2.set_title("One-unseen", fontsize=35)
        ax3.set_title("Both-unseen", fontsize=35)
        ax2.set_ylabel("Scores", fontsize=35)
        ax3.set_ylabel("Scores", fontsize=35)
        ax1.set_ylabel("Scores", fontsize=35)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.show()


def plot_test_perf(fp, leg=False, met1="ap", met2="f1"):
    sns.set_context("paper", font_scale=2.5)
    sns.set_style("ticks")

    def define_categories(dataset, res_dict, n_samples=63472):
        ticks = {}
        x = res_dict['random']['x']
        out = []
        y = []
        max_freq = max(x)
        freq = list(range(0, max_freq + int(max_freq / 3), int(max_freq / 3)))
        freq = [int((count * 100) / n_samples) for count in freq]
        if dataset == "drugbank":
            freq = [0, 3, 5, 7] + freq[1:]
        print(freq)
        i = 0
        for i, j in enumerate(x):
            j = int((j / n_samples) * 100)
            if freq[0] <= j < freq[1]:
                y += [1]
                ticks[1] = f'{str(freq[0])},{str(freq[1])}['
            elif freq[1] <= j < freq[2]:
                y += [2]
                ticks[2] = f'{str(freq[1])},{str(freq[2])}['
            elif freq[2] <= j < freq[3]:
                y += [3]
                ticks[3] = f'{str(freq[2])},{str(freq[3])}['  # finit ici
            elif freq[3] <= j < freq[4]:
                y += [4]
                ticks[4] = f'{str(freq[3])},{str(freq[4])}['  # {str(freq[4])
            elif freq[4] <= j < freq[5]:
                y += [5]
                ticks[5] = f'{str(freq[4])}, {str(freq[5])}['
            elif freq[5] <= j < freq[6]:
                y += [6]
                ticks[6] = f'{str(freq[5])}, {str(freq[6])}['
            elif freq[6] <= j < freq[7]:
                y += [7]
                ticks[7] = f'{str(freq[6])},{str(freq[7])}['
            else:
                y += [8]
                ticks[8] = f'{str(freq[7])}, 100]'
        for split, rdict in res_dict.items():
            df = pd.DataFrame()
            print(len(x), len(y), len(rdict['roc']), len(rdict['ap']), len(rdict['f1']))
            df.loc[:, "split"] = [split] * len(y) * 2
            df.loc[:, "Frequence (%)"] = y * 2
            df.loc[:, "Metric name"] = ["AUPRC"] * len(y) + ["F1-score"] * len(
                y) if met1 == "ap" and met2 == "f1" else ["AUPRC"] * len(y) + ["AUROC"] * len(y)
            df.loc[:, "Scores"] = rdict[met1] + rdict[met2]  # rdict['ap'] + rdict['f1']
            out += [df]
        result = pd.concat(out, ignore_index=True)
        return ticks, result

    output = visualize_test_perf(fp, eval_on_train=True)
    output = {"twosides": output["twosides"]}
    for dataset, res_dict in output.items():
        file_name = f"/home/rogia/Bureau/figs/figure_phenotype_freq_{dataset}.png"
        n_samples = 63472 if dataset == "twosides" else 191878
        print(dataset, res_dict.keys())
        tks, dist = define_categories(dataset, res_dict, n_samples)
        tks = sorted(list(tks.items()))
        print(dist)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(25, 8))
        dist_data = dist[dist['split'] == 'random']
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                       'font.sans-serif':
                           'Liberation Sans'})
        ax1 = sns.boxplot(x='Frequence (%)', y='Scores', hue='Metric name', data=dist_data,
                          ax=ax1
                          )  # palette=['dodgerblue', 'lime'],
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                       'font.sans-serif':
                           'Liberation Sans'})
        dist_data = dist[dist['split'] == 'leave_drugs_out (test_set 1)']
        ax2 = sns.boxplot(x='Frequence (%)', y='Scores', hue='Metric name', data=dist_data
                          , ax=ax2
                          )  # palette=['dodgerblue', 'lime']
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                       'font.sans-serif':
                           'Liberation Sans'})
        dist_data = dist[dist['split'] == 'leave_drugs_out (test_set 2)']
        ax3 = sns.boxplot(x='Frequence (%)', y='Scores', hue='Metric name', data=dist_data, ax=ax3,
                          )
        ax2.get_legend().remove()  # '#fb9a99' rose
        ax3.get_legend().remove()
        leg = True
        if leg:
            ax1.get_legend().remove()
        ax1.set_title("Random", fontsize=35)
        ax2.set_title("One-unseen", fontsize=35)
        ax3.set_title("Both-unseen", fontsize=35)
        ax1.set_xlabel("")
        ax3.set_xlabel("")
        ax2.set_xlabel("")
        # ax2.set_ylabel(f"{dataset}\nScores", fontsize=35)
        # ax3.set_ylabel(f"{dataset}\nScores", fontsize=35)
        ax1.set_ylabel(f"{dataset}-SOC\nScores", fontsize=35)
        # ax2.set_xlabel("\nPhenotype Frequency (%)", fontsize=35) ###
        ax3.set_ylabel("")
        ax2.set_ylabel("")
        # ax3.set_ylabel("")
        ax1.set_xticklabels(list(zip(*tks))[1], fontsize=33)
        ax2.set_xticklabels(list(zip(*tks))[1], fontsize=33)
        ax3.set_xticklabels(list(zip(*tks))[1], fontsize=33)
        # ax2.set_yticklabels([])
        # ax3.set_yticklabels([])
        ax1.set_ylim(0.0, 1.1)
        ax2.set_ylim(0.0, 1.1)
        ax3.set_ylim(0.0, 1.1)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        for tick in ax3.yaxis.get_major_ticks():
            tick.label.set_fontsize(35)
        # plt.setp(ax1.get_legend().get_texts(), fontsize=30)
        # plt.setp(ax1.get_legend().get_title(), fontsize=30)
        plt.tight_layout()
        # plt.show()
        plt.savefig(file_name)
        exit()


def get_correlation(config_file="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json"):
    outfile = f"/home/rogia/Bureau/figs/jaccard_scores.npy"
    train, valid, test1, test2 = __collect_data__(config_file)
    mbl = train.labels_vectorizer
    samples = train.samples + valid.samples + test1.samples
    target = mbl.transform(list(zip(*samples))[2]).astype(np.float32)
    n_samples, n_side_effect = target.shape
    print(n_samples, n_side_effect)

    def __jaccard_score(pos):
        side_effect = np.repeat(a=np.expand_dims(target[:, pos], axis=1), repeats=n_side_effect, axis=1)
        out = jaccard_score(y_true=target, y_pred=side_effect, average=None)
        return out

    output = list(map(__jaccard_score, tqdm(list(range(n_side_effect)))))
    jc_mat = np.array(output)
    print(jc_mat.shape)
    np.save(outfile, jc_mat)


def plot_corr():
    sns.set(font_scale=1.5)
    filename = "/home/rogia/Bureau/figs/"
    z = np.load(filename + "jaccard_scores.npy")
    print(z.max(), z.min(), np.mean(z), np.median(z), np.percentile(z, 25), np.percentile(z, 75))
    exit()
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(z, cmap="Greens", ax=ax)
    plt.savefig(filename + "corr.png")
    plt.show()


# true positives = goood predictions = 1 and 1
# false positives = wrong pairs but true = 0 and 1
# vs train

def plot_sim(true_positives, false_negatives, true_negatives, falses_positives, saved_file, ref=None, debug=False,
             prob=False):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(18, 18))
    ((ax1, ax2), (ax3, ax4)) = axes
    # gridspec_kw={'hspace': 0, 'wspace': 0})
    sns.set_context("paper", font_scale=3)
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif':
                       'Liberation Sans'})
    df = pd.DataFrame()

    def _parse_data(filepath, prob=False):
        drug_pairs = []
        if isinstance(filepath, str) and os.path.isfile(filepath):
            data = pd.read_csv(filepath, sep=",", index_col=0)
            if prob:
                return data["probability"].values
            data = data[["drug_1", "drug_2", "side effect", "probability"]].groupby(["side effect"],
                                                                                    as_index=False).max()
            instances = [tuple(row[:3]) for row in data.values]
            drug_pairs = [(did1, did2, se) for (se, did1, did2) in instances if
                          se in ["heart attack", "panic attack", "muscle spasm", "blood sodium decreased"]]
            drug_pairs.sort(key=lambda tup: tup[2])
        return drug_pairs

    tp, fn, tn, fp = list(map(wrapped_partial(_parse_data, prob=prob),
                              [true_positives, false_negatives, true_negatives, falses_positives]))
    if prob:
        plt.figure(figsize=(8, 6))
        sns.distplot(tp, color=sns.xkcd_rgb['denim blue'], hist_kws={"alpha": 1}, kde=False, norm_hist=True, label="TP")
        sns.distplot(tn, color=sns.xkcd_rgb['pastel orange'], hist_kws={"alpha": 1}, kde=False, norm_hist=True,
                     label="TN")
        sns.distplot(fp, color=sns.xkcd_rgb['light forest green'], hist_kws={"alpha": 1}, norm_hist=True, kde=False,
                     label="potential DDI (FP)")
        sns.distplot(fn, color=sns.xkcd_rgb['cherry'], hist_kws={"alpha": 1}, norm_hist=True, kde=False,
                     label="mislabeled DDI (FN)")

        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"/home/rogia/Bureau/figs/{saved_file}_hist.png")
        return True
    if debug:
        tp, fn, fp, tn = tp[:1], fn[:1], fp[:1], tn[:1]
    tp_sim = list(map(wrapped_partial(get_similarity, ref=ref, label=1), tp))
    fn_sim = list(map(wrapped_partial(get_similarity, ref=ref, label=1), fn))
    fp_sim = list(map(wrapped_partial(get_similarity, ref=ref, label=1), fp))  # 0
    tn_sim = list(map(wrapped_partial(get_similarity, ref=ref, label=1), tn))  # 0

    output = tp_sim + tn_sim + fp_sim + fn_sim
    df["Similarity"] = list(chain.from_iterable(output))
    df["predictions"] = list(chain.from_iterable(
        [["TP"] * len(scores) for i, scores in enumerate(tp_sim)] + [["TN"] * len(scores) for
                                                                     i, scores in
                                                                     enumerate(tn_sim)] + [
            ["PT"] * len(scores) for i, scores in enumerate(fp_sim)] + [["MS"] * len(scores)
                                                                        for i, scores in
                                                                        enumerate(fn_sim)]))
    df["side_effect"] = list(chain.from_iterable(
        [[tp[i][-1]] * len(scores) for i, scores in enumerate(tp_sim)] + [[tn[i][-1]] * len(scores) for i, scores in
                                                                          enumerate(tn_sim)] + [
            [fp[i][-1]] * len(scores) for i, scores in enumerate(fp_sim)] + [[fn[i][-1]] * len(scores) for i, scores in
                                                                             enumerate(fn_sim)]))
    # g = sns.FacetGrid(df, col="triplet", hue="predictions", col_wrap=2, sharey=False, margin_titles=True,
    #                   legend_out=False)
    # g = (g.map(sns.kdeplot, "Similarity", shade=False).add_legend())
    # g.set(xlim=(0, 1.))
    # g.set_titles(col_template='{col_name}')
    axs = [ax1, ax2, ax3, ax4]
    for i, side_effect in enumerate(df["side_effect"].unique()):
        dist = df[df["side_effect"] == side_effect]
        trp = dist[df["predictions"] == "TP"][["Similarity"]]
        trn = dist[df["predictions"] == "TN"][["Similarity"]]
        fap = dist[df["predictions"] == "PT"][["Similarity"]]
        fan = dist[df["predictions"] == "MS"][["Similarity"]]
        sns.distplot(trp.values, color="blue", hist=False, rug=False, label="TP", ax=axs[i])
        sns.distplot(fap.values, color="green", hist=False, rug=False, label="potential DDI (FP)", ax=axs[i])
        sns.distplot(trn.values, color="orange", hist=False, rug=False, label="TN", ax=axs[i])
        sns.distplot(fan.values, color="red", hist=False, rug=False, label="mislabeled DDI (FN)", ax=axs[i])
        axs[i].set_title(side_effect + "\n", fontsize=44, fontweight='bold')
        axs[i].set_xlabel("Average Similarity", fontsize=44)
        axs[i].set_ylabel("Relative Density", fontsize=44)
        axs[i].set_yticklabels([])
        axs[i].set_xlim(0.0, 1.)
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(40)
    plt.legend(prop={'size': 20})
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()
    plt.setp(ax1.get_legend().get_texts(), fontsize=30)
    plt.setp(ax1.get_legend().get_title(), fontsize=30)

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)

    # Get the minimum and maximum extent, get the coordin ate half-way between those
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

    plt.savefig(f"/home/rogia/Bureau/figs/{saved_file}.png")


def compute_phenotype_cooccurence(cluster_file="/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/twosides_umls_fin.json",
                                  config_file="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_params.json"):
    train, valid, test1, test2 = __collect_data__(config_file)
    mbl = train.labels_vectorizer
    side_effects = list(mbl.classes_)

    # Load side effect cluster file
    with open(cluster_file, "r") as cf:
        cls = json.load(cf)
        print(len(cls), len(set(chain.from_iterable(list(cls.values())))))

    i = 0
    y = set(chain.from_iterable(list(cls.values())))
    for se in side_effects:
        if se not in cls and se not in y:
            print(se)
            i += 1

    exit()
    print(i, len(y), len(cls), i + len(y) + len(cls))
    cach_path = "/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/"
    data = load_ddis_combinations(cach_path + "twosides.csv", True, "twosides")
    samples = list(data.values())
    target = mbl.transform(samples).astype(np.float32)
    n_samples, n_side_effect = target.shape
    assert n_samples == len(samples)
    assert n_side_effect == len(set(chain.from_iterable(samples)))

    def __jaccard_score(x, y, target):
        # print(x, len(y), target.shape)
        n = len(y)
        side_effect = np.repeat(a=np.expand_dims(target[:, x], axis=1), repeats=n, axis=1)
        target = target[:, y]
        # print(side_effect.shape, target.shape)
        out = jaccard_score(y_true=target, y_pred=side_effect, average=None)
        return out

    output = list(chain.from_iterable(
        list(map(wrapped_partial(__jaccard_score, target=target), tqdm([side_effects.index(label) for label in cls]),
                 tqdm([[side_effects.index(l) for l in cls[label]] for label in cls])))))
    print(len(output))
    plot_distribution(output, title="Jaccard correlation among synonym", x_label="Jaccard Score",
                      y_label="Phenotypes (Count)",
                      file_name="/home/rogia/Bureau/figs/figure_jaccard_corr_among_synonym_twosides.png")

    # target = load(open("/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9_preds.pkl", "rb"))
    # print(target)
    # #target = target < 0.5
    # target = target >= 0.7
    # print(target.astype(int))
    # output = list(chain.from_iterable(
    #     list(map(wrapped_partial(__jaccard_score, target=target), tqdm([side_effects.index(label) for label in cls]),
    #              tqdm([[side_effects.index(l) for l in cls[label]] for label in cls])))))
    # print(len(output))
    # plot_distribution(output, title="Jaccard correlation models preditions (threshold = 0.5)", x_label="Jaccard Score",
    #                   y_label="Phenotypes (Count)",
    #                   file_name="")  #/home/rogia/Bureau/figs/figure_jaccard_corr_among_synonym_predicted_vec_twosides.png


# target = target

# def plot_data_stats(filepath="/home/rogia/Documents/exps_results/temp/dataset_params_stats.csv"):
#     data = pd.read_csv(filepath, index_col=0)
#     groups = data.groupby(["dataset_name", "splitting scheme"])
#     mean = groups.mean()
#     std = groups.std()
#     ind = np.arange(3)
#     width = 0.35
#     mean = mean.loc["drugbank"]
#     std = std.loc["drugbank"]
#     x =  mean["None of the drugs were seen "].values.tolist()
#     y = mean["Both of the drugs are seen"].values.tolist()
#     z = mean["One of the drug is seen"].values.tolist()
#     p1 = plt.bar(ind, mean["None of the drugs were seen "].values.tolist(), width, yerr=std["None of the drugs were seen "].values.tolist())
#     p2 = plt.bar(ind, mean["Both of the drugs are seen"].values.tolist(), width, bottom=mean["None of the drugs were seen "].values.tolist(), yerr=std["Both of the drugs are seen"].values.tolist())
#     p3 = plt.bar(ind, mean["One of the drug is seen"].values.tolist(), width,
#                  bottom=mean["Both of the drugs are seen"].values.tolist(), yerr=std["One of the drug is seen"].values.tolist())
#
#     plt.legend((p1[0], p2[0], p3[0]), ('None', 'Both', 'One'))
#     plt.xticks(ind, ('Random', 'One-unsen', 'Both-unseen'))
#     plt.show()

if __name__ == '__main__':
    #plot_data_stats()
   # exit()
    # plot_data_distribution()
    # visualize_loss_progress("/media/rogia/5123-CDC3/SOC/cl_deepddi/twosides_deepddi_1d1fb1d1_log.log")
    # exit()
    # Step 1
    # plot_data_distribution#()
    # output = []
    # res = plot_data_distribution(dataset_name="twosides", smothing=True)
    # output += [res]
    # res = plot_data_distribution(dataset_name="twosides", smothing=False)
    # output += [res]
    # res = plot_data_distribution(dataset_name="drugbank", smothing=False)
    # output += [res]
    # exit()
    # print(output)
    # df = pd.DataFrame(output,
    #                   columns=["dataset", "number of drug combinations", "number of side effects", "number of drugs",
    #                            "Median number of side effects per drug combination"])
    # df.to_csv("/home/rogia/Bureau/figs/datasets_description.csv")
    # exit()
    # Step 2
    # plot_results(["/home/rogia/Téléchargements/table-1 - table-1 - all_raw-exp-res.csv", "drugbank"], leg=False)
    # plot_results(["/home/rogia/Téléchargements/table-1 - table-1 - all_raw-exp-res.csv", "twosides"], leg=True)
    # plot_results(["/home/rogia/Bureau/figs/soc.csv", "twosides"], leg=False, smothing=True)

    # plot_test_perf("/home/rogia/Documents/exps_results/temp-2/temp.csv", met1="ap", met2="f1")
    # plot_test_perf("/home/rogia/Documents/exps_results/temp-2/temp.csv", met1="ap", met2="roc")

   # plot_test_perf("/home/rogia/Documents/exps_results/temp-SOC/temp.csv", met1="ap", met2="roc")
    # plot_test_dens("/home/rogia/Documents/exps_results/temp-2/temp.csv")
    # plot_test_dens("/home/rogia/Documents/exps_results/temp-SOC/temp.csv")

    # Step 3
    # res = scorer("twosides", task_path="/media/rogia/CLé USB/expts/CNN/twosides_bmnddi_6936e1f9", split="random",
    #              req=["muscle spasm", "drug withdrawal"]) #'hepatitis c', 'hepatitis a', 'hiv disease', 'hiv disease', 'flu', 'road traffic accident',
    #                   #'herpes simplex', 'adverse drug effect'
    # scores = []
    # for i, j in res.items():
    #     k = (i, *j)
    #     scores.append(k)
    #     print(k)
    # df = pd.DataFrame(scores,
    #                   columns=["se", "AUPRC", "ROC-AUC", "F1-scores",
    #                            "Frequency(%)"])
    # exit()
    # df.to_csv("/home/rogia/Bureau/figs/tab_9_side_effect_description.csv")
    # Step 4
    # get_correlation()
    # Step 5
    # plot_corr()
    # tab10 = [("bupropion", "benazepril", "heart attack"),  # ("didanosine", "stavudine", "acidosis"),
    #          ("bupropion", "fluoxetine", "panic attack"), ("bupropion", "orphenadrine", "muscle spasm"),
    #          # ("tramadol", "zolpidem", "diaphragmatic hernia"),  # ("paroxetine", "fluticasone", "muscle spasm"),
    #          # ("paroxetine", "fluticasone", "fatigue"), ("fluoxetine", "methadone", "pain"),
    #          ("carboplatin", "cisplatin", "blood sodium decreased")]
    # ("chlorthalidone", "fluticasone", "high blood pressure")]

    # plot_sim(true_positi="../../results/1_as_1.csv", pot_ddis=tab10, fp="1AS1", ref=None, debug=False, legend="TP",
    #          label=1)  # doit etre similaires
    # plot_sim(true="../../results/1_as_0.csv", pot_ddis=tab10, fp="1AS0", ref=None, debug=False, legend="FN", label=1)
    # plot_sim(true="../../results/0_as_0.csv", pot_ddis=tab10, fp="0AS0", ref=None, debug=False, legend="TN",
    #          label=0)  # doit etre different
    # plot_sim(true="../../results/0_as_1.csv", pot_ddis=tab10, fp="0AS1", ref=None, debug=False, legend="FP",
    #          label=0)  # autres potentielles ddi
    #
    # plot_sim(true_positives="../../results/1_as_1.csv", true_negatives="../../results/0_as_0.csv",
    #          falses_positives="../../results/0_as_1.csv", false_negatives="../../results/1_as_0.csv",
    #          saved_file="figure_similarity_tp_fp",
    #          ref=None,
    #          debug=False, prob=False)
    # # plot_sim(true="../../results/bad_pred.csv", false=tab10, fp="", ref=None, debug=True)
    # exit()
    compute_phenotype_cooccurence()
