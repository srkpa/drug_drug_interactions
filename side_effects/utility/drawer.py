from collections import Counter
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc

from side_effects.data.loader import load_ddis_combinations, load_side_effect_mapping, relabel
from side_effects.utility.exp_utils import visualize_test_perf

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 15})


# rc('text', usetex=True)


def plot_distribution(dist, title="", x_label="", y_label="", file_name=None):
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


def visualize_data(dataset_name, filepath="/home/rogia/.invivo/cache/datasets-ressources/DDI", smothing=False):
    filepath += f"/{dataset_name}/{dataset_name}.csv"
    data = load_ddis_combinations(filepath, header=True, dataset_name=dataset_name)
    if smothing:
        filepath = os.path.dirname(filepath)
        dataset_name += "-SOC"
        side_effects_mapping = load_side_effect_mapping(filepath, 'SOC')
        data = relabel(data, side_effects_mapping)
    nb_drugs, combo_counter = get_se_counter(data)
    print(max(list(combo_counter.values())), len(data))
    #combo_counter = {side_effect: 100 * (count / len(data)) for side_effect, count in combo_counter.items()}
    distribution_combos = [len(data[combo]) for combo in data]

    # filename = f"/home/rogia/Bureau/figs/{dataset_name}_nb_side_effect_par_sample_hist.png"
    # filename = ""
    # plot_percentage_hist(distribution_combos, "",
    #                      "Percentage of side effects (%)",
    #                      "F", file_name=filename)
    # filename = f"/home/rogia/Bureau/figs/{dataset_name}_number_of_samples_per_side_effects_hist.png"

    filename = ""
    # max_freq = max(np.asarray(sorted(list(zip(*combo_counter.items()))[1])))
    # freq = list(range(0, max_freq + int(max_freq / 3), int(max_freq / 3)))
    # freq = [int((count * 100) / len(distribution_combos)) for count in freq]
    # plot_percentage_hist(np.asarray(sorted(list(zip(*combo_counter.items()))[1])),
    #                      "",
    #                      "Side effect frequency (%)", "Number of side effect", file_name=filename)

    return dataset_name, len(data), len(combo_counter), nb_drugs, np.mean(distribution_combos)


def plot_results(params, title="", file_name="", leg=True):
    result_file, dataset = params
    file_name = f"/home/rogia/Bureau/figs/{dataset}_res.png"
    dist = pd.read_csv(result_file, index_col=0)
    print(list(dist))
    dist = dist[dist["dataset"] == dataset]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 9))
    sns.set_context("paper", font_scale=2)
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif':
                       'Liberation Sans'})
    ax1 = sns.barplot(x='Evaluation schemes', y='micro_roc', hue='Model name', data=dist,
                      palette='deep',
                      capsize=0.05,
                      saturation=8,
                      order=['Random', 'One-unseen', 'Both-unseen'],
                      errcolor='gray', errwidth=2,
                      ci='sd', ax=axs[0]
                      )
    if dataset == "drugbank":
        ax1.set_ylim(0.90, 1)
    else:
        ax1.set_ylim(0.7, 1)
    ax1.get_legend().remove()
    ax1.set_ylabel("ROC-AUC")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                   'font.sans-serif':
                       'Liberation Sans'})
    ax2 = sns.barplot(x='Evaluation schemes', y='micro_auprc', hue='Model name', data=dist,
                      palette='deep',
                      capsize=0.05,
                      saturation=8,
                      errcolor='gray', errwidth=2,
                      order=['Random', 'One-unseen', 'Both-unseen'],
                      ci='sd', ax=axs[1]
                      )
    if not leg:
        ax2.get_legend().remove()
    else:
        plt.setp(ax2.get_legend().get_texts())
        plt.setp(ax2.get_legend().get_title())
    ax2.set_ylabel("AUPRC")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    plt.title(title)
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)
    else:
        plt.show()


def plot_test_perf(fp, leg=False, met1="ap", met2="f1"):
    sns.set_context("paper", font_scale=2.5)
    sns.set_style("ticks")

    def define_categories(res_dict, n_samples=63472):
        ticks = {}
        x = res_dict['random']['x']
        out = []
        y = []
        # freq = [percentile(np.array(x), q=nth) for nth in [0, 50, 75, 100]]
        # print(freq)
        # freq = [int((100 * f) / n_samples) for f in freq]
        max_freq = max(x)
        #
        freq = list(range(0, max_freq + int(max_freq / 3), int(max_freq / 3)))
        freq = [int((count * 100) / n_samples) for count in freq]
        for i, j in enumerate(x):
            j = int((j / n_samples) * 100)
            if freq[0] <= j < freq[1]:
                y += [1]
                ticks[1] = f'[{str(freq[0])},{str(freq[1])}['
            elif freq[1] <= j < freq[2]:
                y += [2]
                ticks[2] = f'[{str(freq[1])},{str(freq[2])}['
            elif freq[2] <= j < freq[3]:
                y += [3]
                ticks[3] = f'[{str(freq[2])},{str(freq[3])}['
            # elif freq[3] <= j < freq[4]:
            #     y += [4]
            #     ticks[4] = f'[{str(freq[3])}, {str(freq[4])}['
            else:
                y += [4]
                ticks[4] = f'[{str(freq[3])}, 100]'
        for split, rdict in res_dict.items():
            df = pd.DataFrame()
            print(len(x), len(y), len(rdict['roc']), len(rdict['ap']), len(rdict['f1']))
            df.loc[:, "split"] = [split] * len(y) * 2
            df.loc[:, "Frequence (%)"] = y * 2
            df.loc[:, "Metric name"] = ["AUPRC"] * len(y) + ["F1-score"] * len(
                y) if met1 == "ap" and met2 == "f1" else ["AUPRC"] * len(y) + ["ROC-AUC"] * len(y)
            df.loc[:, "Scores"] = rdict[met1] + rdict[met2]  # rdict['ap'] + rdict['f1']
            out += [df]
        result = pd.concat(out, ignore_index=True)
        return ticks, result

    output = visualize_test_perf(fp)
    for dataset, res_dict in output.items():
        file_name = f"/home/rogia/Bureau/figs/{met2}_{dataset}_imb.png"
        n_samples = 63472 if dataset == "twosides" else 191878
        print(dataset, res_dict.keys())
        tks, dist = define_categories(res_dict, n_samples)
        tks = sorted(list(tks.items()))
        print(dist)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 9))
        dist_data = dist[dist['split'] == 'random']
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                       'font.sans-serif':
                           'Liberation Sans'})
        ax1 = sns.boxplot(x='Frequence (%)', y='Scores', hue='Metric name', data=dist_data,
                          palette='deep', ax=ax1,
                          )
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                       'font.sans-serif':
                           'Liberation Sans'})
        dist_data = dist[dist['split'] == 'leave_drugs_out (test_set 1)']
        ax2 = sns.boxplot(x='Frequence (%)', y='Scores', hue='Metric name', data=dist_data,
                          palette='deep', ax=ax2,
                          )
        sns.set_style({"xtick.direction": "in", "ytick.direction": "in", 'font.family': 'sans-serif',
                       'font.sans-serif':
                           'Liberation Sans'})
        dist_data = dist[dist['split'] == 'leave_drugs_out (test_set 2)']
        ax3 = sns.boxplot(x='Frequence (%)', y='Scores', hue='Metric name', data=dist_data,
                          palette='deep', ax=ax3,
                          )
        ax1.get_legend().remove()
        ax2.get_legend().remove()
        if leg:
            ax3.get_legend().remove()
        ax1.set_title("Random")
        ax2.set_title("One-unseen")
        ax3.set_title("Both-unseen")
        ax1.set_xlabel("")
        ax2.set_xlabel("Phenotype Frequency (%)")
        ax3.set_xlabel("")
        ax2.set_ylabel("")
        ax3.set_ylabel("")
        ax1.set_xticklabels(list(zip(*tks))[1])
        ax2.set_xticklabels(list(zip(*tks))[1])
        ax3.set_xticklabels(list(zip(*tks))[1])
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax1.set_ylim(0.0, 1.1)
        ax2.set_ylim(0.0, 1.1)
        ax3.set_ylim(0.0, 1.1)
        # plt.show()
        plt.savefig(file_name)


if __name__ == '__main__':
    # Step 1
    output = []
    res = visualize_data(dataset_name="twosides", smothing=True)
    output += [res]
    res = visualize_data(dataset_name="twosides", smothing=False)
    output += [res]
    res = visualize_data(dataset_name="drugbank", smothing=False)
    output += [res]
    print(output)
    df = pd.DataFrame(output,
                      columns=["dataset", "number of drug combinations", "number of side effects", "number of drugs",
                               "Median number of side effects per drug combination"])
    df.to_csv("/home/rogia/Bureau/figs/datasets_description.csv")

    # Step 2
    # plot_results(["/home/rogia/Téléchargements/table-1 - table-1 - all_raw-exp-res.csv", "drugbank"], leg=True)
    # plot_results(["/home/rogia/Téléchargements/table-1 - table-1 - all_raw-exp-res.csv", "twosides"], leg=False)

    # plot_test_perf("/home/rogia/Documents/projects/temp-2/temp.csv", met1="ap", met2="f1")
    plot_test_perf("/home/rogia/Documents/projects/temp-2/temp.csv", met1="ap", met2="roc")
