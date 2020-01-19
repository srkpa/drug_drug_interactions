from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from side_effects.data.loader import load_ddis_combinations


def plot_distribution(dist, title="", x_label="", y_label="", file_name=None):
    plt.figure(figsize=(8, 6))
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    sns.distplot(dist, kde=False, color=sns.xkcd_rgb['denim blue'], bins=20, hist_kws={"alpha": 1})
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


def visualize_data(dataset_name, filepath="/home/rogia/.invivo/cache/datasets-ressources/DDI"):
    filepath += f"/{dataset_name}/{dataset_name}.csv"
    data = load_ddis_combinations(filepath, header=True, dataset_name=dataset_name)
    distribution_combos = [len(data[combo]) for combo in data]
    nb_drugs, combo_counter = get_se_counter(data)
    filename = f"/home/rogia/Bureau/figs/{dataset_name}_number_of_side_effects_per_drug_pairs_hist.png"
    plot_distribution(distribution_combos, "Number of side effects per drug combination", "Number of Side Effects",
                      "Number of \n Drug Combinations", file_name=filename)
    filename = f"/home/rogia/Bureau/figs/{dataset_name}_number_of_drugs_pairs_per_side_effects_hist.png"
    plot_distribution(np.asarray(sorted(list(zip(*combo_counter.items()))[1])), "Side Effect Frequency",
                      "Number of Drug Combinations", "Number of Side Effects", file_name=filename)

    return dataset_name, len(data), len(combo_counter), nb_drugs, np.mean(distribution_combos)


if __name__ == '__main__':
    res = list(map(visualize_data, ["twosides", "drugbank"]))
    df = pd.DataFrame(res,
                      columns=["dataset", "number of drug combinations", "number of side effects", "number of drugs",
                               "Median number of side effects per drug combination"])
    df.to_csv("/home/rogia/Bureau/figs/datasets_description.csv")
