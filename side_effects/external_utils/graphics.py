import collections
import json
import operator
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from side_effects.preprocess.dataset import *


# not good but i have to duplicate it
def save_results(filename, contents, engine='xlsxwriter'):
    writer = pd.ExcelWriter(filename, engine=engine)
    for sheet, data in contents:
        data.to_excel(writer, sheet_name=sheet)
    writer.save()


def unpack_results(outrepo):
    confs = json.load(open(os.path.join(outrepo, "configs.json"), "rb"))
    out = pickle.load(open(os.path.join(outrepo, "output.pkl"), "rb"))
    y_true = pickle.load(open(os.path.join(outrepo, "true_labels.pkl"), "rb"))
    y_probs = pickle.load(open(os.path.join(outrepo, "predicted_labels.pkl"), "rb"))
    losses = json.load(open(os.path.join(outrepo, "history.json"), "rb"))
    return confs, out, y_true, y_probs, losses


def describe_expt(input_path, output_path="/home/rogia/Documents/analysis//figures"):
    params, out, y_true, y_scores, losses = unpack_results(input_path)
    init_fn = params["init_fn"]
    loss_function = params["loss_function"]
    extractor_network = params["extractor"]
    extractor_params = params["extractor_params"]
    anchored_text = "\n".join(
        [f"{hp}:{val}" for hp, val in [("network", extractor_network), ("loss", loss_function), ("init", init_fn)]])

    results = {
        # "fc out size": extractor_params["out_size"],
        # "network": extractor_network,
        # "loss": loss_function,
        # "init_fn": init_fn,
        # "embedding ": (extractor_params["vocab_size"], extractor_params["embedding_size"]),
        # "out size": extractor_params["out_size"],
        # "kernel size": extractor_params["kernel_size"],

        "micro auprc": out["ap"]['micro'],
        "micro rocauc": out["ROC"]['micro'],
        "n features map": extractor_params["cnn_sizes"][-1]
    }
    out["ap"].pop("micro")
    out["ROC"].pop("micro")

    #   Plot losses
    plot_losses(losses, text=anchored_text, save_as=os.path.join(output_path, f"{os.path.basename(input_path)}.png"))

    return results


# Update until now


def describe_all_experiments(output_folder):
    out = []
    for c in os.listdir(output_folder):
        folder = os.path.join(output_folder, c)
        if os.path.isdir(folder):
            files = os.listdir(folder)
            if len(files) > 2:
                print(folder)
                res = describe_expt(input_path=folder)
                out.append(res)
    df = pd.DataFrame(out)

    return df


def plot_bar_x(labels, values, x_label, y_label, title, save_to, rot):
    # this is for plotting purpose
    index = np.arange(len(labels))
    plt.bar(index, values)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    # plt.xticks(index, labels, fontsize=10, rotation=rot)
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.clf()  # clear memory
        plt.close()

    else:
        plt.show()


def plot_pie(sizes, labels, colors, save_to):
    patches, _, autotexts = plt.pie(sizes, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.legend(patches, labels, loc="best")
    for autotext in autotexts:
        autotext.set_color('white')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.show()


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


def plot_dataframe(df, save_as=None, kind='bar', legend=False):
    fig, ax = plt.subplots(1, 1)
    ax.get_xaxis().set_visible(False)
    df.plot(table=True, ax=ax, kind=kind, legend=legend)
    plt.savefig(save_as)
    plt.show()


#
# def plot_roc():
#     plt.figure()
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["micro"]),
#              color='deeppink', linestyle=':', linewidth=4)
#
#     plt.plot(fpr["macro"], tpr["macro"],
#              label='macro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["macro"]),
#              color='navy', linestyle=':', linewidth=4)
#
#     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                  label='ROC curve of class {0} (area = {1:0.2f})'
#                        ''.format(i, roc_auc[i]))
#
#     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Some extension of Receiver operating characteristic to multi-class')
#     plt.legend(loc="lower right")
#     plt.show()


def plot_losses(training_res, save_as=None, text=None):
    loss, val_loss = [epoch['loss'] for epoch in training_res], [epoch['val_loss'] for epoch in training_res]
    epoches = list(range(1, len(loss) + 1))
    f = plt.figure(figsize=(8, 8))
    ax = f.add_subplot(1, 1, 1)
    ax.plot(epoches, loss, '-ko', label='train loss', markersize=3, markerfacecolor='w')
    ax.plot(epoches, val_loss, '-ro', label='val loss', markersize=3, markerfacecolor='w')
    anchored_text = AnchoredText(text, loc="center")
    ax.add_artist(anchored_text)
    ax.set_title('model loss evolution ', fontsize=10)
    ax.set_xlabel("epochs", fontsize=10)
    ax.set_ylabel("loss", fontsize=10)

    plt.legend()
    if save_as is not None:
        plt.savefig(save_as)
        plt.clf()  # clear memory
        plt.close()
    plt.show()


def plot_metrics(dir):
    res = collections.defaultdict(dict)

    for elem in os.listdir(path=dir):
        if elem.endswith("_out.json"):
            with open(os.path.join(dir, elem), "r") as OUT:
                cont = json.load(OUT)
                threshold = float(elem.split("_")[0])
                res[threshold] = cont
        elif elem.startswith("history"):
            print(elem)
            plot_losses(fname=os.path.join(dir, elem))

    thresholds = list(res.keys())
    y_val = list(res.values())
    y1_val = [(x1, y1) for val in y_val for x1, y1 in val["first_group"].items()]
    y1_metrics = list(set([x1 for val in y_val for x1, _ in val["first_group"].items()]))
    y2_val = [(x2, y2) for val in y_val for x2, y2 in val["snd_group"].items()]
    y2_metrics = list(set([x2 for val in y_val for x2, _ in val["snd_group"].items()]))

    tab = collections.defaultdict(list)

    print(y1_metrics)
    for i, met in enumerate(y1_metrics):
        temp1 = [(i, g["first_group"][met]) for i, g in res.items()]
        if met == "accuracy":
            met = "mean_acc"
        temp2 = [(i, g["snd_group"][met]) for i, g in res.items()]
        thres2, max2 = max(temp2, key=operator.itemgetter(1))[0], max(temp2, key=operator.itemgetter(1))[1]
        thres1, max1 = max(temp1, key=operator.itemgetter(1))[0], max(temp1, key=operator.itemgetter(1))[1]
        tab[met].append(thres1)
        tab[met].append(max1)
        tab[met].append(thres2)
        tab[met].append(max2)

    table = pd.DataFrame(tab, index=["first_group : best threshold", "first_group : best value (%)",
                                     "snd_group : best threshold", "snd_group : best value (%)"])
    print(table)

    tmp_1g = collections.defaultdict(list)
    fig1 = plt.figure(figsize=(8, 8))
    nrow = 5
    ncol = 3
    for i, met in enumerate(y1_metrics):
        y1 = [val * 100 for metric, val in y1_val if metric == met]
        tmp_1g[met] = y1
        ax = fig1.add_subplot(nrow, ncol, i + 1)
        ax.plot(thresholds, y1, '.', label=met)
        ax.set_xlabel('thresholds')
        ax.set_ylabel('values (%)')
        ax.set_title(met, fontsize='large')
    plt.show()

    for m, v in tmp_1g.items():
        plt.plot(thresholds, v, ".", label=m)
        plt.xlabel('thresholds')
        plt.ylabel('values (%)')
        plt.title(label=f"First group of metrics -  recap")
        plt.legend()
    plt.show()

    tmp_2g = collections.defaultdict(list)
    fig2 = plt.figure(figsize=(8, 8))
    for i, met in enumerate(y2_metrics):
        y2 = [val * 100 for metric, val in y2_val if metric == met]
        tmp_2g[met] = y2
        ax = fig2.add_subplot(nrow, ncol, i + 1)
        ax.plot(thresholds, y2, '.', label=met)
        ax.set_xlabel('thresholds')
        ax.set_ylabel('values (%)')
        ax.set_title(met, fontsize='large')
    plt.show()

    for m, v in tmp_2g.items():
        plt.plot(thresholds, v, ".", label=m)
        plt.xlabel('thresholds')
        plt.ylabel('values (%)')
        plt.title(label=f"2nd group of metrics -  recap")
        plt.legend(loc='upper right')
    plt.show()


def describe_dataset(input_path, dataset_name, save=None):
    all_drugs_path = f"{input_path}/{dataset_name}-drugs-all.csv"
    df1 = {key: len(val) for key, val in load_smiles(all_drugs_path, dataset_name=dataset_name).items()}
    # Similes distribution
    values = list(df1.values())
    values.sort(reverse=True)
    labels = range(len(values))
    x_label = "drugs"
    y_label = "smiles length"
    title = f"Distribution  of smiles length - {dataset_name}"
    rot = 90
    plot_bar_x(labels, values, x_label, y_label, title, None, rot)

    # Load data 2
    labels, x_train, x_test, x_val, y_train, y_test, y_valid = load_train_test_files(
        input_path=input_path,
        dataset_name=dataset_name,
        one_hot=False, method="Not important")
    x_axis = [i for i in range(len(labels))]
    plt.plot(x_axis, y_train.sum(axis=0).tolist(), "-", label="train")
    plt.plot(x_axis, y_test.sum(axis=0).tolist(), "-", label="test")
    plt.plot(x_axis, y_valid.sum(axis=0).tolist(), "-", label="valid")
    plt.xlabel('type of side effects')
    plt.ylabel('number of samples')
    plt.xlim(0, y_test.shape[1] + 5)
    plt.legend()
    plt.title(f"Repartition des classes - {dataset_name}")
    if save is not None:
        plt.savefig(f"{save}-2.png")
        plt.clf()  # clear memory
        plt.close()

    else:
        plt.show()

    summary_1 = [len(df1), len(y_test) + len(y_train) + len(y_valid),
                 max(df1.items(), key=operator.itemgetter(1))[0], max(df1.items(), key=operator.itemgetter(1))[1],
                 y_train.shape[1], len(y_train), len(y_valid), len(y_test)]

    df = pd.DataFrame.from_records(
        [y_train.sum(axis=0).tolist(), y_test.sum(axis=0).tolist(), y_valid.sum(axis=0).tolist()], columns=labels,
        index=["train", "test", "valid"])
    df.loc['Total'] = df.sum()

    return summary_1, df


def compare(e, f, n1, n2):
    labels = list(e)
    y = [e.iloc[0][i] for i in range(len(labels))]
    z = [f.iloc[0][i] for i in range(len(labels))]
    x = [i for i in range(len(labels))]
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.bar([xs - 0.2 for xs in x], [ys * 100 for ys in y], width=0.2, color='b', align='center',
           label=n1)
    ax.bar([xs for xs in x], [zs * 100 for zs in z], width=0.2, color='r', align='center', label=n2)
    ax.set_title('metrics')
    ax.set_ylabel("values")
    plt.xticks(x, labels, fontsize=10, rotation=25)
    plt.legend()
    plt.show()


def describe_labels_repartition(y, prefix):
    dc1 = {label: tot for label, tot in enumerate(np.sum(y, axis=0).tolist())}
    print(dc1)
    print(pd.DataFrame(dc1, index=[0]))
    plt.plot(dc1.keys(), dc1.values(), '.:')
    plt.xticks(range(1, len(dc1) + 1, 5), range(1, len(dc1) + 1, 5), rotation='vertical', fontsize=5)
    plt.title('Number of Examples by label')
    plt.ylabel('count')
    plt.xlabel('label')
    plt.grid()
    plt.savefig("{}- Distribution of labels.png".format(prefix))
    plt.show()


def plot_correlation_matrix(labels, method='pearson', save=None):
    y = labels
    df = pd.DataFrame(y)
    corr_mat = df.corr(method=method)
    plt.matshow(corr_mat)
    plt.xticks(range(1, len(df.columns) + 1, 5), [i for i, v in enumerate(df.columns) if i % 5 == 0], fontsize=5)
    plt.yticks(range(len(df.columns)), df.columns, fontsize=5)
    plt.colorbar()
    plt.savefig(save)
    plt.show()


def expt_summary(dir):
    res = []
    for c in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, c)):
            print(c)
            hp, auroc, n_epochs, n_epochs_expected, path, expt = {}, 0, 0, 0, None, None
            for file in os.listdir(os.path.join(dir, c)):
                try:
                    if file.endswith("config.json"):
                        with open(os.path.join(dir, c, file), "r") as CONF:
                            conf = json.load(CONF)
                            n_epochs_expected = conf['fit_params']["n_epochs"]
                            expt_name = conf["model_name"]
                            expt = expt_name
                            hp = expt_name.split("expt-")[-1].split("%")
                            hp = dict(zip(hp[:len(hp) // 2], hp[len(hp) // 2:]))

                    if file.endswith("out.json"):
                        with open(os.path.join(dir, c, file), "r") as OUT:
                            cont = json.load(OUT)
                            auroc = cont["first_group"]["auroc"]
                    if file.endswith("history.json"):
                        path = os.path.join(dir, c, file)
                        with open(path, "r") as HIST:
                            n_epochs = len(json.load(HIST))
                            path = os.path.dirname(path)
                    plot_losses(fname=f"{path}/history.json", text="\n".join([f"{hp}:{val}" for hp, val in hp.items()]),
                                save_as=f"/home/rogia/Documents/code/side_effects/figures/fig-4/{expt}.png")

                except Exception:
                    pass
            res.append((hp, auroc, f"{n_epochs}/{n_epochs_expected}", path))

    best = max(res, key=operator.itemgetter(1))
    print(best[-1])
    print(f"The best combination of hyperparameters is: {best[0]}==>{best[1]}")
    return res


def ranking_expts(res, min_auroc=0.58):
    records = []
    for expt in res:
        auroc = expt[1]
        if auroc > min_auroc:
            hps = expt[0]
            hps["auroc"] = auroc
            hps["% COMPLETED"] = expt[2]
            records.append(hps)
    df = pd.DataFrame(records)
    return df


def expts_figs(inputs):
    best = max(inputs, key=operator.itemgetter(1))
    plot_losses(f"{best[-1]}/history.json")


def describe_data(y_train, y_test, y_valid, dataset_name="drugbank"):
    len_labels = y_train.shape[1]
    x_axis = [i for i in range(len_labels)]
    plt.plot(x_axis, y_train.sum(axis=0).tolist(), "-", label="train")
    plt.plot(x_axis, y_test.sum(axis=0).tolist(), "-", label="test")
    plt.plot(x_axis, y_valid.sum(axis=0).tolist(), "-", label="valid")
    plt.xlabel('type of side effects')
    plt.ylabel('number of samples')
    plt.xlim(0, y_test.shape[1] + 5)
    plt.legend()
    plt.title(f"Repartition des classes - {dataset_name}")
    plt.show()

    df = pd.DataFrame.from_records(
        [y_train.sum(axis=0).tolist(), y_test.sum(axis=0).tolist(), y_valid.sum(axis=0).tolist()],
        columns=[str(i) for i in range(1, len_labels + 1)],
        index=["train", "test", "valid"])
    df.loc['Total'] = df.sum()

    return df


if __name__ == '__main__':
    #
    # a = pd.DataFrame([out["ap"], out["ROC"]], index=["AUPRC", "AUROC"]).transpose()
    # i = a.plot(kind='bar',  figsize=(8, 8), fontsize=14, stacked=True).get_figure()
    # #plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    # plt.show()
    # exit()
    # i.savefig("e.png")
    # exit()
    # # on verra plus tard pour a = pickle.load(open("/home/rogia/Documents/git/side_effects/data/labels_code.pkl", "rb"))
    # html_out = "hia.html"
    # a.to_html(html_out)
    # pdfkit.from_file("hia.html", "out.pdf")
    #
    # exit()
    #     all_drugs_path = "/home/rogia/Documents/code/side_effects/data/violette/twosides/twosides-drugs-all.csv"
    #     train_path = "/home/rogia/Documents/code/side_effects/data/violette/twosides/twosides-train_samples.csv"
    #     test_path = "/home/rogia/Documents/code/side_effects/data/violette/twosides/twosides-test_samples.csv"
    #     valid_path = "/home/rogia/Documents/code/side_effects/data/violette/twosides/twosides-valid_samples.csv"
    #     comb = "/home/rogia/Documents/code/side_effects/data/violette/twosides/twosides-combo.csv"
    #
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('-d', '--dir', type=str, default="", help=""" The path of metrics repository.""")
    #     parser.add_argument('-f', '--function_name', type=str, help=""" The path of metrics repository.""")
    #     parser.add_argument('-s', '--smiles', type=str, default="", help=""" The path of metrics repository.""")
    #     args = parser.parse_args()
    #
    #     function_name = args.function_name
    #     if function_name == "plot_metrics":
    #         plot_metrics(dir=args.dir)
    #
    #     elif function_name == "smiles_distribution":
    #         describe_dataset(all_drugs_path, train_path, test_path, valid_path)  # oubie pas de le changer
    #
    #     e, f = describe_dataset(all_drugs_path, train_path, test_path, valid_path)
    #     #
    #     # path = "/home/rogia/Documents/code/side_effects/expts-uniform-weigths/expt-4"
    #     # recap = expt_summary(dir=path)
    #     # top_expts = ranking_expts(recap)
    #     # print(top_expts)
    #     # expts_figs(recap)
    #
    from sklearn.metrics import roc_auc_score, average_precision_score

    a = unpack_results("/home/rogia/Images")
    print(a[1]["ap"]["micro"])
    print(a[1]["ROC"]["micro"])
    print(average_precision_score(a[2], a[3], "micro", pos_label=0))
    plot_losses(a[-1])
    exit()
    df = describe_all_experiments("/home/rogia/Documents/analysis/results")
    print(df)
    save_results(filename="//home/rogia/Documents/analysis/rapport/druud-all-invivo-drugbank-seeds.xlsx",
                 contents=[("res", df)])
    exit()
    # # Revoir axes graphiques
