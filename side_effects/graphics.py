import click
import json
import os
import pickle
import shutil
import tempfile
import matplotlib.pyplot as plt
from ivbase.utils.aws import aws_cli
from side_effects.data.utils import *

INVIVO_RESULTS = "/home/rogia/.invivo/result"


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


def save_results(filename, contents, engine='xlsxwriter'):
    writer = pd.ExcelWriter(filename, engine=engine)
    for sheet, data in contents:
        data.to_excel(writer, sheet_name=sheet)
    writer.save()


def describe_all_experiments(output_folder):
    out = []
    for c in os.listdir(output_folder):
        folder = os.path.join(output_folder, c)
        if os.path.isdir(folder):
            params, res, _, _ = _unpack_results(folder)
            out.append({**params, **res, "task_id": c})

    return pd.DataFrame(out)


def _format_dict_keys(expt_config):
    new_expt_config = {}
    for param, val in expt_config.items():
        new_key = param.split(".")[-1]
        new_expt_config[new_key] = val
    return new_expt_config


def _complete_metrics(all_metrics, keys_to_remove, y_preds, y_true):
    for i in keys_to_remove:
        del all_metrics[i]
    for n, m in all_metrics.items():
        print(n, m(y_preds, y_true))


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


def run(s3_path, dst):
    out_path = download_results(s3_path, dst)
    print(out_path)
    df = describe_all_experiments(out_path)
    df.fillna(0, inplace=True)
    a = df.groupby(["dataset_params.split_mode", "model_params.use_negative_sampled_loss",
                    "model_params.network_params.att_hidden_dim",
                    "dataset_params.use_graph"], as_index=True)
    c = a[["micro_roc", "micro_auprc"]].max()
    return c


def pie(file_path="/home/rogia/Téléchargements/twosides-icd11.csv"):
    data = pd.read_csv(file_path, sep=",")
    icd2se = data.groupby(["icd"])
    fig, axs = plt.subplots(5, 5)
    g = plt.colormaps()
    i = 0
    for name, group in icd2se:
        labels = list(icd2se.groups[name])
        print(len(labels), len(set(labels)))
        sizes = [1] * len(labels)
        if 0 <= i < 5:
            ax1 = axs[0, i]
        elif 5 <= i < 10:
            ax1 = axs[1, i - 5]
        elif 10 <= i < 15:
            ax1 = axs[2, i - 10]
        elif 15 <= i < 20:
            ax1 = axs[3, i - 15]
        else:
            ax1 = axs[4, i - 20]
        cm = plt.get_cmap(g[i])
        cout = cm(np.arange(len(labels)))
        ax1.pie(sizes, labels=None, startangle=90, colors=cout)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.axis('equal')
        ax1.set_title(name)
        plt.tight_layout()

        i += 1
    plt.show()


if __name__ == '__main__':
    pie()
