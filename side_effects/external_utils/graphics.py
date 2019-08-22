import json
import os
import pickle

from side_effects.data.utils import *


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
            out.append({**params, **res, "abs path": folder})

    return pd.DataFrame(out)


def _unpack_results(folder):
    results, y_true, y_preds = None, None, None
    params = json.load(open(os.path.join(folder, "config.json"), "rb"))
    params = params.pop("dataset_params")

    for f in os.listdir(folder):
        filepath = os.path.join(folder, f)
        if os.path.isfile(filepath) and os.path.splitext(filepath)[-1] == ".pkl":
            out = pickle.load(open(filepath, "rb"))
            if f.endswith("_preds.pkl"):
                y_preds = out
            elif f.endswith("_res.pkl"):
                results = out
            elif f.endswith("_targets.pkl"):
                y_true = out

    return params, results, y_true, y_preds


if __name__ == '__main__':
    df = describe_all_experiments("/home/rogia/Documents/analysis/test/")
    # print(df)
    df1 = df[df["split_mode"] == "random"]
    # df2 = df[df["split_mode"] != "random"]
    #print(df1)
    print(df1[df1["micro_auprc"] == max(df1["micro_auprc"])])
    # print(max(df2["micro_auprc"]))

