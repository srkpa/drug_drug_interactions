import csv
import json as js
import os
from collections import defaultdict

import pandas as pd

# Need to be deleted
def load_raw_data(raw_data_file, mapping_file):
    case = 0
    data = {}
    extension = os.path.splitext(mapping_file)[-1]
    if extension is ".json":
        case = 1
        with open(file=mapping_file) as json_file:
            data = js.load(json_file)
    else:
        f = open(mapping_file, 'r')
        f.readline()
        for row in f:
            row = row.split(",")
            data[row[0]] = row[1]

    f = open(raw_data_file)
    f.readline()
    i = 0
    res = defaultdict(set)
    for line in open(raw_data_file):
        a = line.split("\t")
        stitch_id1 = a[0]
        stitch_id2 = a[1]
        event_name = a[5]
        if case > 0:
            for main_class, sub_terms in data.items():
                if event_name.upper() in sub_terms:
                    res[(stitch_id1, stitch_id2)].add(main_class)
        else:
            for side_effect, main_class in data.items():
                if event_name.lower() == side_effect.lower():
                    res[(stitch_id1, stitch_id2)].add(main_class)
    return res


def read_drug_food_interactions(filename):
    res = defaultdict(set)
    df = pd.read_excel(filename, sep="\t")
    df = df[["Food constituent", 'Interacting drug', "DDI type predicted using DeepDDI"]]
    print(df.shape)
    for row in df.itertuples():
        res[(str(row[1]).lower(), str(row[2]).lower())].add(str(row[3]))
    print(len(res))
    return res


def reformat_ddis_file(raw_data_file):
    res = defaultdict(set)
    f = open(raw_data_file)
    f.readline()


if __name__ == '__main__':
    import pickle

    out_1 = read_drug_food_interactions(filename="/home/rogia/Téléchargements/pnas.1803294115.sd05.xlsx")
    # # out = load_raw_data(mapping_file="/home/rogia/Bureau/twosides-2-icd11.csv",
    # #                     raw_data_file="/home/rogia/Documents/raw_data/3003377s-twosides.tsv")
    # arc = list(zip(*list(out.keys())))
    # a = list(set(arc[0]))
    # res = load_smiles(a)
    # pickle.dump(res, open(os.path.join("/home/rogia/", "foods_smiles.pkl"), "wb"))
    w = csv.writer(open("directed-drugbank-combo.csv", "w"))
    w.writerow(["Drug 1", "Drug 2", "ddi type"])
    for ddi, se in out_1.items():
        w.writerow(["_".join(ddi[0].split(",")), ddi[1], ";".join(list(se))])

    approved_drugs_path = "/home/rogia/Documents/code/side_effects/data/violette/drugbank/drugbank_approved_drugs.csv"
    approved_drug = pd.read_csv(approved_drugs_path, sep=",").dropna()
    approved_drug = approved_drug[['drug_id', 'drug_name', 'PubChem Canonical Smiles']]

    r = pickle.load(open(os.path.join("/home/rogia/", "foods_smiles.pkl"), "rb"))
    w = csv.writer(open("drugbank-drugs-all.csv", "w"))
    w.writerow(["drug_id", "drug_name", "smiles"])
    i, j = 0, 0
    f = 1
    for name, smile in r.items():
        if smile != "NA":
            w.writerow(["_".join(name.split(",")), smile])
            i += 1
        else:
            j += 1
        f += 1
    print(i)
    print(j)

    with open('drugbank-drugs-all.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        for row in approved_drug.itertuples():
            writer.writerow([str(row[2]).lower(), row[1], row[3]])
    csvFile.close()
