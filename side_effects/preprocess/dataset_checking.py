import pandas as pd
from side_effects.preprocess.dataset import _extract_interactions, load_ddis_combinations

import csv


def combo():
    deepddi_github_path = "/home/rogia/Documents/git/deepddi/data"
    interactions_file = f"{deepddi_github_path}/Interaction_information.csv"
    df1 = pd.read_csv(interactions_file, sep=",")

    print("interaction information", list(df1), df1.shape)

    drugbank_comb = []
    dk = open(f"{deepddi_github_path}/DrugBank_known_ddi.txt", "r")
    dk.readline()

    for line in dk:
        interaction = line.strip("\n").split("\t")
        drug1 = interaction[0]
        drug2 = interaction[1]
        label = int(interaction[2])
        a = df1[df1["Interaction type"] == label]["DDI type"].values[-1]
        a = a.split("DDI_type")[-1]
        drugbank_comb.append((drug1, drug2, a))

    df2 = pd.DataFrame(drugbank_comb, columns=["drug1", "drug2", "ddi type"])
    print(df2.shape)
    df2.to_csv("/home/rogia/Bureau/new/drugbank-combo.csv", sep=",")


import numpy as np
from collections import defaultdict


def assign_ddi_type():
    combo_path = "/home/rogia/Bureau/new/drugbank-combo.csv"
    b = pd.read_csv(combo_path, sep=",", header=None)

    # a = load_ddis_combinations(combo_path, header=False, dataset_name="drugbank")
    d = [(i[0], i[1]) for i in b[[1, 2]].values]
    print(len(d))
    print(len(set(d)))
    # df2 = pd.read_excel("/home/rogia/Documents/code/side_effects/data/deepddi/drugbank/pnas.1803294115.sd02.xlsx", sheet_name="Dataset S1")
    combo2se, combo = defaultdict(list), defaultdict(list)
    for l in b.values:
        drug1 = l[1]
        drug2 = l[2]
        a = l[3]
        combo2se[(drug1, drug2)].append(a)
    for i, v in combo2se.items():
        r = [str(e) for e in v]
        combo[i] = list(set(r))
    print(len(combo))
    i = 0

    c = load_ddis_combinations(fname="/home/rogia/Documents/code/side_effects/data/deepddi/drugbank/drugbank-combo.csv",
                               header=True, dataset_name="drugbank")

    print(len(c))
    erreur = []
    for k, j in combo.items():
        if k in c:
            if set(j) != set(c[k]):
                # print(k, c[k], j)
                erreur.append(k)

    for jy in erreur:
        rever = (jy[1], jy[0])
        if rever in combo:
            i += 1
            print(jy, combo[jy], rever, combo[rever], c[jy])
    print(i)
    exit()
    w = csv.writer(open("directed-drugbank-combo.csv", "w"))
    w.writerow(["Drug 1", "Drug 2", "ddi type"])
    for (drug1, drug2), lir in combo.items():
        w.writerow([drug1, drug2, ";".join(lir)])
    exit()
    w = csv.writer(open("drugbank-combo.csv", "w"))
    w.writerow(["Drug 1", "Drug 2", "ddi type"])
    for (drug1, drug2), lir in c.items():
        w.writerow([drug1, drug2, ";".join(lir)])
    import pickle
    pickle.dump(c, open("save.p", "wb"))
    # s2_path = "/home/rogia/Bureau/pnas.1803294115.sd02.xlsx"
    # df2 = pd.read_excel(s2_path, sheet_name="Dataset S2")
    # df2["pair"] = df2['Sentences describing the reported drug-drug interactions'].apply(_extract_interactions)
    #
    # fg = df2.values
    # fr = [tuple(i.split(",")) for i in df2["pair"].values]
    # for p in erreur:
    #     if p[0] in fr and (p[0][1], p[0][0]) not in fr:
    #         ind = fr.index(p[0])
    #         print(p[0], fg[ind], p[1:])
    #
    #


def reformat_s2():
    a = csv.writer(open("drugbank-train_samples.csv", "w"))
    b = csv.writer(open("drugbank-test_samples.csv", "w"))
    d = csv.writer(open("drugbank-valid_samples.csv", "w"))
    c = load_ddis_combinations(fname="/home/rogia/Documents/code/side_effects/data/deepddi/drugbank/drugbank-combo.csv",
                               header=True, dataset_name="drugbank")

    print(len(c))
    i = 0
    for o, p in c.items():
        if len(p) == 2:
            i += 1
    print(i)
    exit()
    s2_path = "/home/rogia/Bureau/pnas.1803294115.sd02.xlsx"
    df2 = pd.read_excel(s2_path, sheet_name="Dataset S2")
    mapu = {}
    capo = [_extract_interactions(l[0]) for l in df2.values]

    for i, l in enumerate(df2.values):
        pa = _extract_interactions(l[0])
        cl = l[-1]
        if pa in c:
            if (pa[-1], pa[0]) not in c:
                mapu[pa] = c[pa]
            else:
                mapu[pa] = c[pa][0]
                print("yop")



        else:
            assert (pa[-1], pa[0]) in c
            rem = (pa[-1], pa[0])
            mapu[pa] = c[rem][-1]
        if cl == "training":
            writer = a
        elif cl == "testing":
            writer = b
        else:
            writer = d
        writer.writerow([pa[0], pa[1], mapu[pa]])


if __name__ == '__main__':
    # assign_ddi_type()
    # reformat_s2()
    # assign_ddi_type()
    c = load_ddis_combinations(fname="directed-drugbank-combo.csv",
                               header=True, dataset_name="split")
    tra = load_ddis_combinations(fname="drugbank-train_samples.csv", header=False, dataset_name="split")
    tes = load_ddis_combinations(fname="drugbank-test_samples.csv", header=False, dataset_name="split")
    val = load_ddis_combinations(fname="drugbank-valid_samples.csv", header=False, dataset_name="split")

    for e in  tes:
        assert e in c
        assert set(c[e])==set(tes[e])
        #print(set(c[e]), set(tes[e]), set(c[e])==set(tes[e]))
