import json
import subprocess
from collections import defaultdict
from itertools import chain
from os.path import expanduser, dirname

import pandas as pd
import xlsxwriter

from side_effects.data.loader import load_ddis_combinations


def to_excel(filename, my_dict):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    row_num = 0
    for key, value in my_dict.items():
        for i, item in enumerate(value):
            if i == 0:
                worksheet.write(row_num, 0, key)
            worksheet.write(row_num, 1, item)
            row_num += 1

    workbook.close()


def __get_info__(output_file, idx):
    meth_folder = f"{expanduser('~')}/Documents/projects/uts-rest-api/samples/python"
    apikey = "f9ef9875-6fc6-45af-8339-3051ad18266a"
    current = "2019AB"
    subprocess.run(
        ["python", f"{meth_folder}/retrieve-cui-or-code.py", "-k", apikey, "-v", current, "-i", idx, "-o",
         output_file], capture_output=True)


def retrieve_cui(output_file, path="/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides"):
    data = pd.read_table(path + "/3003377s-twosides.tsv", sep="\t")
    se2name = dict(zip(data["event_umls_id"], data["event_name"]))

    for cui, name in se2name.items():
        __get_info__(output_file, cui)

    with open(output_file, "r") as out:
        output_file_1 = dirname(output_file) + "/final_umls.json"
        result = json.load(out)
        result = {se2name[identifier].lower(): relations for identifier, relations in result.items()}
        json.dump(result, open(output_file_1, "w"), indent=4)


def load_side_effects(filepath=""):
    cach_path = "/home/rogia/.invivo/cache/datasets-ressources/DDI/twosides/"
    data = pd.read_table(cach_path + "/3003377s-twosides.tsv", sep="\t")
    data1 = load_ddis_combinations(cach_path + "twosides.csv", True, "twosides")

    side_effects = list(set(chain.from_iterable(data1.values())))
    name2cuix = dict(zip(data["event_name"].str.lower(), data["event_umls_id"]))
    name2cui = {name: cui for name, cui in name2cuix.items() if name in side_effects}
    cui2name = {cui: name for name, cui in name2cuix.items() if name in side_effects}
    assert len(side_effects) == len(name2cui)

    output = {}
    with open(filepath, "r") as fp:
        se_dict = json.load(fp)
        for side_effect, cui in name2cui.items():
            out = []
            relations = se_dict.get(side_effect, [])
            for rel in relations:
                cui = rel.get("cui", None)
                if cui in cui2name and rel["relationLabel"] != "RO":
                    rel["currentName"] = cui2name[cui]
                    out.append(rel)
            if out:
                output[side_effect] = out

    # repartition
    rn = defaultdict(set)
    for name, rels in output.items():
        tags = [rel["relationLabel"] == "RB" for rel in rels]
        if all(tags):
            rn[name] = set([rel["currentName"] for rel in rels])
        for rel in rels:
            tag = rel["relationLabel"]
            if tag == "RN":
                curr_name = rel["currentName"]
                rn[curr_name].add(name)

    # Check rejected key child
    all_se_dict = json.load(
        open("/home/rogia/Documents/projects/drug_drug_interactions/side_effects/data/umls.json", "r"))
    all_se_dict.update(json.load(
        open("/home/rogia/Documents/projects/drug_drug_interactions/side_effects/data/additionnal_2_umls.json", "r")))
    all_se_dict.update(json.load(
        open("/home/rogia/Documents/projects/drug_drug_interactions/side_effects/data/additionnal_3_umls.json", "r")))

    for name in side_effects:
        if name not in output:
            rels = se_dict.get(name, [])
            rels = [rel for rel in rels if rel["cui"] in all_se_dict]
            for rel in rels:
                if rel["relationLabel"] != "RO":
                    out = all_se_dict[rel["cui"]]
                    for child in out:
                        scui = child["cui"]
                        if scui in cui2name:
                            if rel["relationLabel"] == "RB" and child["relationLabel"] == "RB":
                                rn[name].add(cui2name[scui])
                            elif rel["relationLabel"] == "RN" and child["relationLabel"] == "RN":
                                rn[cui2name[scui]].add(name)

    # Merging
    clusters = defaultdict(set)
    to_remove = []
    for gterm, elem in rn.items():
        if gterm not in to_remove:
            raw_set = elem
            for side_effect in elem:
                if side_effect in rn:
                    to_remove += [side_effect]
                    raw_set = raw_set.union(rn[side_effect])
            clusters[gterm] = raw_set

    # Check if RN is RB of another group
    y = list(set(chain.from_iterable(list(clusters.values()))))
    i = 0
    to_remove = []
    for l in clusters:
        if l in y:
            to_remove += [l]
            i += 1
    temp_dict = defaultdict(set)
    for gphen, sphen in clusters.items():
        for phen in to_remove:
            if phen in sphen:
                temp_dict[phen].add(gphen)
    for phen, cls in temp_dict.items():
        for r in cls:
            clusters[r] = clusters[r].union(clusters[phen])
    for key in temp_dict:
        del clusters[key]
    # Stats + data saving
    n = len(clusters)
    n2 = len(set(chain.from_iterable([k for k in clusters.values()])))
    print(n, n2, n + n2, len(side_effects) - n - n2)
    to_excel("twosides_umls.xlsx", clusters)
    clusters = {key: list(value) for key, value in clusters.items()}
    with open("twosides_umls.json", "w") as cf:
        json.dump(clusters, cf, indent=4)
