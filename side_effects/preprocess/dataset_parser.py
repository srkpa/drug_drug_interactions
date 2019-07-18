import json as js
from collections import defaultdict


def load_raw_data(raw_data_file, mapping_file, criteria="name"):
    with open(file=mapping_file) as json_file:
        data = js.load(json_file)
    # data = {main_term: list(sub_terms.items()) for main_term, sub_terms in data.items()}
    f = open(raw_data_file)
    f.readline()
    i = 0
    res = defaultdict(set)
    for line in open(raw_data_file):
        a = line.split("\t")
        stitch_id1 = a[0]
        stitch_id2 = a[1]
        event_name = a[5]
        for main_class, sub_terms in data.items():
            if event_name.upper() in sub_terms:
                res[(stitch_id1, stitch_id2)].add(main_class)

    return res


if __name__ == '__main__':
    import csv

    out = load_raw_data(mapping_file="/home/rogia/Bureau/twosides_dict.json",
                        raw_data_file="/home/rogia/Documents/raw_data/3003377s-twosides.tsv")
    w = csv.writer(open("directed-drugbank-combo.csv", "w"))
    w.writerow(["Drug 1", "Drug 2", "ddi type"])
    for ddi, se in out.items():
        w.writerow([ddi[0], ddi[1], ";".join(list(se))])
