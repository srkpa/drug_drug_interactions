import pandas as pd
import numpy as np
import re
from side_effects.data.utils import load_ddis_combinations
from collections import defaultdict


if __name__ == '__main__':

    c = load_ddis_combinations(fname="directed-drugbank-combo.csv",
                               header=True, dataset_name="split")
    print(len(c))
    s2 = pd.ExcelFile('/home/rogia/Documents/code/side_effects/data/deepddi/drugbank/pnas.1803294115.sd02.xlsx')
    df1 = s2.parse(s2.sheet_names[-1])

    # Build our regex to extract drugs
    pattern = r'(?P<drugid>DB[0-9]+)'
    current_drugs = [drug_id for sentence in
                     df1['Sentences describing the reported drug-drug interactions'].values.tolist() for drug_id in
                     re.findall(pattern, sentence)]

    current_drugs = np.unique(current_drugs)

    # Our model dataset
    s2 = pd.ExcelFile('/home/rogia/Documents/code/side_effects/data/deepddi/drugbank/pnas.1803294115.sd01.xlsx')
    df4 = s2.parse(s2.sheet_names[-1])

    df5 = pd.DataFrame(df4['General sentence structure'].apply(
        lambda x: re.sub("Drug [ab]", "DB[0-9]+", x.replace("(", "").replace(")", ""))))

    current_ddi = [
        tuple(np.unique(re.findall(pattern, row['Sentences describing the reported drug-drug interactions']))) for
        _, row in df1.iterrows()]

    df1['ddi'] = current_ddi

    # group by drug - drug combinations
    agg = df1.groupby('ddi')
    dictionnaire = []
    data = []
    id = 1

    for name, group in agg:
        print(name, id)
        rev = (name[-1], name[0])
        sentence = group['Sentences describing the reported drug-drug interactions']
        partition = group["Data type used to optimize the DNN architecture"].values

        check = []
        for po, s in enumerate(sentence):
            paire = None
            ddi_type = 0
            s = s.replace("(", "").replace(")", "")
            for _, row in df5.iterrows():
                match = re.findall(row['General sentence structure'], s)
                if match:
                    ddi_type = str(df4.iloc[_]['DDI type'])
            if ddi_type == 0:
                print(s)
            else:
                if name in c:
                    if rev not in c:
                        paire = name
                    else:
                        if ddi_type in c[name]:
                            paire = name
                        elif ddi_type in c[rev]:
                            paire = rev
                else:
                    if rev in c:
                        paire = rev

            if paire is not None:
                check.append([s, partition[po], paire, ddi_type])

        for i in check:
            dictionnaire.append(i)

        id += 1

    print("dictionnaire", len(dictionnaire))
    df2 = pd.DataFrame(dictionnaire,
                       columns=["Sentences describing the reported drug-drug interactions",
                                "Data type used to optimize the DNN architecture", "paire", "ddi type"])
    print(df2.shape)
    df2.to_csv("s2_mapping_01.csv", sep=",")

    sant = defaultdict(list)
    for phrase, part, paire, ddi in dictionnaire:
        sant[paire].append(ddi)

    print(len(sant))

    for cle, val in sant.items():
        assert cle in c
        assert set(val) == set(c[cle])
    # j'ai verifié le mapping et il est correct