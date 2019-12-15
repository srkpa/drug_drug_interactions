import csv
from collections import defaultdict, Counter

import pubchempy as pcp
from ivbase.transformers.features.pathways import *
from rdkit.Chem import MolFromSmiles, Draw


def load_smiles(molecules):
    """
    This function allows to retrieve smiles from PubChem continuously without having to worry about the limit of requests imposed by the site.
    :param molecules: the list of the names of the molecules for which the smiles must be found
    :return: a list of smiles
    """
    res = {}
    for ids, mol in enumerate(molecules):
        smi = ""
        try:
            cs = pcp.get_compounds(mol, 'name')
            smi = cs[0].canonical_smiles
        except IndexError:
            smi = "NA"
        except pcp.PubChemHTTPError:
            checkpoint_id = ids
            molecules = molecules[checkpoint_id:]
            print("CheckPoint:{}\n".format(str(checkpoint_id)))
        print("{}:{}".format(str(mol), smi))
        res[mol] = smi
    return res


def draw_mol_with_rdkit(smile, filename):
    molecule = MolFromSmiles(smile)
    # Draw.MolToFile(mol=molecule, fileName=filename)
    img = Draw.MolToImage(molecule)
    img.save(filename)
    print(filename)


def draw_mol_with_obabel(smile, output_filename):
    command = 'obabel -:"{}" -O {} -xS'.format(smile, output_filename)
    os.system(command)


def draw_mols(tool, smiles, idx):
    mols = dict(zip(idx, smiles))
    if tool == "obabel":
        for id, smi in mols.items():
            draw_mol_with_obabel(smi, id)
    else:
        for id, smi in mols.items():
            draw_mol_with_rdkit(smi, id)


def download_smiles(fname='../dataset/files/3003377s-twosides.tsv', save=None):
    data = pd.read_csv(fname, sep="\t")
    drugbank = pd.read_csv(
        "/home/prtos/srkpa/Projects/invivoai/InvivoBase/projects/DDI/dataset/new_files/drugbank-drugs-all.csv",
        sep=",")

    cids2names = set(zip(data['stitch_id1'].values.tolist(), data['drug1'].values.tolist())).union(
        set(zip(data['stitch_id2'].values.tolist(), data['drug2'].values.tolist())))

    # We need to rename abbreviations by real drugs name
    names = [n.lower() for _, n in cids2names]
    cids = [c for c, _ in cids2names]
    names[names.index("rsa")] = "n-propargyl-1(s)-aminoindan"
    names[names.index("ftc")] = "emtricitabine"
    names[names.index("etv")] = "entecavir"
    names[names.index("ggg")] = "glycylglycylglycine"
    names[names.index("ets")] = "Erythromycin"
    names[names.index("lambda")] = "lanthanum carbonate"

    cids2names = dict(zip(cids, names))
    names2cids = dict(zip(names, cids))

    smiles = [drugbank.loc[drugbank['drug_name'].str.lower() == x.lower()]['PubChem Canonical Smiles'].tolist() for x in
              names]
    assert len(names) == len(smiles)
    fnames, fsmiles = [names[idf] for idf, x in enumerate(smiles) if
                       x], [x[-1] for idf, x in enumerate(smiles) if x]

    drugs_not_found_in_drugbank = list(set(names) - set(fnames))
    if drugs_not_found_in_drugbank:
        res = load_smiles(drugs_not_found_in_drugbank)
        res['colesevelam hydrochloride'] = 'CCCCCCCCCCNCC=C.C[N+](C)(C)CCCCCCNCC=C.C=CCN.C1C(O1)CCl.Cl.[Cl-]'
        res['valacyclovir'] = "CC(C)C(C(=O)OCCOCN1C=NC2=C1NC(=NC2=O)N)N"
        res["acyclovir"] = "C1=NC2=C(N1COCCO)NC(=NC2=O)N"
        for mol, smile in res.items():
            if smile != "NA":
                fnames.append(mol)
                fsmiles.append(smile)
            else:
                del (cids2names[names2cids[mol]])

    assert len(fnames) == len(fsmiles)
    print(len(cids2names))
    names2smiles = dict(zip(fnames, fsmiles))

    if save is not None:
        w = csv.writer(open("../dataset/files/twosides-drugs-all.csv", "w"))
        w.writerow(["drug_id", "drug_name", "smiles"])
        for name, smile in names2smiles.items():
            w.writerow([names2cids[name], name, smile])
    return cids2names, names2smiles


def extract_ddis(fname='../dataset/files/3003377s-twosides.tsv', save=None, cutoff=500):
    data = pd.read_csv(fname, sep="\t")

    # Apply side effect selection
    se2occ = dict(Counter(data['event_name']))
    se2occ = {se: val for se, val in se2occ.items() if val >= cutoff}
    twoside2se = defaultdict(list)

    for elem in data[['stitch_id1', 'stitch_id2', 'event_name']].values.tolist():
        drug1, drug2 = elem[0], elem[1]
        se = elem[2]
        if se in se2occ:
            twoside2se[(drug1, drug2)].append(
                elem[2].lower())

    print(len(set([s for elem in twoside2se.values() for s in elem])))
    print(len(twoside2se))

    if save is not None:
        w = csv.writer(open(save, "w"))
        w.writerow(["Drug 1", "Drug 2", "ddi type"])
        for ddi, se in twoside2se.items():
            w.writerow([ddi[0], ddi[1], ";".join(list(set(se)))])

    return twoside2se


def download_targets(drugbank_parsed_filename="../data/drugbank_preprocess.csv",
                     fname="../data/violette/drugbank/drugbank-drugs-all.csv"):
    fn = open(fname, "r")
    fn.readline()
    drugbank = pd.read_table(drugbank_parsed_filename, sep='|', encoding='utf-8')
    drugs = [line.split(",")[1] for line in fn]
    print(list(drugbank))
    kegg_targets = [drugbank.loc[drugbank['drug_id'].str.lower() == x.lower()]['UniProtKB'].dropna().tolist() for x in
                    drugs]
    assert len(drugs) == len(kegg_targets)
    drugs2targets = dict(zip(drugs, kegg_targets))
    print(len(drugs2targets))
    return drugs2targets


if __name__ == '__main__':
    e = download_targets()
    print(e)
