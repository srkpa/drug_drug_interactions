import re
from collections import defaultdict
from itertools import combinations
from operator import itemgetter

import networkx as nx
import numpy as np
import pandas as pd
import torch
from ivbase.transformers.features import AdjGraphTransformer
from ivbase.transformers.features import DGLGraphTransformer
from ivbase.transformers.features.molecules import FingerprintsTransformer
from ivbase.transformers.features.molecules import SequenceTransformer
from ivbase.transformers.features.targets import GeneEntityTransformer
from ivbase.utils.constants.alphabet import SMILES_ALPHABET
from ivbase.utils.datasets.datacache import DataCache
from ivbase.utils.datasets.external_databases import ExternalDatabaseLoader
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA


# def bioword_to_vec(filepath, terms):
#     embedding = KeyedVectors.load_word2vec_format(
#         datapath(filepath),
#         binary=True)
#     out = torch.Tensor([embedding(t) for t in terms])
#     return dict(zip)


def gene_entity_transformer(drugs, targets):
    transformer = GeneEntityTransformer(include_pathways_features=True)
    transformed_out = transformer.fit_transform(targets)
    res = torch.from_numpy(transformed_out).type("torch.FloatTensor")
    print("gene entity out", res.shape)
    return dict(zip(drugs, res))


def fingerprints_transformer(drugs, smiles):
    transformer = FingerprintsTransformer()
    out = transformer.transform(smiles)
    res = np.stack(out).astype(np.float32)
    # res = torch.from_numpy(res).type("torch.FloatTensor")
    print("fingerprint vectorized out", res.shape)
    return dict(zip(drugs, res))


def sequence_transformer(drugs, smiles, one_hot=False):
    transformer = SequenceTransformer(alphabet=SMILES_ALPHABET, one_hot=one_hot)
    out = transformer.fit_transform(smiles).astype(np.int64)
    # out = torch.from_numpy(out).type("torch.LongTensor")
    print("Sequence vectorized out", out.shape)
    return dict(zip(drugs, out))


def deepddi_transformer(drugs, smiles, approved_drug):
    appr_drugs_mol = [Chem.MolFromSmiles(X) for X in approved_drug]
    appr_drugs_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 4) for mol in appr_drugs_mol]
    drugs_involved_in_mol = [Chem.MolFromSmiles(m) for m in smiles]
    drugs_involved_in_fps = [AllChem.GetMorganFingerprintAsBitVect(X, 4) for X in drugs_involved_in_mol]
    all_ssp = [[DataStructs.FingerprintSimilarity(drg, appr_drugs_fps[i]) for i in range(len(appr_drugs_fps))] for
               ids, drg in enumerate(drugs_involved_in_fps)]
    pca = PCA(n_components=50)
    out = pca.fit_transform(np.array(all_ssp)).astype(np.float32)
    return dict(zip(drugs, out))


# A basd design, i guess.. too much redundancy
def adj_graph_transformer(drugs, smiles, module=AdjGraphTransformer):
    assert len(drugs) == len(smiles)
    X, ids = module()(smiles)
    drugs = list(itemgetter(*ids)(drugs)) if len(ids) < len(drugs) else drugs
    assert len(X) == len(ids)
    assert len(X) == len(drugs)
    return dict(zip(drugs, X))


def dgl_graph_transformer(drugs, smiles, module=DGLGraphTransformer):
    assert len(drugs) == len(smiles)
    X, ids = module()(smiles)
    drugs = list(itemgetter(*ids)(drugs)) if len(ids) < len(drugs) else drugs
    assert len(X) == len(ids)
    assert len(X) == len(drugs)
    return dict(zip(drugs, X))


# create a go graph from raw data: nodes = GO term and edges = GO relations
def __download_gene_ontology(filepath="/home/rogia/Documents/projects/go.obo"):
    fp = open(filepath, "r")
    nodes_desc = fp.read().split('\n\n')[1:]
    all_nodes = []
    for node_info in nodes_desc:
        node = {}
        attr = node_info.split("\n")[1:]
        for e in attr:
            dkey = e.split(":")[0]
            dval = ":".join(e.split(":")[1:])
            if dkey in ["intersection_of", "relationship"]:
                dkey = dval.split()[0]
            if dkey in node:
                if isinstance(node[dkey], list):
                    node[dkey] = " ".join(node[dkey])
                dval = re.findall(r"\w+:\w+", node[dkey]) + re.findall(r"\w+:\w+", dval)
            if len(dval) > 0:
                if "!" in dval:
                    dval = re.findall(r"\w+:\w+", dval)
                node[dkey] = dval
        if node.get('id', "").strip().startswith("GO"):
            all_nodes.append(node)
    print("__download_gene_ontoloy__\nNb-GO terms: ", len(all_nodes), "\n")
    return all_nodes


# Build a Gene Ontology network
def build_gene_ontology_graph():
    nodes_attr = __download_gene_ontology()
    edges_types = ["is_a", "part_of", "positively_regulates", "has_part"]
    G = nx.DiGraph()
    for at in nodes_attr:
        source = at["id"].strip()
        for edge_type in edges_types:
            target = at.get(edge_type, [])
            for u in combinations([source] + target, 2):
                u += ({'relationship': edge_type},)
                G.add_edges_from([u])
    print("__build__go_graph__\nNb-links:", G.number_of_edges(), "\nNb-GoTerm:", G.number_of_nodes())
    return G


def __download_gene_go_terms():
    res = defaultdict(set)
    dc = DataCache()
    fin = open(dc.get_file("s3://datasets-ressources/DDI/drugbank/goa_human_isoform_valid.gaf"))
    for line in fin:
        if line.startswith("UniProtKB"):
            content = line.split("\t")
            geneid = content[2].lower()
            goid = content[4]
            res[geneid].add(goid)
    nb_go_terms = sum([len(r) for i, r in res.items()])
    print("__download_gene_go_terms__\nNb-Genes: {} \nGO terms associated: {}\n".format(len(res), nb_go_terms))
    return res


def __download_drug_gene_targets():
    g = ExternalDatabaseLoader.load_drugbank()
    g = g[pd.notnull(g['gene-name'])]
    drug_2_gene = defaultdict(set)
    drug_2_name = defaultdict(set)
    name_2_drug = defaultdict(set)
    for index, row in g.iterrows():
        drug_2_gene[row["drug_id"]].add(row["gene-name"].lower())
        drug_2_name[row["drug_id"]].add(row["drug_name"].lower())
        name_2_drug[row["drug_name"].lower()].add(row["drug_id"])
    return drug_2_gene, drug_2_name, name_2_drug


def __get_drug_info():
    a, b, c = __download_drug_gene_targets()
    d = __download_gene_go_terms()
    f = {}
    for drug_id, drug_gene_targets in a.items():
        drug_go_terms = set()
        for gene_id in drug_gene_targets:
            go_term = d.get(gene_id, None)
            if go_term:
                drug_go_terms = drug_go_terms.union(go_term)
        if drug_go_terms:
            f[drug_id] = drug_go_terms
    print("__get_drugs_info__\nNb drugs with nb go term >= 1: ", len(f),
          "\nNb-drugs with nb target >=1: ", len(a))
    return b, c, a, f


def build_fi_graph():
    G = nx.Graph()
    dc = DataCache()
    fin = open(dc.get_file("s3://datasets-ressources/DDI/drugbank/BIOGRID-ORGANISM-Homo_sapiens-3.5.176.tab.txt"))
    fin.readline()
    edges = set()
    for line in fin:
        content = line.split("\t")
        gene_a, gene_b = content[2].strip(" "), content[3].strip(" ")
        if len(gene_a) > 0 and len(gene_b) > 0:
            edges.add((gene_a.lower(), gene_b.lower()))
    G.add_edges_from(list(edges))
    print("__build_FI_network__\nNb-Genes: {}\nNb-links: {}\n".format(G.number_of_nodes(), G.number_of_edges()))
    return G


def __compute_drugs_similarity_score(G, p, a, b):
    a, b = list(a), list(b)
    ta = max([len(p[x][y]) if (x in p and y in p) else 0.0 for (x, y) in list(combinations(a, 2))] + [0])
    denom = [(x, y) for x in a for y in b if (x in G and y in G)]
    num = [(x, y) for (x, y) in denom if x in p and y in p]
    Sab = len([len(p[x][y]) if len(p[x][y]) <= ta else 0.0 for (x, y) in num]) / len(denom) if len(denom) > 0 else len(
        [len(p[x][y]) if p[x][y] <= ta else 0.0 for (x, y) in num]) / 1
    return Sab


def lee_et_al_transformer(drugs, smiles):
    fi_graph = build_fi_graph()
    go_graph = build_gene_ontology_graph()
    drug_2_iupac, iupac_2_drug, drugs_gene_targets, drugs_go_terms = __get_drug_info()

    # I need to check if all drugs info is available - change key = drug id --> drug neme
    # drugs_gene_targets_v2 = {list(drug_2_iupac[i])[-1]: v for i, v in drugs_gene_targets.items()}
    # drugs_go_terms_v2 = {list(drug_2_iupac[i])[-1]: v for i, v in drugs_go_terms.items()}

    # Get shortest path
    fshp = nx.shortest_path(fi_graph)
    gshp = nx.shortest_path(go_graph)
    print("__Get all shortest paths__ok")
    #  Remove drugs with no targets
    # TSP  + GSP Profile
    # print(drug_2_iupac)
    # exit()
    # drugs = [list(drug_2_iupac[u])[-1] for u in drugs]

    TSP = {x: [
        __compute_drugs_similarity_score(fi_graph, fshp, drugs_gene_targets[ya], drugs_gene_targets[yb]) for
        (ya, yb) in list(zip([x] * len(drugs), drugs))] for x in drugs if
        x in drugs_gene_targets and x in drugs_go_terms}
    GSP = {
        x: [
            __compute_drugs_similarity_score(go_graph, gshp, drugs_go_terms[ya], drugs_go_terms[yb]) for
            (ya, yb) in
            list(zip([x] * len(drugs), drugs)) if (ya in drugs_go_terms and yb in drugs_go_terms)] for x in drugs if
        x in drugs_gene_targets and x in drugs_go_terms}
    # SSP Profile
    mols = [Chem.MolFromSmiles(X) for X in smiles]
    fgps = [AllChem.GetMorganFingerprintAsBitVect(mol, 4) for mol in mols]
    SSP = {x: [DataStructs.FingerprintSimilarity(fgps[drugs.index(x)], fgps[i]) for i in range(len(fgps))] for
           j, x in enumerate(drugs)}

    print("__Lee_transformer__\nNb-drugs(inputs): ", len(drugs), "\nSSP length: ", len(SSP), "\nTSP length: ", len(TSP),
          "\nGSP length: ", len(TSP))

    return SSP, TSP, GSP


if __name__ == '__main__':
    p = pd.read_table("/home/rogia/.invivo/cache/datasets-ressources/DDI/drugbank/drugbank-drugs-all.csv", sep=",",
                      index_col=0)
    d = p["drug_id"].values.tolist()
    smi = p["PubChem Canonical Smiles"].values.tolist()
    assert len(d) == len(smi)
    lee_et_al_transformer(d, smi)
