import glob
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
from lxml import etree
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA


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
    # m = dict(zip(drugs, drugs_involved_in_mol))
    # pk.dump(m, open("save.p", "wb"))
    # exit()
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
    trans = module()
    X, ids = trans(smiles)
    drugs = list(itemgetter(*ids)(drugs)) if len(ids) < len(drugs) else drugs
    print(trans.n_atom_feat)
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


def download_drug_gene_targets():
    g = ExternalDatabaseLoader.load_drugbank()
    g = g[pd.notnull(g['gene-name'])]
    print(g)
    print(list(g))
    exit()
    drug_2_gene = defaultdict(set)
    drug_2_name = {}
    name_2_drug = {}
    for index, row in g.iterrows():
        drug_2_gene[row["drug_id"]].add(row["gene-name"].lower())
        drug_2_name[row["drug_id"]] = row["drug_name"].lower()
        name_2_drug[row["drug_name"].lower()] = row["drug_id"]
    return drug_2_gene, drug_2_name, name_2_drug


def __get_drug_info():
    a, b, c = download_drug_gene_targets()
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
    a = [G.index(x) for x in a if x in G]
    b = [G.index(x) for x in b if x in G]
    print(a, b)
    # exit()
    # ta = max([p[x][y] if (x in p and y in p) else 0.0 for (x, y) in list(combinations(a, 2))] + [0])
    # denom = [(x, y) for x in a for y in b if (x in G and y in G)]
    # num = [(x, y) for (x, y) in denom if x in p and y in p]
    # Sab = len([len(p[x][y]) if len(p[x][y]) <= ta else 0.0 for (x, y) in num]) / len(denom) if len(denom) > 0 else len(
    #     [len(p[x][y]) if p[x][y] <= ta else 0.0 for (x, y) in num]) / 1
    ta = max([p[x][y] for (x, y) in list(combinations(a, 2))] + [0])
    pairs = [(x, y) for x in a for y in b]
    Sab = len([p[x][y] for (x, y) in pairs if p[x][y] <= ta]) / len(pairs) if len(pairs) > 0 else len(
        [p[x][y] for (x, y) in pairs if p[x][y] <= ta]) / 1
    return Sab


def lee_et_al_transformer(drugs, smiles):
    fi_graph = build_fi_graph()
    go_graph = build_gene_ontology_graph()
    drug_2_iupac, iupac_2_drug, drugs_gene_targets, drugs_go_terms = __get_drug_info()

    # I prefer to use drug name as id for more consistency
    num_drugs = len(drugs)
    drug_names = drugs

    # Remove drugs with no target and no go terms
    assert len(drug_names) == len(smiles)
    smiles = [smiles[i] for i, j in enumerate(drug_names) if j in drugs_gene_targets and j in drugs_go_terms]
    drug_names = [j for j in drug_names if j in drugs_gene_targets and j in drugs_go_terms]

    # Get adjacency mat
    fi_mat = nx.adjacency_matrix(fi_graph)
    print("F Mat shape", fi_mat.shape)
    go_mat = nx.adjacency_matrix(go_graph)
    print("G Mat shape", go_mat.shape)

    # Get shortest path
    f_nodes = [x for x in fi_graph.nodes]
    g_nodes = [x for x in go_graph.nodes]
    fi_graph = csr_matrix(fi_mat)
    go_graph = csr_matrix(go_mat)
    fshp = shortest_path(csgraph=fi_graph, directed=False, indices=None,
                         return_predecessors=False)  # nx.floyd_warshall(fi_graph) #nx.shortest_path(fi_graph)  # .subgraph([i for j, i in enumerate(fi_graph.nodes) if j <= 5]
    gshp = shortest_path(csgraph=go_graph, directed=True, indices=None,
                         return_predecessors=False)  # nx.floyd_warshall(go_graph)#nx.shortest_path(go_graph)  # .subgraph([i for j, i in enumerate(go_graph.nodes) if j <= 5])
    print(fshp.shape, gshp.shape, len(f_nodes), len(g_nodes))
    print("__Get all shortest paths__ok")

    # TSP  + GSP
    TSP = {x: [
        __compute_drugs_similarity_score(f_nodes, fshp, drugs_gene_targets[ya], drugs_gene_targets[yb]) for
        (ya, yb) in list(zip([x] * len(drug_names), drug_names))] for x in drug_names if
        x in drugs_gene_targets and x in drugs_go_terms}
    GSP = {
        x: [
            __compute_drugs_similarity_score(g_nodes, gshp, drugs_go_terms[ya], drugs_go_terms[yb]) for
            (ya, yb) in
            list(zip([x] * len(drug_names), drug_names)) if (ya in drugs_go_terms and yb in drugs_go_terms)] for x in
        drug_names if
        x in drugs_gene_targets and x in drugs_go_terms}
    # SSP Profile
    mols = [Chem.MolFromSmiles(j) for j in smiles]
    fgps = [AllChem.GetMorganFingerprintAsBitVect(mol, 4) for mol in mols]
    SSP = {x: [DataStructs.FingerprintSimilarity(fgps[drug_names.index(x)], fgps[i]) for i in range(len(fgps))] for
           j, x in enumerate(drug_names)}

    print("__Lee_transformer__\nNb-drugs(inputs): ", num_drugs, "\nSSP length: ", len(SSP), "\nTSP length: ",
          len(TSP),
          "\nGSP length: ", len(TSP))
    print(len(list(SSP.values())[0]))
    print(len(list(TSP.values())[0]))
    print(len(list(GSP.values())[0]))
    output = {x: (
        np.array(SSP[x]).astype(np.float32), np.array(TSP[x]).astype(np.float32), np.array(GSP[x]).astype(np.float32))
        for x
        in drug_names}
    return output


def parse_drugbank(fp="../../data/"):
    fin = {}
    for xmlfile in glob.glob(f"{fp}/*-*.xml"):
        print(xmlfile)
        with open(xmlfile) as f:
            tree = etree.parse(f)
            root = tree.getroot()
            for d in root.iter():
                for elem in d.getchildren():
                    for k in elem.getchildren():
                        if elem.tag not in fin or fin[elem.tag] is None:
                            fin[elem.tag] = {}
                        if k.tag not in fin[elem.tag]:
                            fin[elem.tag][k.tag] = []
                        fin[elem.tag][k.tag] += [k.text]
                if d.tag not in fin:
                    fin[d.tag] = d.text

        print(fin)

if __name__ == '__main__':
    # p = pd.read_table("/home/rogia/.invivo/cache/datasets-ressources/DDI/drugbank/drugbank-drugs-all.csv", sep=",",
    #                   index_col=0)
    # d = p["drug_id"].values.tolist()
    # smi = p["PubChem Canonical Smiles"].values.tolist()
    # assert len(d) == len(smi)
    # o = lee_et_al_transformer(d, smi)
    # download_drug_gene_targets()
    parse_drugbank()
