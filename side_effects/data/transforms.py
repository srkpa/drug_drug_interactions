from operator import itemgetter

import numpy as np
import torch
import networkx as nx
from ivbase.transformers.features import DGLGraphTransformer
from ivbase.transformers.features.molecules import FingerprintsTransformer
from ivbase.transformers.features.molecules import SequenceTransformer
from ivbase.utils.constants.alphabet import SMILES_ALPHABET
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from ivbase.transformers.features import AdjGraphTransformer
from ivbase.utils.datasets.external_databases import ExternalDatabaseLoader
from ivbase.utils.datasets.datacache import DataCache
from collections import defaultdict
from itertools import combinations, product
from ivbase.transformers.features.targets import GeneEntityTransformer
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


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


def load_go_idmapping():
    res = defaultdict(set)
    dc = DataCache()
    fin = open(dc.get_file("s3://datasets-ressources/DDI/drugbank/goa_human_isoform_valid.gaf"))
    for line in fin:
        if line.startswith("UniProtKB"):
            content = line.split("\t")
            geneid = content[2].lower()
            goid = content[4]
            # go_refid = content[5]
            res[geneid].add(goid)  # go_refid
    nb_go_terms = sum([len(r) for i, r in res.items()])
    nb_genes = len(res)
    print("GO: {} genes and {} GO terms".format(nb_genes, nb_go_terms))
    return res


def load_biogrid_network():
    go_terms = load_go_idmapping()
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
    for node in G.nodes:
        G.nodes[node]["go_terms"] = go_terms[node]
    print("BioGRID: {} genes and {} interactions".format(G.number_of_nodes(), G.number_of_edges()))
    return G


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


def load_targets(drugids, targetid="gene-name"):
    g = ExternalDatabaseLoader.load_drugbank()
    g.set_index("drug_name", inplace=True)
    g.index = g.index.str.lower()
    res = {entry: g.loc[[entry], targetid].dropna().str.lower().values.tolist() for entry in drugids}
    return res


def deepddi_transformer(drugs, smiles, approved_drug):
    appr_drugs_mol = [Chem.MolFromSmiles(X) for X in approved_drug]
    appr_drugs_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 4) for mol in appr_drugs_mol]
    drugs_involved_in_mol = [Chem.MolFromSmiles(m) for m in smiles]
    drugs_involved_in_fps = [AllChem.GetMorganFingerprintAsBitVect(X, 4) for X in drugs_involved_in_mol]
    all_ssp = [[DataStructs.FingerprintSimilarity(drg, appr_drugs_fps[i]) for i in range(len(appr_drugs_fps))] for
               ids, drg in enumerate(drugs_involved_in_fps)]
    pca = PCA(n_components=50)
    out = pca.fit_transform(np.array(all_ssp)).astype(np.float32)
    # out = torch.from_numpy(reduced_ssp)
    # if torch.cuda.is_available():
    #     out = out.cuda()
    return dict(zip(drugs, out))


def _filter_gene_exists(network, geneids):
    to_remove = [gene_id for gene_id in geneids if gene_id not in network.nodes]
    return set(geneids) - set(to_remove)


def tsp_score(G, pairs_of_nodes):
    res = []
    for (gene_a, gene_b) in pairs_of_nodes:
        try:
            dist = nx.shortest_path_length(G, gene_a, gene_b)
        except nx.NetworkXNoPath:
            dist = 0
        res.append(dist)
    return res


def target_similarity_profile_transformer(drugs_ids):
    G = load_biogrid_network()
    targets = list(map(_filter_gene_exists, [G] * len(drugs_ids), load_targets(drugs_ids)))
    for index, entry in enumerate(drugs_ids):
        target = targets[index]
        if target:
            t_a = tsp_score(G, list(combinations(target, 2)))
            max_dist = max(t_a) if len(t_a) > 0 else 0
            # nodes = list(product(src, dst))

            print(entry, max_dist)


def dgl_transformer(drugs, smiles):
    assert len(drugs) == len(smiles)
    trans = DGLGraphTransformer()
    X, ids = trans(smiles)  # Call the transformer on the smiles using the __call__ method.
    # for index, g in enumerate(X):
    #     if len(g.edata["he"]) == 0:
    #         del X[index]
    #         del ids[index]

    drugs = list(itemgetter(*ids)(drugs)) if len(ids) < len(drugs) else drugs
    assert len(X) == len(ids)
    assert len(X) == len(drugs)
    return dict(zip(drugs, X))


def adj_transformer(drugs, smiles):
    trans = AdjGraphTransformer()
    X, ids = trans(smiles, dtype=np.float32)
    drugs = list(itemgetter(*ids)(drugs))
    return dict(zip(drugs, X))


if __name__ == '__main__':
    b = load_targets()
    print(b)
