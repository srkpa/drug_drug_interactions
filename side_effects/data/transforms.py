from operator import itemgetter

import numpy as np
import torch
from ivbase.transformers.features import DGLGraphTransformer
from ivbase.transformers.features.molecules import FingerprintsTransformer
from ivbase.transformers.features.molecules import SequenceTransformer
from ivbase.utils.constants.alphabet import SMILES_ALPHABET
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA


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
    # out = torch.from_numpy(reduced_ssp)
    # if torch.cuda.is_available():
    #     out = out.cuda()
    return dict(zip(drugs, out))


def dgl_transformer(drugs, smiles):
    assert len(drugs) == len(smiles)
    trans = DGLGraphTransformer()  # Initialize the transformer
    X, ids = trans(smiles)  # Call the transformer on the smiles using the __call__ method.
    # Keep only the ids where the transformation succeeded
    # (the failed transformations are not present in ids)

    for index, g in enumerate(X):
        if len(g.edata["he"]) == 0:
            del X[index]
            del ids[index]

    drugs = list(itemgetter(*ids)(drugs)) if len(ids) < len(drugs) else drugs
    print("Dgl transformer ---> {len(drugs}")
    assert len(X) == len(ids)
    assert len(X) == len(drugs)
    return dict(zip(drugs, X))
