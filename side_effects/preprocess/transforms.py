import torch
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from ivbase.transformers.features.molecules import SequenceTransformer
from ivbase.utils.constants.alphabet import SMILES_ALPHABET

from ivbase.transformers.features.molecules import FingerprintsTransformer


def fingerprints_transformer(drugs, smiles):
    transformer = FingerprintsTransformer()
    out = transformer.transform(smiles)
    res = np.stack(out)
    res = torch.from_numpy(res)
    print("fingerprint vectorized out", res.shape)
    return dict(zip(drugs, res))


def sequence_transformer(smiles, drugs, one_hot=False):
    transformer = SequenceTransformer(alphabet=SMILES_ALPHABET, one_hot=one_hot)
    out = transformer.fit_transform(smiles)
    out = torch.from_numpy(out)
    print("Sequence vectorized out", out.shape)
    return dict(zip(drugs, out))


def deepddi_transformer(smiles, drugs, approved_drug):
    appr_drugs_mol = [Chem.MolFromSmiles(X) for X in approved_drug]
    appr_drugs_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 4) for mol in appr_drugs_mol]
    drugs_involved_in_mol = [Chem.MolFromSmiles(m) for m in smiles]
    drugs_involved_in_fps = [AllChem.GetMorganFingerprintAsBitVect(X, 4) for X in drugs_involved_in_mol]
    all_ssp = [[DataStructs.FingerprintSimilarity(drg, appr_drugs_fps[i]) for i in range(len(appr_drugs_fps))] for
               ids, drg in enumerate(drugs_involved_in_fps)]
    pca = PCA(n_components=50)
    reduced_ssp = pca.fit_transform(np.array(all_ssp))

    return dict(zip(drugs, torch.from_numpy(reduced_ssp)))

