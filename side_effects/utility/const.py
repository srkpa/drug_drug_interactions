import torch
from rdkit import Chem

CUDA_OK = False
if torch.cuda.is_available():
    CUDA_OK = True

CHEMBL_ATOM_LIST = ['O', 'C', 'F', 'Cl', 'Br', 'P', 'I', 'S', 'N']
TOX21_ATOM_LIST = ['Sb', 'Se', 'Pd', 'Sn', 'Si', 'K', 'S', 'Mg', 'Be', 'Ge', 'Fe', 'Zn', 'Bi', 'Br', 'P', 'Mo', 'As',
                   'Pb', 'Cu', 'Ca', 'Pt', 'Na', 'Ni', 'Ag', 'C', 'Nd', 'Cl', 'F', 'Ba', 'V', 'I', 'Sr', 'Dy', 'Yb',
                   'H', 'Co', 'Cr', 'Al', 'Gd', 'Tl', 'O', 'Zr', 'B', 'In', 'Li', 'Ti', 'Au', 'Mn', 'Hg', 'Cd', 'N']

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
