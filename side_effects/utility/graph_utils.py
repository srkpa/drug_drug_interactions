import io
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torchvision import transforms
from torchvision.utils import make_grid

from side_effects.utility import const

TRANSFORMER = transforms.ToTensor()
EDGE_DECODER = const.BOND_TYPES


def to_cuda(x):
    if const.CUDA_OK:
        return x.cuda()
    return x


def power_iteration(mat, n_iter=100):
    res = torch.ones(mat.shape[1], 1)
    if torch.cuda.is_available():
        res = res.cuda()
    for i in range(n_iter):
        res = mat.mm(res) / res.norm()
    return res


def find_largest_eigval(mat, n_iter=100):
    res = power_iteration(mat, n_iter=n_iter)
    # use the Rayleigh quotient.
    return torch.matmul(res.t(), mat).matmul(res) / torch.matmul(res.t(), res)


def restack(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, (tuple, list)):
        return torch.stack([x[i].squeeze() for i in range(len(x))], dim=0)
    raise ValueError("Cannot re-stack tensor")


def pad_graph(adj, max_num_node=None):
    """Returned a padded adj from an input adj"""
    num_node = adj.shape[0]
    old_shape = (adj.shape)
    if max_num_node in [-1, None]:
        return adj
    elif max_num_node < num_node:
        return adj[:max_num_node, :max_num_node]
    else:
        return F.pad(adj.transpose(-1, 0), (0, max_num_node - num_node) * 2, mode='constant', value=0).transpose(-1, 0)


def pad_feats(feats, no_atom_tensor=None, max_num_node=None):
    """Returned a padded adj from an input adj"""
    num_node = feats.shape[0]
    if max_num_node in [-1, None]:
        return feats
    elif max_num_node < num_node:
        return feats[:max_num_node, :max_num_node]
    else:
        if no_atom_tensor is None:
            no_atom_tensor = feats.new_zeros(feats.shape[-1])
        return torch.cat((feats, no_atom_tensor.unsqueeze(dim=0).expand(max_num_node - num_node, -1)), dim=0)


def get_adj_vector(adj):
    """Return a vector containing the upper triangle of an adj mat"""
    return adj[torch.ones_like(adj).triu().byte()]


def deg_feature_similarity(f1, f2):
    return 1 / (abs(f1 - f2) + 1)


def _add_index_to_mol(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber',
                                        str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def mol2im(mc, molSize=(300, 120), kekulize=False, outfile=None, im_type="png"):
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            pass
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    if im_type == "svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])

    drawer.SetFontSize(drawer.FontSize() * 0.8)
    opts = drawer.drawOptions()
    for i in range(mc.GetNumAtoms()):
        opts.atomLabels[i] = mc.GetAtomWithIdx(i).GetSymbol() + str(i)
    opts.padding = 0.1
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()

    im = drawer.GetDrawingText()

    if im_type == "svg":
        im = im.replace('svg:', '')
        if outfile:
            with open(outfile, "w") as OUT:
                OUT.write(im)
    else:
        if outfile:
            drawer.WriteDrawingText(outfile)
    return im


def mol2svg(*args, **kwargs):
    return mol2im(*args, **kwargs, im_type="svg")


def mol2nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def add_index_to_mol(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def adj2nx(adj):
    return nx.from_numpy_matrix(adj.detach().cpu().numpy())


def decode_one_hot(encoding, fail=-1):
    if sum(encoding) == 0:
        return fail
    return np.argmax(encoding)


def data2mol(data, atom_list):
    atom_decoder = np.asarray(atom_list + ["*"])
    adj, x, *x_add = data
    adj = restack(adj)
    x = restack(x)
    mols = []
    n_mols = adj.shape[0]
    for i in range(n_mols):
        mols.append(adj2mol(adj[i].cpu().detach().long().numpy(), x[i].cpu().detach().long().numpy(),
                            atom_decoder=atom_decoder))
    return mols


def adj2mol(adj_mat, atom_one_hot, atom_decoder, edge_decoder=EDGE_DECODER):
    mol = Chem.RWMol()
    atoms = [atom_decoder[decode_one_hot(atom)] for atom in atom_one_hot]
    n_atoms = adj_mat.shape[0]
    # contains edges, so get argmax on the edge
    edge_type = np.argmax(adj_mat, axis=-1)
    adj_mat = np.sum(adj_mat, axis=-1)
    edges = np.triu(adj_mat, 1).nonzero()
    # print(atoms)
    accepted_atoms = {}
    for i, atom in enumerate(atoms):
        a = None
        if atom not in [None, "*"]:
            a = mol.AddAtom(Chem.Atom(atom))
        accepted_atoms[i] = a

    for start, end in zip(*edges):
        start = accepted_atoms.get(start)
        end = accepted_atoms.get(end)

        if (start and end):
            btype = EDGE_DECODER[edge_type[start, end]]
            mol.AddBond(int(start), int(end), btype)
    try:
        Chem.SanitizeMol(mol)
    except:
        pass

    mol = mol.GetMol()
    try:
        smiles = Chem.MolToSmiles(mol)
    except:
        mol = None
    if mol and mol.GetNumAtoms() > 0:
        return mol
    return None


def convert_mol_to_smiles(mollist):
    smiles = []
    for mol in mollist:
        smiles.append(Chem.MolToSmiles(mol))
    return smiles


def sample_gumbel(x, eps=1e-10):
    noise = torch.rand_like(x)
    noise = -torch.log(-torch.log(noise + eps) + eps)
    return noise


def gumbel_sigmoid(logits, temperature):
    noise = sample_gumbel(logits)
    y = (logits + noise) / temperature
    y = F.sigmoid(y)  # , dim=-1)
    return y


def convert_to_grid(mols, molSize=(350, 200)):
    tensor_list = []
    for mol in mols:
        if mol:
            im = mol2im(mol, molSize=molSize)
            im = Image.open(io.BytesIO(im))
            tensor_list.append(TRANSFORMER(im.convert('RGB')))
    if tensor_list:
        return make_grid(torch.stack(tensor_list, dim=0), nrow=4)
    return None


def sample_sigmoid(logits, temperature=1, sample=True, thresh=0.5, gumbel=True, hard=False):
    y_thresh = torch.ones_like(logits) * thresh
    if gumbel:
        y = gumbel_sigmoid(logits, temperature)
    else:
        y = torch.sigmoid(logits)
        if sample:
            y_thresh = torch.rand_like(logits)
    if hard:
        return torch.gt(y, y_thresh).float()
    return y


if __name__ == '__main__':
    pass
