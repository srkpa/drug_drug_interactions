"""
    Downloading QM9/ DECAGON data to args.path (set by default to be "./data/")
    command: python data_download QM9 DECAGON -p path
"""
import tarfile
import zipfile

import wget

from .utils.file_utils import *


def download_decagon_data(dir_path='./data/'):
    """
    Step 0: Download Polypharmacy data
    wget http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz;
    tar -xvzf bio-decagon-combo.tar.gz;

    bio-decagon-combo.csv which has the form
    drug_CID, drug_CID, side_effect_id, side_effect_name
    ( == CIDXXXXX,CIDXXXX,CXXXX,side_effect_name).

    Step 1: Collect drug cid list in drug_raw_feat.idx.jsonl,
    which has the form:
        CIDXXXX : { "atoms": [
                        {"aid": x, "number": x, "x": x, "y": x}
                        ... ]
                    "bonds": [
                        {"aid1": x, "aid2": x, "order": x}
                        ... ]
                    }
    where
        atom.aid is the atom ID within the owning Compound (molecule)
        atom.number is the atomic number for this atom
        atom.x, atom.y are the coordinates

        bond.aid1, bond.aid2 are the begin and end atom of the bond
        bond.order is the (chemical) bond order
    For more details
        https://pubchempy.readthedocs.io/en/latest/api.html#pubchempy.Atom
    """
    prepare_data_dir(dir_path)

    import csv
    drug_idx = set()
    with open(dir_path + '/bio-decagon-combo.csv') as f:
        csv_rdr = csv.reader(f)
        for i, row in enumerate(csv_rdr):
            if i == 0:
                print('Header:', row)
            else:
                drug1, drug2, *_ = row
                drug_idx |= {drug1, drug2}
    print('Instance:', row)

    print('Unique drug count =', len(drug_idx))

    # # Step 2: Search on PubChem
    from tqdm import tqdm_notebook
    import pubchempy as pcp
    # Use int type cid to search with PubChemPy
    drugs = {cid: pcp.Compound.from_cid(int(cid.strip('CID')))
             for cid in tqdm_notebook(drug_idx)}

    # # Step 3: Write to file
    import json
    with open(dir_path + '/drug_raw_feat.idx.jsonl', 'w') as f:
        for cid, drug in drugs.items():
            drug = drug.to_dict(properties=['atoms', 'bonds'])
            f.write('{}\t{}\n'.format(cid, json.dumps(drug)))


def download_qm9_data(data_dir):
    """
        Downloading and extracting necessary data.
    """
    data_dir = os.path.join(data_dir, 'qm9')
    if os.path.exists(data_dir):
        print('Found QM9 dataset - SKIP!')
        return

    prepare_data_dir(data_dir)

    # Download data from figshare
    def download_figshare(file_name, file_ext, dir_path='./data/',
                          change_name=None):
        prepare_data_dir(dir_path)
        url = 'https://ndownloader.figshare.com/files/' + file_name
        wget.download(url, out=dir_path)
        file_path = os.path.join(dir_path, file_name)

        if file_ext == '.zip':
            zip_ref = zipfile.ZipFile(file_path, 'r')
            if change_name is not None:
                dir_path = os.path.join(dir_path, change_name)
            zip_ref.extractall(dir_path)
            zip_ref.close()
            os.remove(file_path)
        elif file_ext == '.tar.bz2':
            tar_ref = tarfile.open(file_path, 'r:bz2')
            if change_name is not None:
                dir_path = os.path.join(dir_path, change_name)
            tar_ref.extractall(dir_path)
            tar_ref.close()
            os.remove(file_path)
        elif change_name is not None:
            os.rename(file_path, os.path.join(dir_path, change_name))

    # README
    download_figshare('3195392', '.txt', data_dir, 'readme.txt')
    # atomref
    download_figshare('3195395', '.txt', data_dir, 'atomref.txt')
    # Validation
    download_figshare('3195401', '.txt', data_dir, 'validation.txt')
    # Uncharacterized
    download_figshare('3195404', '.txt', data_dir, 'uncharacterized.txt')
    # dsgdb9nsd.xyz.tar.bz2
    download_figshare('3195389', '.tar.bz2', data_dir, 'dsgdb9nsd')
    # dsC7O2H10nsd.xyz.tar.bz2
    download_figshare('3195398', '.tar.bz2', data_dir, 'dsC702H10nsd')


def main(path, dataset_name):
    prepare_data_dir(path)
    if dataset_name == 'qm9':
        download_qm9_data(path)
    if dataset_name == 'twosides':
        download_decagon_data(path)
