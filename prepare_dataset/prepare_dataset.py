import os
from random import shuffle, seed

from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from PyFingerprint.All_Fingerprint import get_fingerprint

from SubGNN import config
from . import config_prepare_dataset as pdconfig


# allowable node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_aromatic_list": [True, False],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def get_pubchem_fingerprint(smiles):
    fp = get_fingerprint(smiles, fp_type="pubchem", output="vector")
    return list(map(str, list(fp)))


def find_subgraph_and_label(mol, smiles, patterns):
    subgraphs = dict()
    for i, pat in enumerate(patterns):
        matches = mol.GetSubstructMatches(pat)
        if len(matches) > 0:
            subgraphs[i] = matches
    label = get_pubchem_fingerprint(smiles)
    return subgraphs, label


def find_edge_list(mol):
    edge_list = list()
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_list.append(f"{start} {end}")
    return edge_list


def generate_node_feat(mol):
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = (
            [allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())]
            + [allowable_features["possible_degree_list"].index(atom.GetDegree())]
            + [
                allowable_features["possible_formal_charge_list"].index(
                    atom.GetFormalCharge()
                )
            ]
            + [
                allowable_features["possible_hybridization_list"].index(
                    atom.GetHybridization()
                )
            ]
            + [allowable_features["possible_aromatic_list"].index(atom.GetIsAromatic())]
            + [allowable_features["possible_chirality_list"].index(atom.GetChiralTag())]
        )
        atom_features_list.append(atom_feature)
    return atom_features_list


def write_to_subgraphs(
    edgelist,
    atom_ids,
    subgraphs,
    label,
    smiles,
    atom_features,
    path=None,
    starting_idx=0,
    split="train",
    method="a",
):
    """ Write edgelist to path/edge_list.txt. Write subgraphs to path/subgraphs.pth.

    Args:
        edgelist (iterable): each item has a str with two node numbers seperated by
            space.
        subgraphs (dict): dict with subgraph label as keys and subgraph nodes lists as
            values.
        path (str): root path to save the files. If not set, will save the files to
            current location.
        starting_idx (int): the starting index of the nodes. All node numbers will add
            this index. Default is 0.
        split (str): train, test, or val. Default is train.
        method (str): writing method to the files. 'w': Truncate file to zero length or
            create text file for writing. The stream is positioned at the beginning of
            the file. 'a': Open for writing.  The file is created if it does not exist.
            The stream is positioned at the end of the file.
    """
    assert split in [
        "train",
        "test",
        "val",
    ], f"Value of split must in ['train', 'test', 'val'], now is {split}"
    if path is None:
        path = "."
    else:
        os.makedirs(path, exist_ok=True)
    edge_list_f = open(os.path.join(path, "edge_list.txt"), method)
    for edge in edgelist:
        start, end = map(lambda x: int(x) + starting_idx, edge.split())
        edge_list_f.write(f"{start} {end}\n")
    edge_list_f.close()

    subgraphs_f = open(os.path.join(path, "subgraphs.pth"), method)
    atom_ids = [str(id + starting_idx) for id in atom_ids]
    subgraphs_f.write("-".join(atom_ids) + "\t")
    subgraphs_f.write("-".join(label) + "\t")
    for value in subgraphs.values():
        for subgraph in value:
            subgraphs_f.write(
                "-".join(map(lambda x: str(x + starting_idx), subgraph)) + "\t"
            )
    subgraphs_f.write(smiles)
    subgraphs_f.write("\n")
    subgraphs_f.close()

    atom_features_f = open(os.path.join(path, "atom_features.pth"), method)
    for feat in atom_features:
        atom_features_f.write(" ".join(map(str, feat)) + "\n")
    atom_features_f.close()


def prepare_dataset(ser):
    """ Get some dataset parameters for generating subgraphs.

    Args:
        ser (pandas Series): the Pandas Series containing the SMILES.

    Returns:
        n_samples (int): length of the input.
        train_indices (list): indices of the training samples. Splitting ratio: 8-1-1.
        val_indices (list): indices of the validation samples.
        test_indices (list): indices of the testing samples.
    """
    n_samples = len(ser)
    indices = list(range(n_samples))
    shuffle(indices)
    train_indices = indices[: int(n_samples * 0.8)]
    val_indices = indices[int(n_samples * 0.8) : int(n_samples * 0.9)]
    test_indices = indices[int(n_samples * 0.9) :]
    return n_samples, train_indices, val_indices, test_indices


def to_subgraphs(patterns, ser, save_path):
    """ Save subgraphs.

    Args:
        patters (list): list of rdkit Mol objects generated from SMARTS.
        ser (pandas Series object): pandas Serties with SMILES.
        save_path (str): path to the saving root.
    """
    n_samples, train_indices, val_indices, _ = prepare_dataset(ser)
    starting_idx = 0
    for i, smi in tqdm(enumerate(ser), total=n_samples):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        edgelist = find_edge_list(mol)
        subgraphs, label = find_subgraph_and_label(mol, smi, patterns)
        atom_ids = [a.GetIdx() for a in mol.GetAtoms()]
        atom_features = generate_node_feat(mol)
        if i in train_indices:
            split = "train"
        elif i in val_indices:
            split = "val"
        else:
            split = "test"
        if i == 0:
            method = "w"
        else:
            method = "a"
        write_to_subgraphs(
            edgelist,
            atom_ids,
            subgraphs,
            label,
            smi,
            atom_features,
            save_path,
            starting_idx,
            split=split,
            method=method,
        )
        starting_idx += mol.GetNumAtoms()


def main():
    seed(pdconfig.RANDOM_SEED)
    smiles = pd.read_csv(pdconfig.DATASET_DIR / "smiles.csv")["smiles"]
    patterns_df = pd.read_csv(config.PATTERN_PATH)
    patterns = [Chem.MolFromSmarts(sm) for sm in patterns_df.SMARTS]
    to_subgraphs(patterns, smiles, pdconfig.DATASET_DIR)


if __name__ == "__main__":
    main()
