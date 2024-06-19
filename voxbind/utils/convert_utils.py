from copy import deepcopy
import numpy as np
from rdkit import Chem

from voxbind.utils.base_utils import supress_stdout
from voxbind.utils import reconstruct
from voxbind.constants import ATOM_ELEMENTS, CHANNEL_TO_ATM_NB_CROSSDOCKED


def mol2xyz(mol: dict) -> str:
    """
    Convert a molecular representation to XYZ format.

    Args:
        mol (dict): A dictionary containing the molecular representation.

    Returns:
        str: The molecular representation in XYZ format.
    """
    mask = mol['atoms_channel'] != 999.
    n_atoms = mol['atoms_channel'][mask].shape[-1]
    xyz_str = str(n_atoms) + "\n\n"
    for i in range(n_atoms):
        element = mol['atoms_channel'][mask][i]
        element = ATOM_ELEMENTS[int(element.item())]

        coords = mol['coords'][0, i, :]

        line = element + "\t" + str(coords[0].item()) + "\t" + str(coords[1].item()) + "\t" + str(coords[2].item())
        xyz_str += line + "\n"
    return xyz_str


# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
@supress_stdout
def mol2rdkit_obabel(
    mol: dict,
    sdf_file: str = None
) -> Chem.rdchem.Mol:
    """
    Convert a molecule dictionary to an RDKit molecule using Open Babel.
    This function uses Open Babel to convert the molecule dictionary to an RDKit molecule.
    It reconstructs the molecule using the provided coordinates and atom types.
    If the reconstruction fails, it returns None.

    Args:
        mol (dict): The molecule dictionary containing coordinates, atom types, and other information.
        sdf_file (str, optional): The path to the SDF file. Defaults to None.

    Returns:
        rdkmol (RDKit molecule): The RDKit molecule representation of the input molecule.

    Raises:
        reconstruct.MolReconsError: If molecule reconstruction fails.
    """
    try:
        pred_aromatic = None  # transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
        pred_pos = [[m[0].item(), m[1].item(), m[2].item()] for m in mol["coords"][0]]
        pred_atom_type = [CHANNEL_TO_ATM_NB_CROSSDOCKED[m.item()] for m in mol["atoms_channel"][0].int()]
        rdkmol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        rdkmol = remove_atoms_too_close_rdkit(rdkmol)
        smiles = Chem.MolToSmiles(rdkmol)
        # print("obabel SMILES:", smiles)
        if "." in smiles:
            return None
        return rdkmol
    except reconstruct.MolReconsError:
        print('Reconstruct failed %s' % f'{sdf_file}')
        return None


def remove_atoms_too_close_rdkit(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """
    Removes atoms that are too close to each other in a molecule using RDKit.

    Args:
        mol (Chem.rdchem.Mol): The input molecule.

    Returns:
        Chem.rdchem.Mol: The modified molecule with close atoms removed.
    """
    try:
        dists = Chem.rdmolops.Get3DDistanceMatrix(mol)
        row, col = np.where((dists >= 0) & (dists < .4) & ~np.eye(dists.shape[0], dtype=bool))
        if len(row) == 0:
            return mol
        skip_idx = []
        mol_ = deepcopy(mol)
        for i, idx in enumerate(row):
            if idx in skip_idx:
                continue
            skip_idx.append(col[i])
            edit_mol = Chem.EditableMol(mol_)
            edit_mol.RemoveAtom(int(idx))
            mol_ = edit_mol.GetMol()
        return mol_
    except Exception:
        return mol
