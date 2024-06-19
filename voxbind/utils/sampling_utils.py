import os
from copy import deepcopy
from rdkit import Chem
import shutil
import torch
from omegaconf import OmegaConf
from pyuul import utils
from rdkit import RDLogger

from voxbind.utils.convert_utils import mol2rdkit_obabel
from voxbind.utils.base_utils import makedir
from voxbind.utils.base_utils import load_checkpoint
from voxbind.models import create_model
from voxbind.voxelizer import Voxelizer
from voxbind.utils.dataset_utils import (
    recenter_structures, pad, rotate_coords, atomChannelsToRadius, filter_atoms_by_distance
)
from voxbind.constants import ELEMENTS_HASH_CROSSDOCKED

RDLogger.logger().setLevel(RDLogger.CRITICAL)
RDLogger.DisableLog('rdApp.info')


def sample_molecules(
    model: torch.nn.Module,
    pocket: dict,
    ligand_gt: dict,
    voxelizer: torch.nn.Module,
    target_dirname: str,
    cfg: dict
) -> int:
    """
    Sample molecules using a generative model.

    Args:
        model (torch.nn.Module): The generative model used for sampling.
        pocket (dict): The pocket information.
        ligand_gt (dict): The ground truth ligand information.
        voxelizer (torch.nn.Module): The voxelizer used for converting ligands and pockets to voxels.
        target_dirname (str): The directory where the generated molecules will be saved.
        cfg (dict): Configuration parameters for the sampling process.

    Returns:
        int: The number of valid molecules generated.
    """
    makedir(target_dirname)
    ligands_gt, pockets = make_batch(ligand_gt, pocket, 1)
    ligands_gt_vox = voxelizer.forward(ligands_gt, num_channels=7)
    pockets_vox = voxelizer.forward(pockets, num_channels=4)

    n_valid_mol, n_mol = 0, 0
    rdkmols, list_smiles = [], []
    while n_valid_mol < cfg.wjs.n_samples_per_pocket and n_mol < 500:
        # sample molecules
        gen_vox_mols = model.sample(
            pocket=pockets_vox,
            ligand=ligands_gt_vox,
            warmup_wjs=cfg.wjs.warmup,
            steps=cfg.wjs.steps,
            max_steps=cfg.wjs.max_steps,
            chain_init=cfg.wjs.chain_init,
            mask_pocket=cfg.wjs.mask_pocket > 0,
            n_chains=cfg.wjs.n_samples_per_pocket,
        )
        mols = voxelizer.vox2mol(gen_vox_mols, center_coords=pocket["center_coords"])

        if mols is None:
            break
        for mol in mols:
            sdf_fname = os.path.join(target_dirname, f"sample_{n_valid_mol:03d}.sdf")
            mol = mol2rdkit_obabel(mol, sdf_fname)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if smiles not in list_smiles:
                    list_smiles.append(smiles)
                    n_valid_mol += 1
                    rdkmols.append(mol)
            n_mol += 1

    if n_valid_mol == 0:
        return 0

    # merge into single SDF file and delete individual files
    with Chem.SDWriter(os.path.join(target_dirname, "samples.sdf")) as writer:
        for rdkmol in rdkmols[:cfg.wjs.n_samples_per_pocket]:
            try:
                writer.write(rdkmol)
            except Exception:
                print("cannot save", rdkmol)

    save_pocket_and_ligand(ligand_gt, pocket, cfg.dset.data_dir, target_dirname)

    return n_valid_mol


def make_batch(
    ligand_gt: dict,
    pocket: dict,
    n_samples: int,
    rotate=False
) -> dict:
    """Create a batch of ligands and pockets.

    Args:
        ligand_gt (dict): The ground truth ligand dictionary.
        pocket (dict): The pocket dictionary.
        n_samples (int): The number of samples to create in the batch.
        rotate (bool, optional): Whether to rotate the coordinates. Defaults to False.

    Returns:
        dict: A dictionary containing the ligands and pockets in the batch.
    """
    ligands, pockets = [], []
    for i in range(n_samples):
        ligand_, pocket_ = deepcopy(ligand_gt), deepcopy(pocket)
        if rotate:
            ligand_, pocket_ = rotate_coords(ligand_, pocket_)
        ligands.append(ligand_)
        pockets.append(pocket_)
    ligands = {
        "coords": torch.concat([ligand["coords"] for ligand in ligands]),
        "atoms_channel": torch.concat([ligand["atoms_channel"] for ligand in ligands]),
        "radius": torch.concat([ligand["radius"] for ligand in ligands]),
    }
    pockets = {
        "coords": torch.concat([pocket["coords"] for pocket in pockets]),
        "atoms_channel": torch.concat([pocket["atoms_channel"] for pocket in pockets]),
        "radius": torch.concat([pocket["radius"] for pocket in pockets]),
    }
    return ligands, pockets


def save_pocket_and_ligand(
    ligand_gt,
    pocket,
    data_dir,
    dirname
) -> None:
    """
    Save the ligand and pocket files to the specified directory.

    Args:
        ligand_gt (dict): The ligand ground truth.
        pocket (dict): The pocket information.
        data_dir (str): The directory containing the data files.
        dirname (str): The directory to save the ligand and pocket files.

    Returns:
        None
    """
    ligand_id, pocket_id = ligand_gt["id"][0], pocket["id"][0],
    shutil.copyfile(
        os.path.join(data_dir, "crossdocked_pocket10", ligand_id),
        os.path.join(dirname, ligand_id.replace("/", "__"))
    )
    shutil.copyfile(
        os.path.join(data_dir, "crossdocked_pocket10", pocket_id),
        os.path.join(dirname, pocket_id.replace("/", "__"))
    )


########################################################################################
#  sampling from file utils
def read_structures(pdb_file, sdf_file, cfg):
    """
    Read structures from PDB and SDF files.

    Args:
        pdb_file (str): Path to the PDB file.
        sdf_file (str): Path to the SDF file.
        cfg (object): Configuration object.

    Returns:
        tuple: A tuple containing the ligand and target structures.
            - ligand (dict): Dictionary containing information about the ligand structure.
            - target (dict): Dictionary containing information about the target structure.
    """
    # ligand
    coords, atname = utils.parseSDF(sdf_file)
    atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH_CROSSDOCKED)
    c_min, _ = coords[0].min(axis=0)
    c_max, _ = coords[0].max(axis=0)
    mask = atoms_channel[0] < 7
    ligand = {
        "id": sdf_file,
        "coords": coords[0][mask],
        "atoms_channel": atoms_channel[0][mask].type(torch.uint8),
        "max_len": round((c_max - c_min).max().item(), 2),
        "radius": cfg.dset.ligand_radius * torch.ones_like(atoms_channel[0][mask])
    }
    center_coords = ligand["coords"].mean(axis=0)

    # target
    coords, atname = utils.parsePDB(pdb_file)
    atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH_CROSSDOCKED).type(torch.uint8)
    mask = atoms_channel[0] < 4  # pocket only has C, O, N, S
    if cfg.dset.pocket_radius > 0:
        radius = cfg.dset.pocket_radius * torch.ones_like(atoms_channel[0][mask]).float()
    else:
        radius = atomChannelsToRadius(atoms_channel[0][mask], ELEMENTS_HASH_CROSSDOCKED)
    target = {
        "id": pdb_file,
        "coords": coords[0].clone(),
        "atoms_channel": atoms_channel[0].type(torch.uint8),
        "radius": radius,
        "center_coords": center_coords
    }

    return ligand, target


def preprocess_structures(pdb_file, sdf_file, cfg, aug=False):
    """
    Preprocesses the structures by performing various operations such as reading structures,
    centering the reference frame, adding augmentation, filtering atoms by distance,
    and padding the ligand and pocket.

    Args:
        pdb_file (str): Path to the PDB file.
        sdf_file (str): Path to the SDF file.
        cfg (dict): Configuration parameters.
        aug (bool, optional): Whether to apply augmentation. Defaults to False.

    Returns:
        tuple: A tuple containing the preprocessed ligand and pocket structures.
    """
    # read structures
    ligand, target = read_structures(pdb_file, sdf_file, cfg)

    # center reference of frame to center of mass of ligand
    center_coords = target["center_coords"]
    ligand, target = recenter_structures(ligand, target, center_coords)

    # add aug (rotation then translation)
    if aug:
        ligand, target = rotate_coords(ligand, target)

    # box molecules
    ligand, pocket = filter_atoms_by_distance(ligand, target)

    # pad ligand and pocket
    ligand, pocket = pad(ligand, pocket)

    for key in ligand.keys():
        if key == "id":
            continue
        ligand[key] = ligand[key].unsqueeze(0)
        pocket[key] = pocket[key].unsqueeze(0)

    return ligand, pocket


def sample_from_file(
    pretrained_model_path: str,
    target_pdb: str,
    ligand_sdf: str,
    save_dirname: str,
    n_samples: int = 20,
    chain_init: str = "denovo",
    n_chains: int = 100,
    warmup: int = 400,
    steps: int = 100,
    max_steps: int = 100,
    mask_pocket: int = 0,
):

    makedir(save_dirname)

    # load model and voxelizer
    cfg_model = OmegaConf.structured(OmegaConf.load(os.path.join(pretrained_model_path, "cfg.yaml")))
    device = torch.device("cuda:0")
    model = create_model(cfg_model)
    model.to(device)
    model.eval()
    model, _ = load_checkpoint(model, pretrained_model_path, best_model=False)
    voxelizer = Voxelizer(
        grid_dim=cfg_model.vox.grid_dim,
        resolution=cfg_model.vox.resolution,
        cubes_around=cfg_model.vox.cubes_around,
        device=device,
    )

    # preprocess structures
    ligand_gt, pocket = preprocess_structures(target_pdb, ligand_sdf, cfg_model, aug=False)
    ligand_id, pocket_id = ligand_gt["id"], pocket["id"]
    shutil.copyfile(ligand_id, os.path.join(save_dirname, ligand_id.split("/")[-1]))
    shutil.copyfile(pocket_id, os.path.join(save_dirname, pocket_id.split("/")[-1]))

    # start sampling
    ligands_gt, pockets = make_batch(ligand_gt, pocket, 1)
    ligands_gt_vox = voxelizer.forward(ligands_gt, num_channels=7)
    pockets_vox = voxelizer.forward(pockets, num_channels=4)

    n_valid_mol, n_mol = 0, 0
    rdkmols, list_smiles = [], []
    while n_valid_mol < n_samples and n_mol < 500:
        print(">>>", n_valid_mol)
        # sample molecules
        gen_vox_mols = model.sample(
            pocket=pockets_vox,
            ligand=ligands_gt_vox,
            warmup_wjs=warmup,
            steps=steps,
            max_steps=max_steps,
            chain_init=chain_init,
            mask_pocket=mask_pocket > 0,
            n_chains=n_chains,
        )
        mols = voxelizer.vox2mol(gen_vox_mols, center_coords=pocket["center_coords"])

        if mols is None:
            break
        for i, mol in enumerate(mols):
            sdf_fname = os.path.join(save_dirname, f"sample_{n_valid_mol:03d}.sdf")
            mol = mol2rdkit_obabel(mol, sdf_fname)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if smiles not in list_smiles:
                    list_smiles.append(smiles)
                    n_valid_mol += 1
                    rdkmols.append(mol)
            n_mol += 1
    if n_valid_mol == 0:
        return 0

    # merge into single SDF file and delete individual files
    with Chem.SDWriter(os.path.join(save_dirname, "samples.sdf")) as writer:
        for i, rdkmol in enumerate(rdkmols):
            try:
                writer.write(rdkmol)
            except Exception:
                print("cannot save", rdkmol)

    return n_valid_mol
