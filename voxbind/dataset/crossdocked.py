import os
import random
import torch

from torch.utils.data import Dataset

from voxbind.constants import ELEMENTS_HASH_CROSSDOCKED
from voxbind.utils.dataset_utils import (
    recenter_structures, pad, rotate_coords, translate_coords,
    atomChannelsToRadius, filter_atoms_by_distance
)


class DatasetCrossdocked(Dataset):
    def __init__(
        self,
        data_dir: str = "dataset/data/",
        split: str = "train",
        aug: bool = True,
        small: bool = False,
        ligand_radius: float = .5,
        pocket_radius: float = -1,
        max_len: int = 30,
        verbose: bool = False,
        delta_translate: float = 1.,
    ):
        """
        Dataset class for crossdocked data.

        Args:
            data_dir (str, optional): Directory containing the dataset. Defaults to "dataset/data/".
            split (str, optional): Split of the dataset to use ("train", "val", or "test"). Defaults to "train".
            aug (bool, optional): Whether to apply data augmentation. Defaults to True.
            small (bool, optional): Whether to use a small subset of the dataset. Defaults to False.
            ligand_radius (float, optional): Radius of the ligand atoms. Defaults to 0.5.
            pocket_radius (float, optional): Radius of the pocket atoms. Defaults to -1.
            max_len (int, optional): Maximum length of the molecule. Defaults to 30.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            delta_translate (float, optional): Translation factor for data augmentation in A. Defaults to 1.0.
        """
        assert split in ["train", "val", "test"]

        self.data_dir = data_dir
        self.split = split
        self.aug = aug
        self.ligand_radius = ligand_radius
        self.pocket_radius = pocket_radius
        self.small = small
        self.max_len = max_len
        self.verbose = verbose
        self.delta = delta_translate

        if split == "train" or split == "val":
            data = torch.load(os.path.join(data_dir, "data_train.pt"))
            random.Random(1234).shuffle(data)
            val_sz = 100
            self.data = data[:(len(data) - val_sz)] if split == "train" else data[(len(data) - val_sz):]
        else:
            self.data = torch.load(os.path.join(data_dir, "data_test.pt"))

        # filter dataset
        self.data = self.data[:500] if self.small else self.data
        self._filter_by_size(max_len=max_len)

    def _filter_by_size(self, max_len: int = 21) -> list:
        """Filter dataset by max_len of molecule.

        Args:
            max_len (int, optional): Maximum length in Angstroms (among x, y, z) coordinate. Defaults to 16.
        """
        filtered_data = [(datum[0], datum[1]) for datum in self.data if datum[1]["max_len"] <= max_len]
        if self.verbose:
            print(f"| filter by size reduce n ligands from {len(self.data)} to {len(filtered_data)}")
        self.data = filtered_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        pocket_, ligand_ = self.data[index]

        # ligand
        mask = ligand_["atoms_channel"] < 7
        ligand = {
            "id": ligand_["id"],
            "coords": ligand_["coords"][mask],
            "atoms_channel": ligand_["atoms_channel"][mask],
            "radius": self.ligand_radius * torch.ones_like(ligand_["atoms_channel"][mask])
        }
        center_coords = ligand["coords"].mean(axis=0)

        # pocket
        mask = pocket_["atoms_channel"] < 4  # pocket only has C, O, N, S
        if self.pocket_radius > 0:
            radius = self.pocket_radius * torch.ones_like(pocket_["atoms_channel"][mask]).float()
        else:
            radius = atomChannelsToRadius(pocket_["atoms_channel"][mask], ELEMENTS_HASH_CROSSDOCKED)
        pocket = {
            "id": pocket_["id"],
            "coords": pocket_["coords"][mask],
            "atoms_channel": pocket_["atoms_channel"][mask],
            "radius": radius,
            "center_coords": center_coords
        }

        # center reference of frame to center of mass of ligand
        ligand, pocket = recenter_structures(ligand, pocket, center_coords)

        # add aug (rotation then translation)
        if self.aug:
            ligand, pocket = rotate_coords(ligand, pocket)
            ligand, pocket = translate_coords(ligand, pocket, self.delta)

        # box molecules
        ligand, pocket = filter_atoms_by_distance(ligand, pocket)

        # pad ligand and pocket
        ligand, pocket = pad(ligand, pocket)

        return {"pocket": pocket, "ligand": ligand}


if __name__ == "__main__":
    from tqdm import tqdm
    from voxbind.utils.base_utils import makedir
    from voxbind.utils.vis_utils import visualize_ligand_pocket
    from voxbind.voxelizer import Voxelizer
    out_dir = "data_temp/"
    makedir(out_dir)

    dset = DatasetCrossdocked(
        data_dir="data/",
        aug=False,
        small=False,
        max_len=16,
        split="val",
    )
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=32,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    voxelizer = Voxelizer(grid_dim=64, device="cuda")
    for _, batch in enumerate(loader):
        pocket, ligand = batch["pocket"], batch["ligand"]
        vox_pocket = voxelizer(pocket, num_channels=4)
        vox_lig = voxelizer(ligand, num_channels=7)
        vp = vox_pocket.sum(1)
        vl = vox_lig.sum(1)
        for poc, lig in zip(vp, vl):
            print("intersection", torch.logical_and(poc > 0, lig > 0).sum().item())
        for i in tqdm(range(vox_pocket.shape[0])):
            visualize_ligand_pocket(vox_lig[i], vox_pocket[i], f"lig_poc_{i}", out_dir, threshold=0., downsample=1)
            visualize_ligand_pocket(None, vox_pocket[i], f"poc_{i}", out_dir, threshold=0., downsample=1)
            visualize_ligand_pocket(vox_lig[i], None, f"lig_{i}", out_dir, threshold=0., downsample=1)
        break
