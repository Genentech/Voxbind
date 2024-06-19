import numpy as np
import torch

from copy import deepcopy
from functools import partial
from scipy import ndimage as ndi
from pyuul import VolumeMaker


class Voxelizer(torch.nn.Module):
    """
    Voxelizer module for converting molecular structures to voxel representations.

    Args:
        grid_dim (int): The dimension of the voxel grid (default: 64).
        resolution (float): The resolution of the voxel grid (default: 0.25).
        radius (float): The radius used for voxelization (default: 0.5).
        cubes_around (int): The number of cubes around each atom used for voxelization (default: 8).
        device (str): The device to use for computation (default: "cuda").

    Attributes:
        grid_dim (int): The dimension of the voxel grid.
        device (str): The device used for computation.
        radius (float): The radius used for voxelization.
        resolution (float): The resolution of the voxel grid.
        cubes_around (int): The number of cubes around each atom used for voxelization.
        vol_maker (VolumeMaker.Voxels): The voxelization module.

    """

    def __init__(
            self,
            grid_dim: int = 64,
            resolution: float = 0.25,
            radius: float = 0.5,
            cubes_around: int = 8,
            device="cuda"
    ):
        super(Voxelizer, self).__init__()
        self.grid_dim = grid_dim
        self.device = device
        self.radius = radius
        self.resolution = resolution
        self.cubes_around = cubes_around

        self.vol_maker = VolumeMaker.Voxels(
            device=device,
            sparse=False,
        )

    def forward(self, batch: list, num_channels: int = 7) -> torch.Tensor:
        """
        Forward pass of the Voxelizer module.

        Args:
            batch (list): The input batch of molecular structures.
            num_channels (int): The number of channels in the voxel grid (default: 7).

        Returns:
            torch.Tensor: The voxelized representation of the input batch.

        """
        return self.mol2vox(batch, num_channels=num_channels)

    def mol2vox(self, batch: list, num_channels: int = 7) -> torch.Tensor:
        """
        Convert a batch of molecular structures to voxel representations.

        Args:
            batch (list): The input batch of molecular structures.
            num_channels (int): The number of channels in the voxel grid (default: 7).

        Returns:
            torch.Tensor: The voxelized representation of the input batch.

        """

        # dumb coordinate to center ligand and pocket voxel
        batch = self._add_dumb_coords(batch)

        # to device
        batch["coords"] = batch["coords"].to(self.device)
        batch["radius"] = batch["radius"].to(self.device)
        batch["atoms_channel"] = batch["atoms_channel"].to(self.device)

        # voxelize
        voxels = []
        batch_sz = batch["coords"].shape[0]
        n_chuncks = 4 if batch_sz > 16 else 1
        chk = batch["coords"].shape[0] // n_chuncks
        for i in range(n_chuncks):
            voxels_ = self.vol_maker(
                batch["coords"][i * chk:(i + 1) * chk],
                batch["radius"][i * chk:(i + 1) * chk],
                batch["atoms_channel"][i * chk:(i + 1) * chk],
                resolution=self.resolution,
                cubes_around_atoms_dim=self.cubes_around,
                function="gaussian",
                numberchannels=num_channels,
            )
            # extract center box (and get rid of dumb coordinates)
            c = voxels_.shape[-1] // 2
            box_min, box_max = c - self.grid_dim // 2, c + self.grid_dim // 2
            voxels_ = voxels_[:, :, box_min:box_max, box_min:box_max, box_min:box_max]
            voxels.append(voxels_)
        voxels = torch.cat(voxels, axis=0)

        return voxels

    def vox2mol(
        self,
        voxels: torch.Tensor,
        refine: bool = True,
        center_coords: torch.Tensor = None
    ) -> list:
        """
        Convert voxel representations back to molecular structures.

        Args:
            voxels (torch.Tensor): The input voxel representations.
            refine (bool): Whether to refine the coordinates using optimization (default: True).
            center_coords (torch.Tensor): The center coordinates for recentering the molecular structures.

        Returns:
            list: The reconstructed molecular structures.

        """
        assert len(voxels.shape) == 5

        # intialize coods with simple peak detection
        mol_inits = []
        voxel_inits = []
        for voxel in voxels:
            mol_init = get_atom_coords(voxel.cpu(), rad=self.radius, resolution=self.resolution)
            if mol_init is not None and mol_init["coords"].shape[1] < 200:
                mol_inits.append(mol_init)
                voxel_inits.append(voxel.unsqueeze(0))

        if len(mol_inits) == 0:
            return None

        if not refine:
            mols = recenter_mols(mol_inits, center_coords)
            return mols
        voxel_inits = torch.cat(voxel_inits, axis=0)

        # refine coords
        optim_factory = partial(
            torch.optim.LBFGS, history_size=10, max_iter=4, line_search_fn="strong_wolfe",
        )

        mols = self._refine_coords(mol_inits, voxel_inits, optim_factory, maxiter=10)
        del voxels, mol_inits
        torch.cuda.empty_cache()

        mols = recenter_mols(mols, center_coords)

        return mols

    def _refine_coords(
        self,
        mol_inits: list,
        voxels: torch.Tensor,
        optim_factory,
        tol: float = 1e-6,
        maxiter: int = 15,
        callback=None
    ) -> list:
        """
        Refine the coordinates of molecular structures using optimization.

        Args:
            mol_inits (list): The initial molecular structures.
            voxels (torch.Tensor): The voxel representations.
            optim_factory: The optimization algorithm used for refinement.
            tol (float): The tolerance for convergence (default: 1e-6).
            maxiter (int): The maximum number of iterations (default: 15).
            callback: The callback function for monitoring the refinement process.

        Returns:
            list: The refined molecular structures.

        """
        assert len(voxels.shape) == 5, "voxels need to have dimension 5 (including the batch dim.)"

        mols = []
        for i in range(voxels.shape[0]):
            mol_init = mol_inits[i]
            voxel = voxels[i].unsqueeze(0)

            mol = deepcopy(mol_init)
            mol["coords"].requires_grad = True

            optimizer = optim_factory([mol["coords"]])

            def closure():
                optimizer.zero_grad()
                voxel_fit = self.forward(mol)
                loss = torch.nn.functional.mse_loss(voxel, voxel_fit)
                loss.backward()
                return loss

            loss = 1e10
            for _ in range(maxiter):
                try:
                    prev_loss = loss
                    loss = optimizer.step(closure)
                except Exception:
                    print(
                        "refine coords diverges, so use initial cordinates...",
                        f"(coords min: {mol['coords'].min().item()}, max: {mol['coords'].max().item()})"
                    )
                    mol = deepcopy(mol_init)
                    break

                if callback is not None:
                    callback(mol)

                if abs(loss.item() - prev_loss) < tol:
                    break

            mols.append({
                "coords": mol["coords"].detach().cpu(),
                "atoms_channel": mol["atoms_channel"].detach().cpu(),
                "radius": mol["radius"].detach().cpu(),
            })

        return mols

    def _add_dumb_coords(self, batch: dict) -> dict:
        """
        Add dumb coordinates to the input batch for centering the ligand and pocket voxel.

        Args:
            batch (dict): The input batch of molecular structures.

        Returns:
            dict: The modified batch with dumb coordinates.

        """
        bsz = batch['coords'].shape[0]
        return {
            "coords": torch.cat(
                (batch['coords'], torch.Tensor(bsz, 1, 3).fill_(-25), torch.Tensor(bsz, 1, 3).fill_(25)), 1
            ),
            "atoms_channel": torch.cat(
                (batch['atoms_channel'], torch.Tensor(bsz, 2).fill_(0)), 1
            ),
            "radius": torch.cat(
                (batch['radius'], torch.Tensor(bsz, 2).fill_(.5), ), 1
            )
        }


########################################################################################
# aux functions
def local_maxima(
    data: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Find local maxima in a 3D array.

    Args:
        data (np.ndarray): The input 3D array.
        order (int, optional): The order of the local maxima. Defaults to 1.

    Returns:
        np.ndarray: The modified 3D array with local maxima set to 0.
    """
    data = data.numpy()
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    data[data <= filtered] = 0
    return data


def find_peaks(voxel: torch.Tensor) -> torch.Tensor:
    """
    Find peaks in a voxel.

    Args:
        voxel (torch.Tensor): Input voxel.

    Returns:
        torch.Tensor: Peaks found in the voxel.
    """
    voxel[voxel < .25] = 0
    voxel = voxel.squeeze().clone()
    peaks = []
    for channel_idx in range(voxel.shape[0]):
        vox_in = voxel[channel_idx]
        peaks_ = local_maxima(vox_in, 1)
        peaks_ = torch.Tensor(peaks_).unsqueeze(0)
        peaks.append(peaks_)
    peaks = torch.concat(peaks, axis=0)
    return peaks


def get_atom_coords(
    grid: torch.Tensor,
    rad: float = 0.5,
    resolution: float = 0.25
) -> dict:
    """
    Get the coordinates of atoms from a grid.

    Args:
        grid (torch.Tensor): The input grid.
        rad (float, optional): The radius of the atoms. Defaults to 0.5.
        resolution (float, optional): The resolution of the grid. Defaults to 0.25.

    Returns:
        dict: A dictionary containing the coordinates, atom channels, and radii of the atoms.
    """
    peaks = find_peaks(grid.cpu())
    coords = []
    atoms_channel = []
    radius = []
    grid_dim = peaks.shape[-1]

    for channel_idx in range(peaks.shape[0]):
        px, py, pz = torch.where(peaks[channel_idx] > 0)
        px, py, pz = px.float(), py.float(), pz.float()
        coords.append(torch.cat([px.unsqueeze(1), py.unsqueeze(1), pz.unsqueeze(1)], axis=1))
        atoms_channel.append(torch.Tensor(px.shape[0]).fill_(channel_idx))
        radius.append(torch.Tensor(px.shape[0]).fill_(rad))
    coords = (torch.cat(coords, 0).unsqueeze(0) - (grid_dim - 1) / 2) * resolution

    if coords.shape[1] == 0:
        return None

    mol = {
        "coords": coords,
        "atoms_channel": torch.cat(atoms_channel, 0).unsqueeze(0),
        "radius": torch.cat(radius, 0).unsqueeze(0),
    }

    return mol


def recenter_mols(mols: list, center_coords: torch.Tensor) -> list:
    """
    Recenter the molecules based on the given center coordinates.

    Args:
        mols (list): List of molecules.
        center_coords (torch.Tensor): Center coordinates.

    Returns:
        list: List of recentered molecules.
    """
    centered_mols = []
    for mol in mols:
        coords = mol["coords"]
        if center_coords is not None:
            center_coords_ = center_coords.unsqueeze(0).tile((1, coords.shape[0], 1))
            coords += center_coords_
        centered_mols.append({
            "coords": coords,
            "atoms_channel": mol["atoms_channel"],
            "radius": mol["radius"]
        })

    return centered_mols
