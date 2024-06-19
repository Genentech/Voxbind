import math
import random
import torch
from typing import Tuple

from voxbind.constants import RADIUS_PER_ATOM


def pad(
    ligand: dict,
    target: dict
) -> Tuple[dict, dict]:
    """
    Pads the input dictionaries with zeros or a specified value to match a desired length.

    Args:
        ligand (dict): Dictionary containing information about the ligand.
        target (dict): Dictionary containing information about the target.

    Returns:
        tuple: A tuple containing the padded ligand and target dictionaries.
    """
    pad_len = 100 - ligand["atoms_channel"].shape[0]  # max n_atoms on ligands is 73
    ligand["atoms_channel"] = ligand["atoms_channel"].float()

    pad_ = torch.nn.functional.pad

    ligand = {
        "id": ligand["id"],
        "coords": pad_(ligand["coords"], [0, 0, 0, pad_len], value=999),
        "atoms_channel": pad_(ligand["atoms_channel"], [0, pad_len], value=999),
        "radius": pad_(ligand["radius"], [0, pad_len], value=999),
    }

    pad_len = 2000 - target["atoms_channel"].shape[0]
    target["atoms_channel"] = target["atoms_channel"].float()

    target = {
        "id": target["id"],
        "coords": pad_(target["coords"], [0, 0, 0, pad_len], value=999),
        "atoms_channel": pad_(target["atoms_channel"], [0, pad_len], value=999),
        "radius": pad_(target["radius"], [0, pad_len], value=999),
        "center_coords": target["center_coords"]
    }

    return ligand, target


def get_pocket(
    target: dict,
    grid_len: int = 20
) -> dict:
    """
    Retrieves the pocket from the target dictionary based on the given grid length.

    Args:
        target (dict): The target dictionary containing the coordinates, atoms channel, and radius.
        grid_len (int, optional): The length of the grid. Defaults to 20.

    Returns:
        dict: The updated target dictionary with the pocket coordinates, atoms channel, and radius.
    """
    coords = target["coords"]
    coords_min, _ = coords.min(axis=1)
    coords_max, _ = coords.max(axis=1)
    mask = torch.logical_and(coords_min > -grid_len / 2, coords_max < grid_len / 2)

    target["coords"] = coords[mask]
    target["atoms_channel"] = target["atoms_channel"][mask]
    target["radius"] = target["radius"][mask]

    return target


def recenter_structures(
    ligand: dict,
    target: dict,
    center_coords: torch.Tensor
) -> Tuple[dict, dict]:
    """
    Recenter the ligand and target structures based on the provided center coordinates.

    Args:
        ligand (dict): Dictionary containing the ligand structure information.
        target (dict): Dictionary containing the target structure information.
        center_coords (torch.Tensor): Tensor representing the center coordinates.

    Returns:
        tuple: A tuple containing the centered ligand and centered target structures.
    """
    # subtract center of mass from ligand
    coords = ligand["coords"]
    center_coords_tiled = center_coords.unsqueeze(0).tile((coords.shape[0], 1))
    centered_ligand = {k: v for k, v in ligand.items()}
    centered_ligand["coords"] = coords - center_coords_tiled

    # subtract center of mass from target
    coords = target["coords"]
    center_coords_tiled = center_coords.unsqueeze(0).tile((coords.shape[0], 1))
    centered_target = {k: v for k, v in target.items()}
    centered_target["coords"] = coords - center_coords_tiled

    return centered_ligand, centered_target


def rotate_coords(
    ligand: dict,
    pocket: dict
) -> Tuple[dict, dict]:
    """Randomly rotate coordinates

    Args:
        ligand (dict): Dictionary containing ligand information.
            It should have a key "coords" with a torch.Tensor of shape [Nx3].
        pocket (dict): Dictionary containing pocket information.
            It should have a key "coords" with a torch.Tensor of shape [Nx3].

    Returns:
        tuple: A tuple containing the updated ligand and pocket dictionaries.
            The "coords" key in both dictionaries will be updated with the rotated coordinates.
    """
    rot_matrix = random_rot_matrix()

    coords_pocket = pocket["coords"]
    coords_pocket = torch.reshape(coords_pocket, (-1, 3))
    pocket["coords"] = torch.einsum("ij, kj -> ki", rot_matrix, coords_pocket)

    coords_ligand = ligand["coords"]
    coords_ligand = torch.reshape(coords_ligand, (-1, 3))
    ligand["coords"] = torch.einsum("ij, kj -> ki", rot_matrix, coords_ligand)

    return ligand, pocket


def random_rot_matrix() -> torch.Tensor:
    """Apply random rotation in each of hte x, y and z axis.
    First compute the 3D matrix for each rotation, then multiply them

    Returns:
        torch.Tensor: return rotation matrix (3x3)
    """
    theta_x = random.uniform(0, 2) * math.pi
    rot_x = torch.Tensor([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)],
    ])
    theta_y = random.uniform(0, 2) * math.pi
    rot_y = torch.Tensor([
        [math.cos(theta_y), 0, -math.sin(theta_y)],
        [0, 1, 0],
        [math.sin(theta_y), 0, math.cos(theta_y)],
    ])
    theta_z = random.uniform(0, 2) * math.pi
    rot_z = torch.Tensor([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1],
    ])

    return rot_z @ rot_y @ rot_x


def translate_coords(
    ligand: dict,
    pocket: dict,
    delta: float = 1.
) -> Tuple[dict, dict]:
    """
    Translates the coordinates of the ligand and pocket by adding random noise.

    Args:
        ligand (dict): Dictionary containing the ligand coordinates.
        pocket (dict): Dictionary containing the pocket coordinates.
        delta (float, optional): Maximum magnitude of the random noise. Defaults to 1.

    Returns:
        tuple: Tuple containing the updated ligand and pocket dictionaries.
    """
    noise = (torch.rand((1, 3)) - 1 / 2) * 2 * delta

    pocket["coords"] += noise.repeat(pocket["coords"].shape[0], 1)
    ligand["coords"] += noise.repeat(ligand["coords"].shape[0], 1)

    return ligand, pocket


def atomChannelsToRadius(
    atoms_channel: torch.Tensor,
    hashing: dict
) -> torch.Tensor:
    """
    Convert atom channels to corresponding atomic radii.

    Args:
        atoms_channel (torch.Tensor): Tensor containing atom channels.
        hashing (dict): Dictionary mapping atom indices to element symbols.

    Returns:
        torch.Tensor: Tensor containing atomic radii corresponding to the atom channels.
    """
    radius = []
    element_ids = [k for k in hashing.keys()]
    for atom_channel in atoms_channel:
        if atom_channel < len(element_ids):
            element = element_ids[atom_channel]
            radius.append(RADIUS_PER_ATOM["MOL"][element])
        else:
            radius.append(999)
    return torch.Tensor(radius)


def filter_atoms_by_distance(
    ligand: dict,
    pocket: dict,
    max_dim: int = 25
) -> Tuple[dict, dict]:
    """
    Filters atoms in the ligand and pocket dictionaries based on their distance from the origin.

    Args:
        ligand (dict): Dictionary containing ligand information.
        pocket (dict): Dictionary containing pocket information.
        max_dim (int, optional): Maximum dimension for filtering atoms. Defaults to 25.

    Returns:
        tuple: A tuple containing the filtered ligand and pocket dictionaries.
    """
    # ligand
    ligand["coords"][ligand["coords"] < -max_dim] = -max_dim
    ligand["coords"][ligand["coords"] > max_dim] = max_dim

    # pocket
    pocket["coords"][pocket["coords"] < -max_dim] = -max_dim
    pocket["coords"][pocket["coords"] > max_dim] = max_dim

    return ligand, pocket
