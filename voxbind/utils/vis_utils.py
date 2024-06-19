import torch
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

from voxbind.utils.base_utils import makedir

sns.set_theme()

# ATOM_ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "P", "Br", "H", "I", "B"]
COLORS_LIGAND = [
    "Greys",
    "Reds",
    "Blues",
    [[0, "rgb(0,0,0)"], [1, "rgb(255,191,0)"]],
    "Greens",
    "Mint",
    "Oranges",
    "Magenta",
    "Purples",
    [[0, "rgb(0,0,0)"], [1, "rgb(175, 96, 26)"]],
]

COLORS_POCKET = [
    "gray",
    "gray",
    "gray",
    "gray",
    "gray",
    "gray",
    "gray",
    "gray",
]


def visualize_ligand_pocket(
    ligand: torch.Tensor,
    pocket: torch.Tensor,
    name: str = "temp",
    dirname: str = "figures/",
    threshold: float = 0.1,
    downsample: int = 1,
    to_png: bool = True,
    to_html: bool = False,
):
    """
    Visualizes the ligand and pocket volumes (voxel grids) using a 3D plot.

    Args:
        ligand (torch.Tensor): The ligand volume tensor.
        pocket (torch.Tensor): The pocket volume tensor.
        name (str, optional): The name of the output file. Defaults to "temp".
        dirname (str, optional): The directory to save the output files. Defaults to "figures/".
        threshold (float, optional): The threshold value for voxel visualization. Defaults to 0.1.
        downsample (int, optional): The downsampling factor for voxel visualization. Defaults to 1.
        to_png (bool, optional): Whether to save the visualization as a PNG image. Defaults to True.
        to_html (bool, optional): Whether to save the visualization as an HTML file. Defaults to False.
    """
    if ligand is not None:
        assert len(ligand.shape) == 4
        ligand = ligand.cpu()
    if pocket is not None:
        assert len(pocket.shape) == 4
        pocket = pocket.cpu()

    makedir(dirname)
    fig = go.Figure()

    for voxel, is_pocket in [[pocket, True], [ligand, False]]:
        if voxel is None:
            continue
        if downsample > 1:
            voxel = torch.nn.functional.avg_pool3d(voxel, (downsample, downsample, downsample))
        colors = COLORS_POCKET if is_pocket else COLORS_LIGAND
        voxel = voxel.squeeze()
        voxel[voxel < threshold] = 0
        X, Y, Z = np.mgrid[:voxel.shape[-3], :voxel.shape[-2], :voxel.shape[-1]]

        for channel in range(voxel.shape[0]):
            voxel_channel = voxel[channel:channel + 1]
            if voxel_channel.sum().item() == 0:
                continue
            fig.add_volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=voxel_channel.flatten(),
                isomin=.19,
                isomax=.2,  # 0.3
                opacity=.1,  # 0.075, # needs to be small to see through all surfaces
                surface_count=17,  # needs to be a large number for good volume rendering
                colorscale=colors[channel],
                showscale=False,
            )

    if to_html:
        fig.write_html(f"{dirname}/{name}.html")
    if to_png:
        fig.write_image(f"{dirname}/{name}.png")
