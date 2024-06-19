from voxbind.models.voxbind import VoxBind
from voxbind.constants import N_POCKET_ELEMENTS, N_LIGAND_ELEMENTS


def create_model(cfg, device="cuda") -> VoxBind:
    """
    Create a VoxBind model.

    Args:
        cfg (Config): The configuration object containing model parameters.
        device (str): The device to use for model computation. Defaults to "cuda".

    Returns:
        VoxBind: The created VoxBind model.
    """
    model = VoxBind(
        n_channels_ligand=N_LIGAND_ELEMENTS,
        n_channels_pocket=N_POCKET_ELEMENTS,
        n_channels=cfg.model.n_channels,
        ch_mults=cfg.model.ch_mults,
        is_attn=cfg.model.is_attn,
        n_blocks=cfg.model.n_blocks,
        n_groups=cfg.model.n_groups,
        dropout=cfg.model.dropout,
        smooth_sigma=cfg.smooth_sigma,
    )
    model.to(device)
    return model
