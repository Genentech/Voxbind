import torch

from voxbind.dataset.crossdocked import DatasetCrossdocked


def create_dataloaders(cfg) -> tuple:
    """
    Create data loaders for training, validation, and sampling.

    Args:
        cfg (Config): Configuration object containing dataset and training parameters.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and sampling data loader.
    """
    if cfg.dset.dset_name == "crossdocked":
        Dataset = DatasetCrossdocked
    else:
        NotImplementedError(f"{cfg.dset.dset_name} Not implemented yet")

    # create train loader
    dset_train = Dataset(
        data_dir=cfg.dset.data_dir,
        split="train",
        aug=cfg.aug,
        small=cfg.debug,
        ligand_radius=cfg.dset.ligand_radius,
        pocket_radius=cfg.dset.pocket_radius,
    )

    loader_train = torch.utils.data.DataLoader(
        dset_train,
        batch_size=cfg.bsz,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # create val loader
    dset_val = Dataset(
        data_dir=cfg.dset.data_dir,
        split="val",
        aug=False,
        small=cfg.debug,
        ligand_radius=cfg.dset.ligand_radius,
        pocket_radius=cfg.dset.pocket_radius,
    )
    loader_val = torch.utils.data.DataLoader(
        dset_val,
        batch_size=cfg.bsz,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # create sampling loader
    loader_sampling = torch.utils.data.DataLoader(
        dset_val,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return loader_train, loader_val, loader_sampling


def create_sampling_dataloader(cfg, split="val") -> torch.utils.data.DataLoader:
    """
    Create a data loader for sampling.

    Args:
        cfg (dict): Configuration dictionary.
        split (str, optional): Split name. Defaults to "val".

    Returns:
        torch.utils.data.DataLoader: Data loader for sampling.
    """
    if cfg.dset.dset_name == "crossdocked":
        Dataset = DatasetCrossdocked
    else:
        NotImplementedError(f"{cfg['dataset']['dset_name']} Not implemented yet")
    dset_val = Dataset(
        data_dir=cfg.dset.data_dir,
        split=split,
        aug=False,
        small=cfg.debug,
        ligand_radius=cfg.dset.ligand_radius,
        pocket_radius=cfg.dset.pocket_radius,
    )
    # create sampling loader
    loader_sampling = torch.utils.data.DataLoader(
        dset_val,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return loader_sampling
