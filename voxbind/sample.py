import logging
import os
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from voxbind.voxelizer import Voxelizer
from voxbind.models import create_model
from voxbind.utils.base_utils import load_checkpoint, seed_everything
from voxbind.utils.sampling_utils import sample_molecules
from voxbind.dataset import create_sampling_dataloader

logger = logging.getLogger("sampling")


@hydra.main(config_path="configs/", config_name="config_sample", version_base=None)
def main(cfg: DictConfig) -> None:
    # -----------------------------------------------------------
    # basic inits
    assert cfg.wjs.chain_init in ["denovo", "ligand"]
    assert torch.cuda.is_available(), "not a good idea to sample on cpu..."
    logger.info(f"n gpus available: {torch.cuda.device_count()}")
    logger.info(f"saving in dirname: {cfg.save_dir}")
    torch.set_default_dtype(torch.float32)
    seed_everything(cfg.seed)

    # ----------------------
    # model, voxelizer, loader
    cfg_model = OmegaConf.structured(OmegaConf.load(os.path.join(cfg.pretrained_path, "cfg.yaml")))
    cfg = OmegaConf.merge(cfg_model, cfg)
    device = torch.device("cuda:0")
    model = create_model(cfg)
    model.to(device)
    model, n_epochs = load_checkpoint(model, cfg.pretrained_path, best_model=False)
    logger.info(f"model trained for {n_epochs} epochs")

    voxelizer = Voxelizer(
        grid_dim=cfg_model.vox.grid_dim,
        resolution=cfg_model.vox.resolution,
        cubes_around=cfg_model.vox.cubes_around,
        device=device,
    )

    loader = create_sampling_dataloader(cfg_model, split=cfg.wjs.split)
    logger.info(f"test set size: {len(loader.dataset.data)}")

    # ----------------------
    # start sampling
    logger.info("start sampling...")
    model.eval()

    for pocket_id, batch in enumerate(loader):
        if pocket_id < cfg.wjs.start:
            continue
        if pocket_id > cfg.wjs.end:
            break
        logger.info(f"| sampling pocket {pocket_id}")
        target_dirname = os.path.join(cfg.save_dir, f"target_{pocket_id:02d}")

        # generate samples
        pocket, ligand_gt = batch["pocket"], batch["ligand"]
        t0 = time.time()
        sample_molecules(model, pocket, ligand_gt, voxelizer, target_dirname, cfg)
        logger.info(f"| sampling took {(time.time() - t0):.2f}s")

        if pocket_id == cfg.wjs.n_targets:
            break


if __name__ == "__main__":
    main()
