import os
import hydra
import logging
from omegaconf import DictConfig
import time
import torch

from voxbind.utils.base_utils import seed_everything
from voxbind.utils.sampling_utils import sample_from_file


logger = logging.getLogger("sampling")


@hydra.main(config_path="configs/", config_name="config_sample_from_file", version_base=None)
def main(cfg: DictConfig) -> None:
    assert cfg.wjs.chain_init in ["denovo", "ligand"]
    assert torch.cuda.is_available(), "not a good idea to sample on cpu..."
    torch.set_default_dtype(torch.float32)
    seed_everything(cfg.seed)
    logger.info(f"n gpus available: {torch.cuda.device_count()}")
    logger.info(f"saving in dirname: {cfg.save_dir}")

    logger.info(f"start sampling from pocket: {cfg.target_pdb}")
    save_dirname = os.path.join(
        cfg.save_dir,
        os.path.basename(cfg.target_pdb).split(".")[0],
        cfg.wjs.chain_init
    )

    t0 = time.time()
    sample_from_file(
        pretrained_model_path=cfg.pretrained_path,
        target_pdb=cfg.target_pdb,
        ligand_sdf=cfg.ligand_sdf,
        save_dirname=save_dirname,
        n_samples=cfg.n_samples,
        chain_init=cfg.wjs.chain_init,
        n_chains=cfg.wjs.n_chains,
        warmup=cfg.wjs.warmup,
        steps=cfg.wjs.steps,
        max_steps=cfg.wjs.max_steps,
        mask_pocket=cfg.wjs.mask_pocket,
    )
    logger.info(f"| sampling took {(time.time() - t0):.2f}s")


if __name__ == "__main__":
    main()
