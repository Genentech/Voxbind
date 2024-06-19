import getpass as gt
import hydra
import logging
import math
import os
from omegaconf import DictConfig, OmegaConf
import time
import torch
import torchmetrics
import wandb

from voxbind.voxelizer import Voxelizer
from voxbind.models import create_model
from voxbind.utils.base_utils import (
    create_exp_dir, makedir, seed_everything, save_checkpoint, load_checkpoint
)
from voxbind.utils.sampling_utils import sample_molecules
from voxbind.dataset import create_dataloaders
from voxbind.models.adamw import AdamW
from voxbind.models.ema import ModelEma
from voxbind.metrics import create_metrics_for_training

logger = logging.getLogger("training")


@hydra.main(config_path="configs", config_name="config_train", version_base=None)
def main(cfg: DictConfig) -> None:
    # -----------------------------------------------------------
    # basic inits
    assert torch.cuda.is_available(), "not a good idea to train on cpu..."
    start_epoch = 0
    create_exp_dir(cfg)
    logger.info(f"n gpus available: {torch.cuda.device_count()}")
    logger.info(f"saving experiments in: {cfg.output_dir}")
    torch.set_default_dtype(torch.float32)
    logger.info("set matmul precision to high")
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.seed)

    # resume?
    if cfg.resume is not None and os.path.isdir(cfg.resume):
        logger.info(f"resuming from: {cfg.resume}")
        resume = cfg.resume
        cfg = OmegaConf.load(os.path.join(cfg.resume, "cfg.yaml"))
        cfg.output_dir, cfg.resume = resume, resume
        cfg.wandb = False

    logger.info("cfg:")
    logger.info(OmegaConf.to_yaml(cfg))

    # wandb?
    if cfg.wandb:
        wandb.init(
            project="voxbind",
            entity=gt.getuser(),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=cfg.exp_name,
            dir=cfg.output_dir,
            settings=wandb.Settings(code_dir=".")
        )

    # -----------------------------------------------------------
    # create training objects
    # data loaders
    loader_train, loader_val, loader_sampling = create_dataloaders(cfg)
    n_train, n_val = len(loader_train.dataset.data), len(loader_val.dataset.data)
    logger.info(f"training/val set size: {n_train}/{n_val}")

    # model, criterion, optimizer
    device = torch.device("cuda")
    model = create_model(cfg)
    model.to(device)
    criterion = torch.nn.MSELoss(reduction="sum").to("cuda")
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optimizer.zero_grad()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model has {(n_params/1e6):.02f}M parameters")

    # voxelizer
    voxelizer = Voxelizer(
        grid_dim=cfg.vox.grid_dim,
        resolution=cfg.vox.resolution,
        cubes_around=cfg.vox.cubes_around,
        device=device,
    )

    # optionally reload states of model, optimizer
    if cfg.resume is not None:
        logger.info("reloading states of model, optimizer")
        model, optimizer, start_epoch = load_checkpoint(
            model, cfg.output_dir, optimizer, best_model=False
        )
        os.system(f"cp {os.path.join(cfg.output_dir, 'checkpoint.pth.tar')} "
                  + f"{os.path.join(cfg.output_dir, f'checkpoint_{start_epoch}.pth.tar')}")
        logger.info(f"model trained for {start_epoch} epochs")

    # DP/ema (exponential moving average)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
    model.to(device)
    model_ema = ModelEma(model, decay=.999)

    # metrics
    metrics_denoise, _ = create_metrics_for_training()

    # -----------------------------------------------------------
    # start training
    logger.info("start training...")
    for epoch in range(start_epoch, start_epoch + cfg.num_epochs):
        t0 = time.time()

        # train
        train_metrics = train(
            cfg, loader_train, voxelizer, model, criterion, optimizer, metrics_denoise, model_ema
        )

        # val
        val_metrics = val(
            cfg, loader_val, voxelizer, model_ema.module, criterion, metrics_denoise
        )

        # sample
        if epoch > 0 and (epoch % 50 == 0 or epoch == cfg.num_epochs - 1):
            sample(cfg, loader_sampling, voxelizer, model_ema.module, epoch)

        # log and wandb
        log_metrics(epoch, train_metrics, val_metrics, time.time() - t0)
        if cfg.wandb:
            wandb.log({"train": train_metrics, "val": val_metrics})

        # save model
        save_checkpoint({
            "epoch": epoch,
            "metrics": {"train": train_metrics, "val": val_metrics},
            "cfg": cfg,
            "state_dict_ema": model_ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, save_dir=cfg.output_dir)
        torch.cuda.empty_cache()


def train(
    cfg: DictConfig,
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn,
    criterion: torch.nn,
    optimizer: torch.optim,
    metrics: torchmetrics.MetricCollection,
    model_ema: torch.nn,
) -> torch.Tensor:
    """Train one epoch of the model.

    Args:
        cfg (DictConfig): Configuration parameters for training.
        loader (torch.utils.data.DataLoader): Data loader for training data.
        voxelizer (Voxelizer): Voxelizer object for voxelizing input data.
        model (torch.nn): Model to be trained.
        criterion (torch.nn): Loss criterion for training.
        optimizer (torch.optim): Optimizer for updating model parameters.
        metrics (torchmetrics.MetricCollection): Metrics for evaluating model performance.
        model_ema (torch.nn): Exponential moving average model for model updates.

    Returns:
        torch.Tensor: Computed metrics for the epoch.
    """
    metrics.reset()
    model.train()

    for i, batch in enumerate(loader):
        # voxelize and add noise
        with torch.no_grad():
            voxels_lig = voxelizer.forward(batch["ligand"], num_channels=7)
            smooth_voxels_lig = add_noise_vox(voxels_lig, cfg.smooth_sigma)
            voxels_poc = voxelizer.forward(batch["pocket"], num_channels=4)
            voxels_poc[:math.ceil(.2 * voxels_poc.shape[0])].zero_()  # drop 20% of pockets

        # fwd
        pred = model(smooth_voxels_lig, voxels_poc)
        loss = criterion(pred, voxels_lig)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # update model EMA
        model_ema.update(model)

        # update metrics
        metrics.update(loss, pred, voxels_lig)

        if cfg.debug and i == 10:
            break

    return metrics.compute()


def val(
    cfg: DictConfig,
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn,
    criterion: torch.nn,
    metrics: torchmetrics.MetricCollection
) -> torch.Tensor:
    """Evaluate model.

    Args:
        cfg (DictConfig): Configuration dictionary.
        loader (torch.utils.data.DataLoader): Data loader for evaluation dataset.
        voxelizer (Voxelizer): Voxelizer object for voxelizing input data.
        model (torch.nn): Model to be evaluated.
        criterion (torch.nn): Loss criterion for evaluation.
        metrics (torchmetrics.MetricCollection): Collection of metrics for evaluation.

    Returns:
        float: Computed metrics for evaluation.
    """
    metrics.reset()
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # voxelize and add noise
            voxels_lig = voxelizer.forward(batch["ligand"], num_channels=7)
            smooth_voxels_lig = add_noise_vox(voxels_lig, cfg.smooth_sigma)
            voxels_poc = voxelizer.forward(batch["pocket"], num_channels=4)

            # fwd and metrics
            pred = model(smooth_voxels_lig, voxels_poc)
            loss = criterion(pred, voxels_lig)
            metrics.update(loss, pred, voxels_lig)

            if cfg.debug and i == 10:
                break

    return metrics.compute()


def sample(
    cfg: DictConfig,
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn,
    epoch: int
) -> None:
    """Sample ligands with WJS

    Args:
        cfg (DictConfig): Configuration settings for the sampling process.
        loader (torch.utils.data.DataLoader): DataLoader for loading the data.
        voxelizer (Voxelizer): Voxelizer object for voxelizing the input data.
        model (torch.nn): Model for generating samples.
        epoch (int): Current epoch number.

    Returns:
        None
    """
    if torch.cuda.device_count() > 1:
        model = model.module
    model.eval()

    dirname = os.path.join(cfg.output_dir, f"samples_training/{epoch:02d}")
    makedir(dirname)

    for pocket_id, batch in enumerate(loader):
        if pocket_id == cfg.wjs.n_targets:
            break
        logger.info(f"| sampling pocket {pocket_id}")
        target_dirname = os.path.join(dirname, f"target_{pocket_id:02d}")

        # generate samples
        pocket, ligand_gt = batch["pocket"], batch["ligand"]
        sample_molecules(model, pocket, ligand_gt, voxelizer, target_dirname, cfg)

    return None


def add_noise_vox(voxels: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the input voxels.

    Args:
        voxels (torch.Tensor): Input voxels.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy voxels.
    """
    if sigma > 0:
        return voxels + voxels.clone().normal_(0, sigma)
    return voxels


def log_metrics(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    time: float
) -> None:
    """
    Logs the metrics for each epoch.

    Args:
        logger (logging.Logger): The logger object used for logging.
        epoch (int): The current epoch number.
        train_metrics (dict): The metrics for the training set.
        val_metrics (dict): The metrics for the validation set.
        sample_metrics (dict): The metrics for the sample set.
        time (float): The time taken for the epoch.

    Returns:
        None
    """

    all_metrics = [train_metrics, val_metrics]
    metrics_names = ["train", "val"]

    logger.info(f"epoch: {epoch} ({time:.2f}s)")
    for (split, metric) in zip(metrics_names, all_metrics):
        if metric is None:
            continue
        # str_ += "\n"
        str_ = f"[{split}]"
        for k, v in metric.items():
            if k == "loss":
                str_ += f" | {k}: {v:.2f}"
            else:
                str_ += f" | {k}: {v:.4f}"
        logger.info(str_)


if __name__ == "__main__":
    main()
