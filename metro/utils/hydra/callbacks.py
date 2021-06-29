import logging
import os
import tempfile

import wandb
from hydra.experimental.callback import Callback

from detectron2.utils import comm as d2_comm
from detectron2.utils.logger import setup_logger

from metro.constants import ROOT_OUTPUT_DIR, TMP_DIR
from metro.utils.comm import mpi_broadcast
from metro.utils.s3 import s3_copy


@mpi_broadcast
def init_wandb(cfg):
    if cfg.WANDB.DRYRUN:
        os.environ['WANDB_MODE'] = 'dryrun'

    tags = cfg.WANDB.TAGS # TODO: how to update tags from multiple configs?
    wandb.init(project=cfg.WANDB.PROJECT, config=cfg, tags=tags)

    wandb_run_dir = wandb.run.dir
    if wandb_run_dir.endswith('/files'):  # wandb 0.10.x
        wandb_run_dir = wandb_run_dir[:-6]
    datetime_str, wandb_run_id = wandb_run_dir.split('-')[-2:]
    assert wandb_run_id == wandb.run.id

    output_dir = os.path.join(ROOT_OUTPUT_DIR, '-'.join([wandb_run_id, datetime_str]))
    return output_dir


class WandbOutputDirCallback(Callback):
    """If W&B is enabled, then
        1) initialize W&B,
        2) derive the path of output directory using W&B ID,
        3) and set it as hydra working directory.
    """
    def on_run_start(self, config, **kwargs):
        if not config.WANDB.ENABLED:
            return

        output_dir = init_wandb(config)
        config.hydra.run.dir = output_dir


class D2LoggerCallback(Callback):
    def on_run_start(self, config, **kwargs):
        rank = d2_comm.get_rank()
        log_output_dir = os.path.join(config.hydra.run.dir, 'logs')
        setup_logger(log_output_dir, distributed_rank=rank, name="hydra")
        setup_logger(log_output_dir, distributed_rank=rank, name="detectron2", abbrev_name="d2")
        setup_logger(log_output_dir, distributed_rank=rank, name="metro")


@mpi_broadcast
def maybe_download_ckpt_from_s3(cfg):
    """If the checkpoint is an S3 path, the main process download the weight under, by default, `/mnt/fsx/tmp/`.

    NOTE: All workers must update `cfg.MODEL.CHECKPOINT` to use the new path.
    """
    LOG = logging.getLogger(__name__)

    ckpt_path = cfg.model.CHECKPOINT
    if not ckpt_path.startswith("s3://"):
        return ckpt_path

    os.makedirs(TMP_DIR, exist_ok=True)
    _, ext = os.path.splitext(ckpt_path)
    tmp_path = tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=ext).name
    LOG.info(f"Downloading initial weights from {ckpt_path} to {tmp_path}")
    s3_copy(ckpt_path, tmp_path)
    LOG.info("Done.")

    return tmp_path


class DownloadCkptCallback(Callback):
    """
    If the checkpoint (`config.model.CHECKPOINT`) is an S3 path, then downloaded it and replace the path with
    local path.
    """
    def on_run_start(self, config, **kwargs):
        new_ckpt_path = maybe_download_ckpt_from_s3(config)
        config.model.CHECKPOINT = new_ckpt_path
