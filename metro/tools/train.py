"""
Usage:
python metro/tools/train.py
python metro/tools/train.py \
    model.CHECKPOINT=s3://scratch-tri-global/dennis.park/tridet_output_dirs/104d4vvr-20210502_174921/model_final.pth
"""
import logging
from collections import defaultdict

import hydra
import torch
import wandb
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel

from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, get_event_storage

from metro.data import build_test_dataloader, build_train_dataloader
from metro.modeling.build import build_model
from metro.solver import build_lr_scheduler, build_optimizer
from metro.utils.comm import is_distributed
from metro.utils.s3 import sync_output_dir_s3
from metro.utils.setup import setup
from metro.utils.wandb_util import get_wandb_step, step_wandb

LOG = logging.getLogger('metro')


@hydra.main(config_path="../../configs", config_name="defaults")
def main(cfg):
    # 1. Set up.
    # 2. Build model.
    # 3. Load checkpoint.
    # 4. If `eval_only`, then do_test() and exit.
    # 5. If distributed, then wrap model with distributed data parallel.
    # 6. do_train()
    # 7. do_test()


    # 1.
    setup(cfg)

    # 2.
    # model: METRO network
    # body_model: SMPL or MANO
    # mesh: Mesh object (used to handle graph ops)

    # Requirement:
    #   In train mode (model.train()), the forward method must return a dict of losses.
    #   Each key must starts with "loss_".
    model, body_model, mesh_sampler = build_model(cfg)

    # 3.
    checkpoint_file = cfg.model.CHECKPOINT
    if checkpoint_file:
        Checkpointer(model).load(checkpoint_file)

    # 4.
    if cfg.EVAL_ONLY:
        test_results = do_test(cfg, model)
        return test_results, cfg

    # 5.
    if is_distributed():
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.optimizer.DDP_FIND_UNUSED_PARAMETERS
        )

    # 6 and 7.
    do_train(cfg, model)
    test_results = do_test(cfg, model)


def do_train(cfg, model):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = Checkpointer(model, './', optimizer=optimizer, scheduler=scheduler)
    max_iter = cfg.solver.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.solver.CHECKPOINT_PERIOD, max_iter=max_iter)

    writers = [CommonMetricPrinter(max_iter)] if comm.is_main_process() else []

    dataloader, dataset = build_train_dataloader(cfg)
    LOG.info("Starting training")
    storage = get_event_storage()

    # In mixed-precision training, gradients are scaled up to keep them from being vanished due to half-precision.
    # They're scaled down again before optimizers use them to compute updates.
    scaler = amp.GradScaler(enabled=cfg.solver.USE_MIXED_PRECISION)

    num_images_seen = 0
    optimizer.zero_grad()
    for data, iteration in zip(dataloader, range(max_iter)):
        # this assumes drop_last=True, so all workers has the same size of batch.
        num_images_seen += len(data) * comm.get_world_size()
        storage.step()

        with amp.autocast(enabled=cfg.solver.USE_MIXED_PRECISION):
            loss_dict = model(data)

        losses = sum(loss_dict.values())
        if not torch.isfinite(losses):
            LOG.critical(f"The loss DIVERGED: {loss_dict}")

        # Track losses for logging.
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        assert torch.isfinite(torch.as_tensor(list(loss_dict_reduced.values()))).all(), loss_dict_reduced

        scaler.scale(losses).backward()

        scaler.step(optimizer)
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        scaler.update()

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

        # Reset optimizer.
        optimizer.zero_grad()

        if iteration > 5 and (iteration % 20 == 0 or iteration == max_iter):
            for writer in writers:
                writer.write()

            if comm.is_main_process() and cfg.WANDB.ENABLED:
                wandb.log({"epoch": num_images_seen // len(dataset)}, step=storage.iter)
                wandb.log({"num_images_seen": num_images_seen}, step=storage.iter)

    if comm.is_main_process():  # TODO (dennis.park): is this necessary?
        periodic_checkpointer.step(storage.iter)

    if iteration > 0 and iteration % cfg.OUTPUT_DIR_SYNC.PERIOD == 0:
        sync_output_dir_s3(cfg)

    if cfg.TEST.ENABLED and (
        (iteration % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter) or \
        iteration in cfg.TEST.ADDITIONAL_EVAL_STEPS):
        do_test(cfg, model)

def do_test(cfg, model):
    raise NotImplementedError()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
    LOG.info("DONE.")
