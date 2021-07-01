import logging
import subprocess

LOG = logging.getLogger(__name__)


def s3_copy(source_path, target_path, verbose=False):
    """Copy single file from local to s3, s3 to local, or s3 to s3.

    Parameters
    ----------
    source_path: str
        Path of file to copy

    target_path: str
        Path to copy file to

    verbose: bool, default: True
        If True print some helpful messages

    Returns
    -------
    bool: True if successful
    """
    success = False
    command_str = "aws s3 cp --acl bucket-owner-full-control {} {}".format(source_path, target_path)
    LOG.info("Copying file with '{}'".format(command_str))
    try:
        subprocess.check_output(command_str, shell=True)
        success = True
    except subprocess.CalledProcessError as e:
        success = False
        LOG.error("{} failed with error code {}".format(command_str, e.returncode))
        LOG.error(e.output)
    if verbose:
        LOG.info("Done copying file")

    return success


def sync_output_dir_s3(cfg):
    import pdb; pdb.set_trace()
    output_dir = cfg.OUTPUT_DIR

    if cfg.WANDB.ENABLED:
        if comm.is_main_process():
            assert os.path.basename(output_dir) == wandb.run.id, f"{output_dir}, {wandb.run.id}"
            wandb_run_dir = wandb.run.dir
            if wandb_run_dir.endswith('/files'):  # wandb 0.10.x
                wandb_run_dir = wandb_run_dir[:-6]
            datetime_str, wandb_run_id = wandb_run_dir.split('-')[-2:]
            assert wandb_run_id == wandb.run.id
        else:
            wandb_run_id, datetime_str = None, None
        wandb_run_id = MPI.COMM_WORLD.bcast(wandb_run_id, root=0)
        datetime_str = MPI.COMM_WORLD.bcast(datetime_str, root=0)
        if wandb_run_id is None:
            LOG.critical("W&B run ID is None. Something's wrong.")
            assert wandb_run_id is not None

        _tar_output_dir = '-'.join([wandb_run_id, datetime_str])
        tar_output_dir = os.path.join(cfg.OUTPUT_DIR_SYNC.ROOT_IN_S3, _tar_output_dir)

        if comm.is_main_process():
            LOG.info(f"Syncing output_dir: {output_dir} -> {tar_output_dir}")
            sync_dir(output_dir, tar_output_dir)

            # Sync W&B run dir.
            tar_wandb_run_dir = os.path.join(tar_output_dir, 'wandb')
            LOG.info(f"Syncing W&B run dir: {wandb.run.dir} -> {tar_wandb_run_dir}")
            sync_dir(wandb.run.dir, tar_wandb_run_dir)
        elif comm.get_local_rank() == 0:
            # local master -- only sync the log files
            log_output_dir, log_tar_output_dir = os.path.join(output_dir, 'logs'), os.path.join(tar_output_dir, 'logs')
            LOG.info(f"Syncing log output_dir: {log_output_dir} -> {log_tar_output_dir}")
            sync_dir(log_output_dir, log_tar_output_dir)
    else:
        # User is responsible for setting a distinctive cfg.OUTPUT_DIR that is unqiue for this run;
        # this will be used as a target output_dir.
        output_dir = os.path.basename(cfg.OUTPUT_DIR)
        tar_output_dir = os.path.join(cfg.OUTPUT_DIR_SYNC.ROOT_IN_S3, output_dir)

        if comm.is_main_process():
            LOG.info(f"Syncing output_dir: {output_dir} -> {tar_output_dir}")
            sync_dir(output_dir, tar_output_dir)
        elif comm.get_local_rank() == 0:
            # local master -- only sync the log files
            log_output_dir, log_tar_output_dir = os.path.join(output_dir, 'logs'), os.path.join(tar_output_dir, 'logs')
            LOG.info(f"Syncing log output_dir: {log_output_dir} -> {log_tar_output_dir}")
            sync_dir(log_output_dir, log_tar_output_dir)
    comm.synchronize()
