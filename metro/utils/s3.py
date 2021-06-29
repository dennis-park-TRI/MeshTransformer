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
