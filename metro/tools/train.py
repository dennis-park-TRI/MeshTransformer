"""
Usage:
python metro/tools/train.py
python metro/tools/train.py \
    model.CHECKPOINT=s3://scratch-tri-global/dennis.park/tridet_output_dirs/104d4vvr-20210502_174921/model_final.pth
"""
import hydra

from metro.utils.wandb_util import get_wandb_step, step_wandb
from metro.utils.setup import setup


@hydra.main(config_path="../../configs", config_name="defaults")
def main(cfg):
    setup(cfg)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
