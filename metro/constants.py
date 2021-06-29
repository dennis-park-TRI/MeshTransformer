# If W&B is enabled, the output directory is set to `./output/<WANDB_ID>-<DATETIME>`
# e.g., "./output/tcgjbluf-20210629_061603"
ROOT_OUTPUT_DIR = "output"

# This serves the purpose of `/tmp/`, but is shared across nodes when mulitple instances provisioned.
TMP_DIR = "/mnt/fsx/tmp"
