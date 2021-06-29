# Buffer for training step
TRAINING_STEP = 0

def step_wandb():
    global TRAINING_STEP
    TRAINING_STEP += 1
    return TRAINING_STEP

def get_wandb_step():
    return TRAINING_STEP
