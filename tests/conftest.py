import os

os.environ["WANDB_MODE"] = "disabled"
os.environ.pop("WANDB_API_KEY", None)
