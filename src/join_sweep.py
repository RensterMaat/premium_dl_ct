import wandb
from cv import train

wandb.agent("teja229p", function=train, project="sweep19")
