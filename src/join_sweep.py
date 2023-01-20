import wandb
from cv import train

wandb.agent("lq38gvsl", function=train, project="sweep13")
