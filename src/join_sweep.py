import wandb
from cv import train

wandb.agent(
    'f1fi5fnn',
    function=train,
    project='sweep3'
)
