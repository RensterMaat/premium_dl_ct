import wandb
from cv import train

wandb.agent(
    'mx1dgla6',
    function=train,
    project='sweep7'
)
