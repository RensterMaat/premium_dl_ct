import wandb
from cv import train

wandb.agent(
    'wos4wef0',
    function=train,
    project='sweep6'
)
