import wandb
from cv import train

wandb.agent(
    '2k4tt472',
    function=train,
    project='sweep4'
)
