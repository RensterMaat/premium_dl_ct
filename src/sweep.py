# https://www.paepper.com/blog/posts/hyperparameter-tuning-on-numerai-data-with-pytorch-lightning-and-wandb/
import wandb
from cv import train


sweep_config = {
    "method": "random",
    "metric": {"name": "valid_patient_auc", "goal": "maximize"},
    "parameters": {
        "aggregation_function": {"values": ["max", "min", "mean"]},
        "roi_size": {"values": [50, 100, 150]},
        "optimizer": {"values": ["adamw", "sgd"]},
        "weight_decay": {"min": 0.0, "max": 0.1},
        "model": {"values": ["densenet121", "densenet169", "densenet201"]},
        "dropout": {"min": 0.0, "max": 0.8},
        "momentum": {"values": [0.0, 0.5, 0.9, 0.99]},
        "pretrained": {"values": [True, False]},
        "learning_rate_max": {"min": -6, "max": -1, "distribution": "log_uniform"},
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="sweep1")

wandb.agent(sweep_id, function=train, count=5)
