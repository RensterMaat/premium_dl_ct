import wandb
import numpy as np
from cv import train


sweep_config = {
    "method": "random",
    "metric": {"name": "valid_patient_auc", "goal": "maximize"},
    "parameters": {
        "aggregation_function": {"values": ["min"]},
        "optimizer": {"values": ["adamw"]},  # , "sgd"]},
        "weight_decay": {"values": [0.0, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
        "model": {
            "values": [
                "densenet121",
                "densenet169",
                "densenet201",
                # "efficientnet-b0",
                # "efficientnet-b1",
                # "efficientnet-b2",
                "SEResNet50",
                "SEResNet101",
                "SEResNet152",
            ]
        },
        "dropout": {"min": 0.0, "max": 0.7},
        # "momentum": {"values": [0.0, 0.5, 0.9, 0.99]},
        "pretrained": {"values": [False]},  # , True]},
        "learning_rate_max": {
            "min": -13.82,  # corresponds to 10e-6
            "max": -9.210,
            # "max": -6.908,  # corresponds to 10e-3
            # "max": -4.605,  # corresponds to 10e-2
            "distribution": "log_uniform",
        },
        "roi_selection_method": {"values": ["crop"]},  # , "zoom"]},
        "roi_size": {"values": [50, 100, 150]},
        # "margin": {"values": [0, 10, 50]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="sweep8")

wandb.agent(sweep_id, function=train)
