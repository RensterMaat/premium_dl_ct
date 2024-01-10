import wandb
import numpy as np
from cv import train


sweep_config = {
    "method": "random",
    "metric": {"name": "valid_patient_auc", "goal": "maximize"},
    "parameters": {
        'lesion_target': {'values': ["lesion_response"]},
        'patient_target': {'values': ["response"]},
        "dim": {"values": [3]},
        "aggregation_function": {"values": ["min"]},
        "optimizer": {"values": ["adamw"]},  # , "sgd"]},
        "weight_decay": {
            "values": [
                # 0.0,
                1e-9,
                1e-7,
                1e-5,
                # 1e-3,
                # 1e-1
            ]
        },
        "model": {
            "values": [
                # "densenet121",
                # "densenet169",
                # "densenet201",
                # "efficientnet-b0",
                # "efficientnet-b1",
                # "efficientnet-b2",
                "SEResNet50",
                # "SEResNet101",
                # "SEResNet152",
                # 'SEResNext50',
                # 'SEResNext101',
                "vit_b_16",
                "vit_b_32",
                "vit_l_16",
                "vit_l_32",
            ]
        },
        "dropout": {"values": [0]},  # {"min": 0.0, "max": 0.7},
        # "momentum": {"values": [0.0, 0.5, 0.9, 0.99]},
        "pretrained": {"values": [False]},
        "learning_rate_max": {
            "values": [1e-5]
            # "min": -13.82,  # corresponds to 10e-6
            # "max": -9.210,
            # # "max": -6.908,  # corresponds to 10e-3
            # # "max": -4.605,  # corresponds to 10e-2
            # "distribution": "log_uniform",
        },
        "roi_selection_method": {"values": ["crop"]},  # , "zoom"]},
        "roi_size": {"values": [142]},  # 50, 100, 150]},
        # "margin": {"values": [0, 10, 50]},
        "n_forward_per_backwards": {"values": [1]},  # ,2,4,8,16,32,64]},
        "sampler": {
            "values": [
                # "label_stratified",
                # "label_organ_stratified",
                # "patient_grouped",
                "vanilla",
            ]
        },
        "test_center": {
            "values": [
                "umcu",
                "maxima",
                "amphia",
                "isala",
                "lumc",
                "mst",
                "vumc",
                "zuyderland",
                "radboud",
                "umcg",
            ]
        },
        "augmentation_noise_std": {"values": [0.001]},
        "inner_fold": {"values": [0, 1, 2, 3, 4]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="sweep25")

wandb.agent(sweep_id, function=train)
