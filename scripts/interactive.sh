#!/bin/bash

#SBATCH -t 200:00:00 
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -c 12
#SBATCH --gpus-per-node=1
#SBATCH --gres=tmpspace:10G

bash

