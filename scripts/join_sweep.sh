#!/bin/bash

#SBATCH -t 100:00:00 
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --gpus-per-node=1
#SBATCH --gres=tmpspace:10G

env
source /hpc/dla_patho/premium/rens/miniconda3/etc/profile.d/conda.sh
conda activate rens

cd /hpc/dla_patho/premium/rens/premium_dl_ct/src

# chmod +x cv.py
# python cv.py

chmod +x join_sweep.py
python join_sweep.py