#!/bin/bash

#SBATCH --job-name=ar_spt             # Job name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --gpus-per-node=1                      # GPUs per node
#SBATCH --cpus-per-task=12                      # Number of CPUs per task
#SBATCH --mem=32G                             # Total memory per node (32GB)
#SBATCH --output=slurm_logs/%j_%x.out           # Standard output
#SBATCH --error=slurm_logs/%j_%x.err            # Standard error
#SBATCH --time=24:00:00                         # Time limit (hh:mm:ss)
#SBATCH --exclusive                             # Exclusive node allocation
#SBATCH --ntasks-per-node=1                     # Number of tasks per node

source /lustre/fs0/scratch/shared/miniconda3/bin/activate && conda activate srm-heliofm

python train_baselines.py --config_path ./ds_configs/config_spectformer_ar_sta.yaml --gpu --wandb
