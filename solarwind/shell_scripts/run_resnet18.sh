#!/usr/bin/sh

module purge
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4

export CONDA_ENVS_PATH=/nobackupnfs1/sroy14/.conda/envs
export CONDA_PKGS_DIRS=/nobackupnfs1/sroy14/.conda/pkgs
export NODE_RANK=$1
export NUM_NODES=$2
export WORLD_SIZE=$3
export RDZV_ADDR=$4
export RDZV_PORT=$5
export RDZV_ID=$6
export PBS_JOBID=$7
export TMPDIR=$8
source activate heliofm
sleep 5

export PYTHONPATH=$PWD
echo 'Number of nodes '$NUM_NODES
echo 'World Size '$WORLD_SIZE
echo 'Rendezvous address '$RDZV_ADDR
echo 'Rendezvous port '$RDZV_PORT
echo 'Rendezvous ID '$RDZV_ID
echo 'job ID '$PBS_JOBID

echo 'current directory is'$PWD

torchrun \
  --nnodes $NUM_NODES\
  --node-rank $NODE_RANK\
  --nproc_per_node 4\
  --rdzv_id $RDZV_ID \
  --rdzv_endpoint "$RDZV_ADDR:$RDZV_PORT" \
  --rdzv_backend c10d\
  train_baseline.py \
    --config_path ./ds_configs/config_resnet_18.yaml \
    --gpu \
    --wandb