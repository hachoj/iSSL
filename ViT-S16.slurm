#!/bin/bash

#SBATCH --job-name=ViTS-16-RMS-no-bias
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200

#SBATCH --time=48:00:00
#SBATCH --output=output/logs/%x-%j.out
#SBATCH --error=output/logs/%x-%j.err
mkdir -p output/logs

#SBATCH --mail-user=7023555729@tmomail.net
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=REQUEUE,INVALID_DEPEND,STAGE_OUT
#SBATCH --mail-type=TIME_LIMIT,TIME_LIMIT_80,TIME_LIMIT_50

hostname;date;pwd

module purge
module load mamba
module load cuda/12.8
mamba activate issl
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}
export OMP_NUM_THREADS=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + ($SLURM_JOB_ID % 20000)))
RDZV_ID=$SLURM_JOB_ID

# added debugging
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# Copy dataset to node-local scratch
LOCAL_DATA_PATH="/scratch/local/$SLURM_JOB_ID/ImageNet1k"
SOURCE_DATA_PATH="/blue/weishao/chojnowski.h/data/imagenet-1k"
mkdir -p "$LOCAL_DATA_PATH"
echo "Copying dataset to local scratch..."
rsync -a --info=progress2 "$SOURCE_DATA_PATH/" "$LOCAL_DATA_PATH/"
echo "Dataset copied to local scratch: $LOCAL_DATA_PATH"
export LOCAL_DATA_PATH
trap 'rm -rf "$LOCAL_DATA_PATH"' EXIT

srun --export=ALL --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$RDZV_ID \
    scripts/train_MAE.py
