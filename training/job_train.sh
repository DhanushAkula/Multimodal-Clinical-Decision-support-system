#!/bin/bash
#SBATCH --job-name=dr_minerva
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --output=/scratch/arutla.a/medpix-outputs/logs/train_%j.out
#SBATCH --error=/scratch/arutla.a/medpix-outputs/logs/train_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

# Load modules
module load anaconda3/2024.06
module load cuda/12.8.0

# Activate conda environment
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate dr_minerva

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Navigate to project
cd /home/arutla.a/medpix-project/training

# Create output directories
mkdir -p /scratch/arutla.a/medpix-outputs/checkpoints
mkdir -p /scratch/arutla.a/medpix-outputs/logs
mkdir -p /scratch/arutla.a/medpix-outputs/results

# Run training
echo "Starting training at $(date)"
python train_dr_minerva.py

EXIT_CODE=$?

echo "Finished at $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
