#!/bin/bash
#SBATCH --job-name=dr_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --output=/scratch/arutla.a/medpix-outputs/logs/eval_%j.out
#SBATCH --error=/scratch/arutla.a/medpix-outputs/logs/eval_%j.err

echo "Evaluation Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

# Load modules
module load anaconda3/2024.06
module load cuda/12.8.0

# Activate conda
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate dr_minerva

# Set environment
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Navigate and run
cd /home/arutla.a/medpix-project/training

echo "Running evaluation..."
python eval_dr_minerva.py

EXIT_CODE=$?

echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
