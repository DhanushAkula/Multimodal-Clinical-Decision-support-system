#!/bin/bash
#SBATCH --job-name=eval_preds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/arutla.a/medpix-outputs/logs/eval_preds_%j.out
#SBATCH --error=/scratch/arutla.a/medpix-outputs/logs/eval_preds_%j.err

echo "=========================================="
echo "DR-Minerva Evaluation with Prediction Saving"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module load anaconda3/2024.06

# Activate dr_minerva environment
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate dr_minerva

# Run evaluation
cd /home/arutla.a/medpix-project/training
python eval_dr_minerva_save_preds.py

echo ""
echo "=========================================="
echo "Ended: $(date)"
echo "=========================================="
