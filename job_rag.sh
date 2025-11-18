#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/scratch/%u/medpix-outputs/logs/rag-%j.out

module load anaconda3/2024.06 cuda/12.8.0
source activate medpix
cd /home/arutla.a/medpix-project
python build_rag.py
