#!/bin/bash
#SBATCH -p work
#SBATCH --mem=64G
#SBATCH -c 64
#SBATCH --time=48:00:00
#SBATCH --output=5_results/slurm-%j.out
#SBATCH --error=5_results/slurm-%j.err
#SBATCH --job-name=predict_ts

source ~/.bashrc
micromamba activate DPL_JAX_copy

echo "======================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "======================================"

# Ensure results directory exists
mkdir -p 5_results

# Run the prediction script
python 5_predict.py

echo "======================================"
echo "Job completed successfully"
echo "======================================"

