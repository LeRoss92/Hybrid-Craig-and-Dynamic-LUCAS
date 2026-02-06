#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH -o 6_hybrid_outputs/slurm-%j.out

source ~/.bashrc
micromamba activate DPL_JAX_copy

mkdir -p 6_hybrid_outputs
combo="$1"
echo "Combo: $combo"
IFS=':' read -r temp fold md mt sat targets <<< "$combo"

cmd=(python 6_hybrid.py --temp "$temp" --fold "$fold" --md "$md" --mt "$mt" --sat "$sat" --targets "$targets")
"${cmd[@]}"

