#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH --time=2:00:00
#SBATCH -o hybrid_outputs/slurm-%j.out

source ~/.bashrc
micromamba activate hybrid-lucas

mkdir -p hybrid_outputs
combo="$1"
echo "Combo: $combo"
IFS=':' read -r temp fold md mt sat targets <<< "$combo"

cmd=(python 2_hybrid.py --temp "$temp" --fold "$fold" --md "$md" --mt "$mt" --sat "$sat" --targets "$targets")
"${cmd[@]}"

