#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -J nsys_conv

module purge
module load cuda/12.2   

export TMPDIR=$PWD/tmp
mkdir -p $TMPDIR

nsys profile --trace=cuda,nvtx,osrt 
  -o nsys_total \
  ./conv_experiment 4096
