#!/bin/bash

#SBATCH -A nhr-rtindex
#SBATCH -J rtindex
#SBATCH -e rtindex.%j.err
#SBATCH -o rtindex.%j.out
#SBATCH -p smallgpu
#SBATCH -n 1
#SBATCH -t 5:00:00
#SBATCH --mem 64G
#SBATCH --gres=gpu:nvidia_a40:1

ml devel/CMake
ml use /apps/easybuild/2023/cuda/modules/all/
ml system/CUDA

srun build/index_prototype
