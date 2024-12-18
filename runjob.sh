#!/bin/bash

#SBATCH -A ptao_ml_protein_dynamics_0001
#SBATCH --job-name=UwU_Diffusion_^w^
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -o runjob.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=500G


# Specify node to run on
###SBATCH --nodelist=bcm-dgxa100-0003


# Number of nodes
nnodes=1
# Number of tasks per node
nproc_per_node=1




cd /users/gmongaras/work/DiffusionStuffUwU
python generate_batch.py
