#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=tensorboard
#SBATCH --partition=batch
#SBATCH --time=60:00:00
#SBATCH --mem=100GB              # make sure we use all the memory available

#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8


module load conda
source activate prose


cd /scratch/ii1g17/protein-embeddings

# ----------------TENSORBOARD----------------------------
tensorboard --logdir runs/multitask/   --bind_all --max_reload_threads 8 --samples_per_plugin scalars=999999999 
#----------------TENSORBOARD----------------------------
