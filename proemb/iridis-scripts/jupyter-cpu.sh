#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=jupyter

#SBATCH --partition=amd

#SBATCH --time=60:00:00
#SBATCH --mem=200GB              # make sure we use all the memory available        

#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=64


module load conda
source activate prose

cd $HOME/protein-embeddings/proemb

#jupyter lab  --no-browser --port=8080
jupyter lab --no-browser --ip "*"
