#!/bin/sh

#SBATCH --job-name=jupyter

#SBATCH --nodes=1                # node count
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=60G
#SBATCH --time=60:00:00


module load conda
source activate prose

cd $HOME/protein-embeddings


#jupyter lab  --no-browser --port=8080
jupyter lab --no-browser --ip "*"
