#!/bin/bash
#SBATCH --job-name=protein-surfaces

#SBATCH --partition=batch
#SBATCH --open-mode=append

#SBATCH --mem=25G
#SBATCH --time=24:00:00

#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8   


# pymesh does not work with older gcc
module load gcc/11.1.0  

module load conda
source activate prose

# run python code
cd $HOME/protein-embeddings/proemb/surfaces


# "$@" is equivalent to "$1" "$2" "$3"...
# arguments must be passed in quotes
python  process_surface_dataset.py "$@"
