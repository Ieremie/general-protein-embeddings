#!/bin/bash

#SBATCH --job-name=protein-surfaces

#SBATCH --partition=batch,amd
#SBATCH --open-mode=append

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40    # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks), (40 CPUS max on batch nodes)


#SBATCH --mem=32G
#SBATCH --time=30:00:00


# pymesh does not work with older gcc
module load gcc/11.1.0  

module load conda
source activate prose

# run python code
cd $HOME/protein-embeddings/proemb/surfaces


# msms generates the surface(mesh) of a protein
export MSMS_BIN='/home/ii1g17/.conda/envs/prose/bin/msms'

# "$@" is equivalent to "$1" "$2" "$3"...
# arguments must be passed in quotes
srun python -u generate_scop_surfaces.py "$@"
