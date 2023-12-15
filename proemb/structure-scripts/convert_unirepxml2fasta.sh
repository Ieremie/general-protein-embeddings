#!/bin/bash

#SBATCH --nodes=1                # node count
#SBATCH --partition=highmem
#SBATCH --time=60:00:00
#SBATCH --mem=200G #JUST IN CASE

module load conda
source activate prose

cd $HOME/protein-embeddings/proemb/structure-scripts

python  unirefxml2fasta.py "/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/uniref90.xml" -o="/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/uniref90.fasta" -l="UniRef90"

