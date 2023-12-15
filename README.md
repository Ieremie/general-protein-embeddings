# Structure, Surface and Interface Informed Protein Language Model

This repository contains the implementation of various protein language models trained on reduced amino acid alphabets, along with the notebooks to recreate the figures found in the paper.

**For more details, see:** [NeurIPS-MLSB2023](https://www.mlsb.io/papers_2023/Structure_Surface_and_Interface_Informed_Protein_Language_Model.pdf). 

## About
Language models applied to protein sequence data have gained a lot of interest in recent years, mainly due to their ability to capture complex patterns at the protein sequence level. However, their understanding of why certain evolution-related conservation patterns appear is limited. This work explores the potential of protein language models to further incorporate intrinsic protein properties stemming from protein structures, surfaces, and interfaces. The results indicate that this multi-task pretraining allows the PLM to learn more meaningful representations by leveraging information obtained from different protein views. We evaluate and show improvements in performance on various downstream tasks, such as enzyme classification, remote homology detection, and protein engineering datasets. 

## Datasets
The model is trained and evaluated using publicly available datasets:
- PLM pretraining dataset: [Uniref90](https://www.uniprot.org/help/downloads)
- Enzyme Commission (EC) dataset: [IEConv_proteins](https://github.com/phermosilla/IEConv_proteins)
- Fold recognition dataset: [TAPE](https://github.com/songlab-cal/tape)
- FLIP benchmark datasests: [FLIP](https://github.com/J-SNACKKB/FLIP)

All of these datasets can be downloaded using the release feature on Github, apart from Uniref90 which is very large. This can be downloaded and then modified using our dataset script.

## Pretraining PLM
To pretrain the protein language model you can run [`train_prose_multitask.py`](./proemb/train_prose_multitask.py).
The implementation uses multiple GPUs and can be run on a single machine or on a cluster. The scripts for running the
file on a cluster can be found at [`iridis-scripts`](./proemb/iridis-scripts/multitask). The progress of the training
can be monitored using [`tensorboard.sh`](./proemb/iridis-scripts/tensorboard.sh). All trained models can be downloaded in the release section.

## Finetuning on downstream tasks
After pretraining the protein language model, you can finetune it on downstream tasks. You can do this by running
the following python files:
- [`train_enzyme.py`](./proemb/train_enzyme.py) for the EC dataset
- [`train_fold.py`](./proemb/train_fold.py) for the Fold recognition dataset
- [`train_flip.py`](./proemb/train_flip.py) for the FLIP benchmark datasets

If you want to run these experiments on a cluster, take a look in the folder: [`iridis-scripts`](./proemb/iridis-scripts)

## Reproducing plots from the paper
To reproduce the plots for the protein embedding projection using TSNE, use the notebook [`scop-tsne.ipynb`](./proemb/media/scop-tsne.ipynb).

## Embedding protein sequences
If you want to embedd a set of protein sequences using any of the models, you can use the [`embedd.py`](./proemb/embedd.py) script. You only need to provide a fasta file.

#### This code contains various bits of code taken from other sources. If you find the repo useful, please cite the following work too:

- Surface generation code: [MASIF](https://github.com/LPDI-EPFL/masif)
- LDDT calculation: [AlphaFold](https://github.com/deepmind/alphafold)
- Model archiecture and uniprot tokenization: [Prose](https://github.com/tbepler/prose)

## Authors
Ioan Ieremie, Rob M. Ewing, Mahesan Niranjan

## Citation
```
to be added
```

## Contact
ii1g17 [at] soton [dot] ac [dot] uk
