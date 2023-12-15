import argparse
import json
import os
import sys

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from proemb.utils.enzyme_util import PDB_OBSOLETE_REMAP

from Bio import SeqIO
from tqdm import tqdm

from proemb.utils.hmm_util import run_hmmscan
import pickle as pkl
import pandas as pd
import proemb.utils.interface_util as iu


"""
This script generates the HMMSCAN output for the different datasets. It applies the HMMSCAN to each sequence
and saves the output to a dictionary. The dictionary is saved as a pickle file. The keys of the dictionary are
made in such a way so they can be easily recognised during training.
"""

def write_enzyme_hmm(root='../../data/'):
    with open(f'{root}enzyme/metadata/base_split.json') as json_file:
        data = json.load(json_file)

    # all the proteins we need to generate fasta files for
    data = data['train'] + data['valid'] + data['test']

    pdb_fasta_dict = SeqIO.to_dict(SeqIO.parse(f'{root}enzyme/metadata/pdb_seqres.txt', "fasta"))
    hmm_output = {}
    for id in tqdm(data, total=len(data)):
        prot_id = id if id not in PDB_OBSOLETE_REMAP else PDB_OBSOLETE_REMAP[id]
        qresult = run_hmmscan(root, pdb_fasta_dict[prot_id], prot_id)

        if not qresult.hits:
            print(f"Skipping {prot_id} because no hits were found")
            hmm_output[id] = None
            continue
        hmm_output[id] = qresult

    # save the hmm output
    with open(f'{root}enzyme/metadata/pfam_hmm_hits_output.pkl', 'wb') as f:
        pkl.dump(hmm_output, f)


def write_homology_hmm(root='../../data'):
    df = pd.read_csv(f'{root}/HomologyTAPE/training.txt', delimiter='\t', header=None, names=['scopid', 'name'])
    data = df['scopid'].values
    scop_fasta = SeqIO.to_dict(SeqIO.parse(f'{root}/HomologyTAPE/astral-scopdom-seqres-gd-sel-gs-bib-95-1.75.fa',
                                           "fasta"))
    hmm_output = {}
    for id in tqdm(data, total=len(data)):
        prot_id = id if id in scop_fasta else 'g' + id[1:]
        qresult = run_hmmscan(root, scop_fasta[prot_id], prot_id)
        if not qresult.hits:
            print(f"Skipping {prot_id} because no hits were found")
            hmm_output[id] = None
            continue
        hmm_output[id] = qresult

    # save the hmm output
    with open(f'{root}HomologyTAPE/pfam_hmm_hits_output.pkl', 'wb') as f:
        pkl.dump(hmm_output, f)


def write_interface_hmm(base_pth='../../data/masif-ppis', root='../../data/'):
    prot_list = [line.rstrip('\n') for line in open(f'{base_pth}/lists/training.txt')]
    prot_list = [f'{id.split("_")[0].lower()}_{chain}' for id in prot_list for chain in id.split('_')[1]]

    pdb_fasta_dict = SeqIO.to_dict(SeqIO.parse(f'{base_pth}/fasta/pdb_seqres.txt', "fasta"))
    hmm_output = {}
    for prot_id in tqdm(prot_list, total=len(prot_list)):

        pdb_id, chain = prot_id.split('_')
        remapped = iu.PDB_OBSOLETE_REMAP.get(pdb_id.lower(), pdb_id.lower())

        if f'{remapped}_{chain}' not in pdb_fasta_dict:
            print(f"Skipping {prot_id} because no fasta file was found")
            hmm_output[prot_id] = None
            continue
        qresult = run_hmmscan(root, pdb_fasta_dict[f'{remapped}_{chain}'], remapped)

        if not qresult.hits:
            print(f"Skipping {prot_id} because no hits were found")
            hmm_output[prot_id] = None
            continue
        hmm_output[prot_id] = qresult

    with open(f'{base_pth}/pfam_hmm_hits_output.pkl', 'wb') as f:
        pkl.dump(hmm_output, f)


def write_flip_hmm(base_pth='../../data/FLIP', hmm_path='../../data/'):
    hmm_output = {}

    # for AAV we run an HMM scan for each type of sequence length
    # the HMMSCAN is the same for all the sequences that have the same length
    # we do it for different lengths, so we can map the HMM probabilities to the correct position
    df = pd.read_csv(f'{base_pth}/aav/full_data.csv', low_memory=False).full_aa_sequence
    seq_to_len_map = {}
    for seq in df:
        seq_to_len_map[len(seq)] = seq

    for length, seq in tqdm(seq_to_len_map.items(), total=len(seq_to_len_map)):
        id = f"aav-length={length}"
        hmm_output[id] = run_hmmscan(hmm_path, SeqRecord(Seq(seq), id=id), "aav")

    # for GB1 all sequences have the same length, so we only run one HMMSCAN
    df = pd.read_csv(f'{base_pth}/gb1/four_mutations_full_data.csv', low_memory=False).sequence
    hmm_output['gb1'] = run_hmmscan(hmm_path, SeqRecord(Seq(df[0]), id='gb1'), 'gb1')

    # for meltome we have to go through all the sequences in turn
    df = pd.read_csv(f'{base_pth}/meltome/splits/mixed_split.csv').sequence
    for seq in tqdm(df, total=len(df)):
        id = f"meltome-seq={seq}"
        qresult = run_hmmscan(hmm_path, SeqRecord(Seq(seq), id=id), "meltome")
        if not qresult.hits:
            print(f"Skipping {id} because no hits were found", flush=True)
            hmm_output[id] = None
            continue
        hmm_output[id] = qresult

    with open(f'{base_pth}/pfam_hmm_hits_output.pkl', 'wb') as f:
        pkl.dump(hmm_output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--hmm_path', type=str)

    args = parser.parse_args()

    # args to dict
    args = vars(args)

    # write_enzyme_fasta()
    # write_enzyme_hmm(t)
    # write_homology_hmm()
    write_flip_hmm(args['path'], args['hmm_path'])
    # write_interface_hmm()
