import os

import numpy as np
from Bio import SearchIO, SeqIO
from pyhmmer.plan7 import HMMFile
import subprocess

def run_hmmscan(base, protein_seq, PROT_ID):
    """
    Runs hmmscan on a protein sequence and returns the qresult object (parsed hmmscan output).

    Args:   PROT_ID (str): protein ID
            protein_fasta_dict (dict): dictionary of protein sequences, keys are protein IDs
            base (str): path to the proemb directory

    Returns:
        qresult (biopython parsing object): hmmscan output
    """

    # save SCOPid seq as fasta
    SeqIO.write(protein_seq, f"{PROT_ID}.fasta", "fasta")

    # RUN HMMSCAN using subprocess
    hmm_scan = subprocess.Popen(
        ["hmmscan", "-o", f"{PROT_ID}.hmmscan", f"{base}pfam-hmm/Pfam-A.hmm", f"{PROT_ID}.fasta"])
    # wait until hmmscan is done
    hmm_scan.wait()

    qresult = SearchIO.read(f'{PROT_ID}.hmmscan', 'hmmer3-text')

    # remove fasta and hmmscan files
    os.remove(f"{PROT_ID}.fasta")
    os.remove(f"{PROT_ID}.hmmscan")

    return qresult


def setup_hmm_dict(base):
    """
    Returns a dictionary of HMMs, keys are HMM names.

    Args:   base (str): path to the proemb directory

    Returns:
        hmm_dict (dict): dictionary of HMMs, keys are HMM names
    """

    with HMMFile(f"{base}pfam-hmm/Pfam-A.hmm") as hmm_file:
        hmms = list(hmm_file)
    hmm_dict = {hmm.name: hmm for hmm in hmms}

    return hmm_dict

def get_seq_hmm_probabilities_for_hsp(qresult, hmm_dict, hit_index=0, hsp_index=0):
    """
    Returns the probability distribution of the 20 amino acids for each residue in the protein sequence.
    The probability distributions are taken from the HMM match/insertion states.

    Args:
        qresult (biopython parsing object): hmmscan output
        hmm_dict (dict): dictionary of HMMs, keys are HMM names
        hit_index (int, optional): Index of the hit. Default is 0.
        hsp_index (int, optional): Index of the HSP. Default is 0.

    Returns:
        np.array: Probability distribution of the 20 amino acids for each residue in the protein sequence
        np.array: Mask of the protein sequence, 1 where we found hmm info, 0 where we didn't

    Bipython parses the hmmscan output and also changes the indexes to 0-based.
    for example (hmmfrom, hhm to) = (3,78) to (2,78)
    (2, 78) and not (2, 77) because when we select a list with a[2,78] it is the equivalent of [2, 78)
    this is done for all the ranges, hmm hit range, query range and envelope range

    hsp = (high-scoring pair), 1 hit can have multiple hsps (basically different alignment positions and different evalue)

    Pyhmmer doc: 'The property exposes a matrix of shape (M+1, K),
    with one row per node and one column per alphabet symbol'
    """

    # by default, we look at only 1 hit and take the fist hsp
    selected_hsp = qresult.hits[hit_index].hsps[hsp_index]
    hmm_name = selected_hsp.hit_id

    hmm_hit_range, hmm_hit, seq_hit_range, seq_hit = (
        selected_hsp.hit_range, selected_hsp.hit.seq,
        selected_hsp.query_range, selected_hsp.query.seq
    )

    hmm = hmm_dict[hmm_name.encode()]
    seq_probabilities = np.zeros((qresult.seq_len, 20))

    seq_prob_index, hmm_hit_index = seq_hit_range[0], hmm_hit_range[0]

    # we go along the protein sequence
    for hit_index in range(len(seq_hit)):

        # ignore gaps (sequence deletions compared to the hmm)
        if seq_hit[hit_index] == '-':
            # we go to the next state in the hmm
            hmm_hit_index += 1
            continue

        # match
        if hmm_hit[hit_index] != '.':
            # +1 because the first column is the start state
            seq_probabilities[seq_prob_index] = hmm.match_emissions[hmm_hit_index + 1]
            hmm_hit_index += 1
        else:
            # insert, we don't go to the next state in the hmm
            seq_probabilities[seq_prob_index] = hmm.insert_emissions[hmm_hit_index + 1]

        seq_prob_index += 1

    # 1 where we added prob, 0 where we didn't
    hit_mask = np.where(seq_probabilities.sum(axis=1) > 0, 1, 0)

    return seq_probabilities, hit_mask

def get_combined_seq_hmm_probabilities(qresult, hmm_dict, evalue_treshold=0.01, degree_of_overlap=False):

    """
    Returns the probability distribution of the 20 amino acids for each residue in the protein sequence, by combining
    the probability distributions from all the hits. The default is 0.01, meaning that on average, about 1 false
    positive would be expected in every 100 searches with different query sequences.

    Args:
        qresult (biopython parsing object): hmmscan output
        hmm_dict (dict): dictionary of HMMs, keys are HMM names
        evalue_treshold (float, optional): E-value treshold for hits. Default is 0.01.
        degree_of_overlap (bool, optional): If True, returns the degree of overlap of the hmm hits. Default is False.
        This is useful to visualise how many hits return evolutionary information for the same residue.

    Returns:
        np.array: Probability distribution of the 20 amino acids for each residue in the protein sequence
        np.array: Mask of the protein sequence, 1 where we found hmm info, 0 where we didn't
        float: Degree of overlap of the hmm hits. Only returned if degree_of_overlap is True.
    """

    seq_probabilities = np.zeros((qresult.seq_len, 20))
    hit_mask = np.zeros(qresult.seq_len)
    overlap_mask = np.zeros(qresult.seq_len)

    for hit_index, hsps in enumerate([hit.hsps for hit in qresult.hits]):

        if len(hsps) == 0:
            # no hsps for this hit
            continue

        # select the hps with the lowest evalue
        hsp_index = np.argmin([hps.evalue for hps in hsps])
        if hsps[hsp_index].evalue > evalue_treshold:
            continue

        seq_probabilities_for_hsp, hit_mask_hsp = get_seq_hmm_probabilities_for_hsp(qresult, hmm_dict,
                                                                                    hit_index, hsp_index)

        # we add probabilities only where we have not added probabilities before
        for i in range(len(seq_probabilities)):
            if seq_probabilities[i].sum() == 0 and seq_probabilities_for_hsp[i].sum() > 0:
                seq_probabilities[i] = seq_probabilities_for_hsp[i]
            elif seq_probabilities_for_hsp[i].sum() > 0 and seq_probabilities[i].sum() > 0:
                overlap_mask[i] = 1

        hit_mask = np.where(hit_mask == 1, hit_mask, hit_mask_hsp)

    if degree_of_overlap:
        return seq_probabilities, hit_mask, overlap_mask.sum() / hit_mask.sum()

    return seq_probabilities, hit_mask