""" Adapted from https://github.com/tbepler/prose and
     https://github.com/songlab-cal/tape """
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pymesh
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser, Selection
from torch.utils.data.dataset import Dataset

from proemb import scop, fasta
from proemb.alphabets import Uniprot21
from pathlib import Path
import lmdb
import pickle
from PIL import Image
from proemb.utils.enzyme_util import PDB_OBSOLETE_REMAP, MIXED_SPLIT_CHAINS
from proemb.utils import surface_util
from proemb.utils.hmm_util import setup_hmm_dict,  get_seq_hmm_probabilities_for_hsp

from proemb.alphabets import PFAM_INDEX_TO_AA
import proemb.utils.interface_util as iu
from proemb.utils.surface_util import remove_chains_from_model, expand_struct_data_to_full_seq
import re


class SCOPeDataset:
    def __init__(self, path, alphabet=Uniprot21(), augment=None):

        self.augment = augment
        names, structs, sequences = self.load(path, alphabet)

        self.names = names
        self.x = [torch.from_numpy(x) for x in sequences]
        self.y = torch.from_numpy(structs)

        print('# loaded', len(self.x), 'SCOP train sequences', self.y.shape)

    def load(self, path, alphabet):

        with open(path, 'rb') as f:
            names, structs, sequences = scop.parse_astral(f, encoder=alphabet)

        names_filtered = []
        structs_filtered = []
        sequences_filtered = []
        for i in range(len(sequences)):
            s = sequences[i]
            if len(s) > 0:
                names_filtered.append(names[i])
                structs_filtered.append(structs[i])
                sequences_filtered.append(s)

        structs = np.stack(structs_filtered, 0)
        return names_filtered, structs, sequences_filtered

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i].long()
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[i]


class AllPairsDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, augment=None):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X) ** 2

    def __getitem__(self, k):
        n = len(self.X)

        # i,j will take all possible pairs, e.g. 0(i)->[0,1,2 ... n](j)
        i = k // n  # max i would be last element = n, because n*2/n= n
        j = k % n

        x0 = self.X[i].long()
        x1 = self.X[j].long()
        if self.augment is not None:
            x0 = self.augment(x0)
            x1 = self.augment(x1)

        y = self.Y[i, j]  # choosing the vector for seq_i level and picking the  level for j

        return x0, x1, y


class SCOPePairsDataset:

    def __init__(self, path, alphabet=Uniprot21()):
        table = pd.read_csv(path, sep='\t')
        x0 = [x.encode('utf-8').upper() for x in table['sequence_A']]
        self.x0 = [torch.from_numpy(alphabet.encode(x)) for x in x0]
        x1 = [x.encode('utf-8').upper() for x in table['sequence_B']]
        self.x1 = [torch.from_numpy(alphabet.encode(x)) for x in x1]

        self.y = torch.from_numpy(table['similarity'].values).long()

        print('# loaded', len(self.x0), 'SCOP test sequence pairs')

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, i):
        return self.x0[i].long(), self.x1[i].long(), self.y[i]


class ContactMapDataset:
    def __init__(self, path, root='/scratch/ii1g17/protein-embeddings/data/SCOPe/pdbstyle-2.06',
                 k=1, min_length=0, max_length=0, alphabet=Uniprot21(), augment=None):

        names, sequences, contact_maps = self.load(path, root, k=k)

        self.names = names
        self.x = [torch.from_numpy(alphabet.encode(s)) for s in sequences]
        self.y = contact_maps

        self.augment = augment

        self.min_length = min_length
        self.max_length = max_length

        self.fragment = False
        if self.min_length > 0 and self.max_length > 0:
            self.fragment = True

        print('# loaded', len(self.x), 'contact maps')

    def load(self, path, root, k=1):

        with open(path, 'rb') as f:
            names, sequences = fasta.parse(f)

        # find all of the contact maps and index them by protein identifier
        cmap_paths = glob.glob(root + os.sep + '*' + os.sep + '*.png')
        cmap_index = {os.path.basename(path).split('.cmap-')[0]: path for path in cmap_paths}

        # match the sequences to the contact maps
        names_filtered = []
        sequences_filtered = []
        contact_maps = []
        for (name, seq) in zip(names, sequences):
            name = name.decode('utf-8')
            pid = name.split()[0]
            if pid not in cmap_index:
                # try changing first letter to 'd'
                # required for some SCOPe identifiers
                pid = 'd' + pid[1:]

            path = cmap_index[pid]
            im = np.array(Image.open(path), copy=False)
            contacts = np.zeros(im.shape, dtype=np.float32)
            # set the positive, negative, and masked residue pairs
            contacts[im == 1] = -1  # mask neighbour pairs
            contacts[im == 255] = 1
            # mask the matrix below the kth diagonal
            mask = np.tril_indices(contacts.shape[0], k=k)
            contacts[mask] = -1

            # filter out empty contact matrices
            if np.any(contacts > -1):
                contact_maps.append(torch.from_numpy(contacts))
                names_filtered.append(name)
                sequences_filtered.append(seq)

        return names_filtered, sequences_filtered, contact_maps

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):

        x = self.x[i]
        y = self.y[i]

        if self.fragment and len(x) > self.min_length:
            l = np.random.randint(self.min_length, self.max_length + 1)
            if len(x) > l:
                i = np.random.randint(len(x) - l + 1)
                xl = x[i:i + l]
                yl = y[i:i + l, i:i + l]
            else:
                xl = x
                yl = y
            # make sure there are unmasked observations
            # crop another section from original if the current one does not have unmasked observations
            while torch.sum(yl >= 0) == 0:
                l = np.random.randint(self.min_length, self.max_length + 1)
                if len(x) > l:
                    i = np.random.randint(len(x) - l + 1)
                    xl = x[i:i + l]
                    yl = y[i:i + l, i:i + l]
            y = yl.contiguous()
            x = xl

        x = x.long()
        if self.augment is not None:
            x = self.augment(x)

        return x, y


# --------------------------------------------------SURFACES-----------------------------------------------------------
class ScopSurfaceDataset(Dataset):

    def __init__(self, scop_seq_path, scop_surface_path):

        self.surface_paths = glob.glob(f'{scop_surface_path}{os.sep}*{os.sep}*.ply')
        self.surface_indexes = {os.path.basename(path).split('.ply')[0]: path for path in self.surface_paths}

        # keep only the seqs that have surfaces generated for them
        with open(scop_seq_path, 'rb') as f:
            names, sequences = fasta.parse(f)
            self.names, sequences = self.filter_seq_without_surf(names, sequences)

        self.sequences = sequences
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        seq = self.sequences[index]

        pid = self.names[index].decode('utf-8').split()[0]
        pid = 'd' + pid[1:] if pid not in self.surface_indexes else pid

        path = self.surface_indexes[pid]
        surface_mesh = pymesh.load_mesh(path)
        # getting the first model from the file
        struct_path = path.split('.ply')[0] + '.ent'
        struct = self.parser.get_structure(struct_path, struct_path)[0]

        return pid, seq, struct, surface_mesh

    def filter_seq_without_surf(self, names, sequences):
        filtered_names, filtered_seqs = [], []
        for (name, seq) in zip(names, sequences):
            pid = name.decode('utf-8').split()[0]
            # try changing first letter to 'd', required for some SCOPe identifiers
            pid = 'd' + pid[1:] if pid not in self.surface_indexes else pid

            if pid in self.surface_indexes:
                filtered_names.append(name)
                filtered_seqs.append(seq)
            else:
                print("Surface not found for: ", pid)
        return filtered_names, filtered_seqs


class ProcessedSurfaceDataset(Dataset):

    def __init__(self, scopSurfaceDataset, alphabet=Uniprot21(), normalise=True,
                 max_length=None, processed_path=None, atom_shape=True, average_over=3,
                 augment=None):
        self.scopSurfaceDataset = scopSurfaceDataset
        self.alphabet = alphabet
        self.normalise = normalise
        self.max_length = max_length
        self.atom_shape = atom_shape
        self.average_over = average_over
        self.augment = augment

        # load the processed data if it exists from pkl file path
        self.processed_list = pickle.load(open(processed_path, 'rb')) if processed_path is not None else None

    def __len__(self):
        return len(self.scopSurfaceDataset.sequences)

    def __getitem__(self, index):

        if self.processed_list is not None:
            pid, encoded_seq, aa_distances, aa_shapes, aa_mask, rsasa, seq_extracted_from_struct = self.processed_list[
                index].values()
        else:

            pid, orig_seq, struct, surface_mesh = self.scopSurfaceDataset[index]

            encoded_seq = list(self.alphabet.encode(orig_seq))
            residues_depth = surface_util.residues_depth(struct, surface_mesh.vertices)
            aa_distances = [v[0] for v in residues_depth.values()]

            aa_shapes = list(surface_util.get_aa_closest_shape_index(struct, surface_mesh,
                                                                     atom_avg=self.atom_shape,
                                                                     average_over=self.average_over).values())
            seq_extracted_from_struct = surface_util.get_seq_from_struct(struct)
            rsasa = list(surface_util.residue_solvent_accessible_surface_area(struct).values())

            # encode seq with uniprot alphabet and decode back
            # this is because the 3 letter AA code from the structure is mapped to the uniprot alphabet
            seq_uniprot21 = Uniprot21().encode(orig_seq)
            aa_mask = surface_util.get_alignment_mask(Uniprot21().decode(seq_uniprot21).decode("utf-8"),
                                                      ''.join(seq_extracted_from_struct))

        if self.normalise:
            aa_distances = [aa_dist / surface_util.DIST_TO_SURF_NORM[aa_name] for aa_name, aa_dist in
                            zip(seq_extracted_from_struct, aa_distances)]

            rsasa = [sasa / surface_util.SASA_NORM[aa_name] for aa_name, sasa in
                     zip(seq_extracted_from_struct, rsasa)]

        if self.augment is not None:
            encoded_seq = self.augment(torch.LongTensor(encoded_seq)).to(torch.int32)

        # crop to random max_length
        if self.max_length is not None and len(encoded_seq) > self.max_length:
            low = np.random.randint(0, len(encoded_seq) - self.max_length)
            high = low + self.max_length
            encoded_seq = encoded_seq[low:high]

            # where the mask is 1, we put the distance values, then we crop
            aa_distances_expanded = expand_struct_data_to_full_seq(aa_mask, aa_distances)
            aa_distances_expanded = aa_distances_expanded[low:high]
            # remove nans to go back to the structure values
            aa_distances = aa_distances_expanded[~np.isnan(aa_distances_expanded)]

            aa_shapes_expanded = expand_struct_data_to_full_seq(aa_mask, aa_shapes)
            aa_shapes_expanded = aa_shapes_expanded[low:high]
            aa_shapes = aa_shapes_expanded[~np.isnan(aa_shapes_expanded)]

            aa_rsasa_expanded = expand_struct_data_to_full_seq(aa_mask, rsasa)
            aa_rsasa_expanded = aa_rsasa_expanded[low:high]
            rsasa = aa_rsasa_expanded[~np.isnan(aa_rsasa_expanded)]

            aa_mask = aa_mask[low:high]

        return pid, encoded_seq, aa_distances, aa_shapes, aa_mask, rsasa, seq_extracted_from_struct


# --------------------------------------------------Interface DATASET-------------------------------------------------------------
class InterfaceDataset(Dataset):

    def __init__(self, prot_list_path, base, masif_interfaces=False, alphabet=Uniprot21(), output=sys.stdout,
                 max_length=500, atom_interface=False, closest_vertices=3, processed_path=None, normalise=True,
                 augment_prob=None, sppider_interface=True, augment_type=None, softmax_tmp=1.0):

        """
       the data we work with is the masif site data not the interaction data
       we want to predit the interface not if two proteins interact
       most of the  examples in the interface dataset contain a single chain (some have 2 or 3 or 4 chains, so we
       glue them together). To get information about the SASA change upon complex formation we need to look at the
    complex SASA and we get this from the PPI dataset
       """

        self.prot_list = [line.rstrip('\n') for line in open(prot_list_path)]
        self.pdb_fasta_dict = SeqIO.to_dict(SeqIO.parse(f'{base}/masif-ppis/fasta/pdb_seqres.txt', "fasta"))
        self.base = base
        self.mesh_base = f'{self.base}/masif-ppis/01-benchmark_surfaces/' \
            if masif_interfaces else f'{self.base}/masif-ppis/ply/'
        self.masif_ppis = masif_interfaces
        self.alphabet = alphabet
        self.output = output
        self.max_length = max_length
        self.atom_interface = atom_interface
        self.closest_vertices = closest_vertices
        self.normalise = normalise
        self.augment_prob = augment_prob
        self.augment_type = augment_type
        self.complex_chains = self.get_all_chains_in_complex()
        self.sppider_interface = sppider_interface
        self.softmax_tmp = softmax_tmp

        self.filter_dataset()

        if processed_path is not None:
            # using preprocessed dataset
            print('-----Using preprocessed interface dataset--------', file=self.output)
            self.processed_list = pickle.load(open(processed_path, 'rb'))
        else:
            self.processed_list = None

        self.hmm_dict = setup_hmm_dict(f'{base}/')
        if self.augment_type == 'pfam-hmm':
            with open(f"{base}/masif-ppis/pfam_hmm_hits_output.pkl", "rb") as f:
                self.noise = pickle.load(f)
        elif self.augment_type == 'uniref90':
            self.noise = LMDBDataset('/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/'
                                     'uniref90.fasta.lmdb').noise

    def get_all_chains_in_complex(self):
        # --------------------------------------- load the ppi data ---------------------------------------
        with open(f'{self.base}/masif-ppis/lists/all_ppi.txt', 'r') as f:
            ppi_list = f.readlines()

        ppi_list = [l.strip('\n').lower() for l in ppi_list]
        # dict of PDBid -> [chainsA, chainsB, ...]
        complex_chains = defaultdict(list)
        for ppi_id in ppi_list:
            pdb_id = ppi_id.split('_')[0]
            complex_chains[pdb_id] = sorted(list(
                set([chain.upper() for chains in ppi_id.split("_")[1:] for chain in chains] + complex_chains[pdb_id])))
        return complex_chains

    def filter_dataset(self):

        print('-----Filtering interface dataset--------', file=self.output)
        ids_to_remove = []

        for prot_id in self.prot_list:
            complex_id, int_part = prot_id.split("_")

            chain_not_found = any(
                f"{iu.PDB_OBSOLETE_REMAP.get(complex_id.lower(), complex_id.lower())}_{chain_id}" not in self.pdb_fasta_dict
                for chain_id in int_part
            )
            if chain_not_found:
                print("Chain not found for:", prot_id, file=self.output)
                ids_to_remove.append(prot_id)
                continue

            complex_id = complex_id.lower() if not self.masif_ppis else complex_id
            mesh_path = f"{self.mesh_base}{'pdb' if not self.masif_ppis else ''}{complex_id}_{int_part}.ply"

            if not os.path.exists(mesh_path):
                print(f"Mesh not found for: {prot_id}", file=self.output)
                ids_to_remove.append(prot_id)
                continue

            surface_mesh = pymesh.load_mesh(mesh_path)

            try:
                iface = surface_mesh.get_attribute('vertex_iface')
            except RuntimeError as e:
                iface = None

            if iface is None or sum(iface) >= 0.75 * len(iface) or sum(iface) <= 30:
                message = "Interface not found" if iface is None else \
                    f"Iface is too {'large' if sum(iface) >= 0.75 * len(iface) else 'small'}"
                print(f"{message} for: {prot_id}", file=self.output)
                ids_to_remove.append(prot_id)

        self.prot_list = [prot_id for prot_id in self.prot_list if prot_id not in ids_to_remove]

    def __len__(self):
        # count the number of protein chains
        return len(self.prot_list) if self.processed_list is None else len(self.processed_list)

    def __getitem__(self, idx):

        if self.processed_list is not None:
            glued_encoded_seq, aa_mask, iface, dist_to_iface, delta_sasa, seq_extracted_from_struct \
                = self.processed_list[idx].values()
        else:
            pdb_id, int_part = self.prot_list[idx].split("_")

            # ---------------------------- getting the protein sequence------------------------------------
            remapped = iu.PDB_OBSOLETE_REMAP.get(pdb_id.lower(), pdb_id.lower())

            # glue the sequences of the interacting part. E.g. 1A2K_AB -> will have the sequence of chain A and B
            # we need to sort the chains to make sure that the order is always the same
            sequences = [str(self.pdb_fasta_dict[f'{remapped}_{chain_id}'].seq) for chain_id in sorted(list(int_part))]
            glued_seq = ''.join(sequences)
            glued_encoded_seq = self.alphabet.encode(str.encode(glued_seq)).tolist()

            # ------------------ getting the surface of the int part-------------------------------------------
            pdb_id = pdb_id.lower() if not self.masif_ppis else pdb_id
            mesh_path = f"{self.mesh_base}{'pdb' if not self.masif_ppis else ''}{pdb_id}_{int_part}.ply"
            surface_mesh = pymesh.load_mesh(mesh_path)

            # ------------------------getting the structure of the int part------------------------------------
            struct_path = f'{self.base}masif-ppis/pdb/pdb{pdb_id}.ent'
            ppi_part_model = PDBParser(QUIET=True).get_structure(struct_path, struct_path)[0]
            # keep only the chains we are interested in
            all_chains = Selection.unfold_entities(ppi_part_model, "C")
            chains_to_remove = [ch.id for ch in all_chains if ch.id not in list(int_part)]
            remove_chains_from_model(model=ppi_part_model, chains_to_remove=chains_to_remove)

            # ------------------------getting the interface of the int part------------------------------------
            # encode to uniprot21 and decode back to map the groupings of unknown amino acids as the seq from
            # the structure
            seq_uniprot21 = Uniprot21().encode(str.encode(glued_seq)).tolist()
            seq_extracted_from_struct = surface_util.get_seq_from_struct(ppi_part_model, sort_chains=True)
            aa_mask = surface_util.get_alignment_mask(Uniprot21().decode(seq_uniprot21).decode("utf-8"),
                                                      ''.join(seq_extracted_from_struct))

            # ------------------------getting the complex structure--------------------------------------------
            complex_model = PDBParser(QUIET=True).get_structure(struct_path, struct_path)[0]
            # keep all chains that appear in the complex
            chains_to_remove = [ch.id for ch in all_chains if ch.id not in self.complex_chains[pdb_id]]
            remove_chains_from_model(model=complex_model, chains_to_remove=chains_to_remove)

            if not self.sppider_interface:
                iface = list(
                    surface_util.get_aa_interface_based_on_mesh(model=ppi_part_model, surface_mesh=surface_mesh,
                                                                atom_avg=self.atom_interface,
                                                                closest_vertices=self.closest_vertices,
                                                                sort_chains=True).values())
            else:
                iface = surface_util.get_aa_interface_label_based_on_delta_sasa(complex_model=complex_model,
                                                                                ppi_part_model=ppi_part_model,
                                                                                int_part=int_part)

            dist_to_iface = list(surface_util.get_aa_interface_based_on_mesh(model=ppi_part_model,
                                                                             surface_mesh=surface_mesh,
                                                                             atom_avg=self.atom_interface,
                                                                             closest_vertices=self.closest_vertices,
                                                                             distance_to_interface=True,
                                                                             sort_chains=True).values())

            delta_sasa = surface_util.get_delta_sasa(complex_model=complex_model, ppi_part_model=ppi_part_model,
                                                      int_part=int_part)[0]


        if self.normalise:
            dist_to_iface = [aa_dist / surface_util.DIST_TO_IFACE_NORM[aa_name] for aa_name, aa_dist in
                             zip(seq_extracted_from_struct, dist_to_iface)]
            delta_sasa = [sasa / surface_util.DElTA_SASA_NORM[aa_name] for aa_name, sasa in
                            zip(seq_extracted_from_struct, delta_sasa)]

        if self.augment_prob is not None:
            # augment using pfam hmm noise
            # we need to do it for each chain if the interface is based on multiple chains

            pdb_id, int_part = self.prot_list[idx].split("_")
            remapped = iu.PDB_OBSOLETE_REMAP.get(pdb_id.lower(), pdb_id.lower())
            sequences = [(str(self.pdb_fasta_dict[f'{remapped}_{chain_id}'].seq), chain_id)
                         for chain_id in sorted(list(int_part))]

            encoded_seqs = []
            for seq, chain_id in sequences:
                encoded_seq = torch.from_numpy(self.alphabet.encode(str.encode(seq)))
                encoded_seq = seq_augment(self.noise, encoded_seq, self.augment_prob, self.augment_type,
                                          f'{pdb_id.lower()}_{chain_id}', self.hmm_dict,
                                          self.alphabet, self.softmax_tmp)
                encoded_seqs.append(encoded_seq)

            glued_encoded_seq = np.array([n.item() for seq in encoded_seqs for n in seq])

        if self.max_length is not None and len(glued_encoded_seq) > self.max_length:
            low = np.random.randint(0, len(glued_encoded_seq) - self.max_length)
            high = low + self.max_length
            glued_encoded_seq = glued_encoded_seq[low:high]

            # where the mask is 1, we put the distance values, then we crop
            iface_expanded = expand_struct_data_to_full_seq(aa_mask, iface)
            dist_to_iface_expanded = expand_struct_data_to_full_seq(aa_mask, dist_to_iface)
            delta_sasa_expanded = expand_struct_data_to_full_seq(aa_mask, delta_sasa)

            iface_expanded = iface_expanded[low:high]
            dist_to_iface_expanded = dist_to_iface_expanded[low:high]
            delta_sasa_expanded = delta_sasa_expanded[low:high]

            # remove nans to go back to the structure values
            iface = iface_expanded[~np.isnan(iface_expanded)]
            dist_to_iface = dist_to_iface_expanded[~np.isnan(dist_to_iface_expanded)]
            delta_sasa = delta_sasa_expanded[~np.isnan(delta_sasa_expanded)]

            aa_mask = aa_mask[low:high]

        return glued_encoded_seq, aa_mask, iface, dist_to_iface, delta_sasa, seq_extracted_from_struct


def uniref90_agument(noise, length):
    return torch.multinomial(noise, length, replacement=True)


# --------------------------------------------------AUGMENTATION----------------------------------------

def get_dont_augment_mask(seq_len, dont_augment_prob):
    """
    We create a mask that will be used to select a region of the sequence that we don't want to augment.
    This is based on the dont_augment_prob. e.g if dont_augment_prob=0.2 we will select a region of the sequence
    that is 20% of the sequence length that will not be agumented.
    """

    # we select a continuos region of the sequencce that we don't augment
    dont_augment_mask = np.zeros(seq_len)
    dont_augment_length = int(dont_augment_prob * seq_len)
    if dont_augment_length > 0:
        low = np.random.randint(0, seq_len - dont_augment_length)
        high = low + dont_augment_length
        dont_augment_mask[low:high] = 1
    return dont_augment_mask


def pfam_hmm_augment(noise, prot_id, hmm_dict, mask, softmax_tmp, hit_function=get_seq_hmm_probabilities_for_hsp):

    """
    We sample residues to be used for augmentation using information from the pfam hmm hit. Positions that are
    outside the hit regions can't be augmented.

    Args:
        noise (dict): dictionary with the pfam HMMSCAN output for each protein as a qresult object
        prot_id (str): the protein id
        hmm_dict (dict): dictionary with the pfam hmm models for each pfam id
        mask (torch.tensor): the mask that will be used to select the positions to augment
        softmax_tmp (float): the temperature for the softmax (in case we want to play around with the probabilities)
        hit_function (function): the function that will be used to calculate the probabilities for each position
        It can be either get_seq_hmm_probabilities_for_hsp or get_combined_seq_hmm_probabilities (for a single hit or
        from all)

    Returns:
        torch.tensor: the sampled residues for each position in the sequence
        torch.tensor: the mask that was used to select the positions to augment
    """

    seq_probabilities, hit_mask = hit_function(qresult=noise[prot_id], hmm_dict=hmm_dict)

    # replace 0s with 1s outside hit regions to allow multinomial sampling
    seq_probabilities[hit_mask == 0] = 1
    seq_probabilities = torch.from_numpy(seq_probabilities)

    # softmax applied with temperature 1 still changes the distribution
    # so we only apply it if the temperature is different than 1
    if softmax_tmp != 1:
        # apply softmax with temperature
        seq_probabilities = seq_probabilities / softmax_tmp
        seq_probabilities = torch.softmax(seq_probabilities, dim=1)

    sampled_aas = torch.multinomial(seq_probabilities.float(), 1, replacement=True).squeeze()
    # map the AA names to the uniprot21 alphabet
    sampled_aas = Uniprot21().encode(
        str.encode(''.join([PFAM_INDEX_TO_AA[n.item()] for n in sampled_aas])))

    # mask values that were outside the hit regions
    mask[hit_mask == 0] = 0

    return sampled_aas, mask


def seq_augment(noise, seq, augment_prob, noise_type, prot_id, hmm_dict, alphabet, softmax_tmp=1, dont_augment_mask=None,
                hit_function=get_seq_hmm_probabilities_for_hsp):
    # create the random mask... i.e. which positions to infer
    mask = torch.rand(len(seq), device=seq.device)
    mask = (mask < augment_prob).long()

    if dont_augment_mask is not None:
        # where dont_augment_mask is 1 we make the mask 0
        mask = mask * (1 - dont_augment_mask)

    if noise_type == 'pfam-hmm':
        if noise[prot_id] is None:
            return seq
        sampled_aas, mask = pfam_hmm_augment(noise, prot_id, hmm_dict, mask, softmax_tmp, hit_function)

    elif noise_type == 'uniref90':
        sampled_aas = uniref90_agument(noise, len(seq))

    # replace the selected positions with the sampled noise (21 AA)
    # but translate the noise into the dataset's alphabet (21 or less AA groupings)
    # decoded with uniprot because the noise was calculated with uniprot alphabet
    sampled_aas = Uniprot21().decode(sampled_aas)
    sampled_aas = alphabet.encode(sampled_aas)
    seq = (1 - mask) * seq + mask * sampled_aas  # unmasked AA + new values in masked positions

    return seq


# -------------------------------------------FLIP DATASET--------------------------------------------

class FlipDataset(Dataset):
    def __init__(self, base_pth='../../data/FLIP', dataset='aav', split='des_mut', type='train',
                 alphabet=Uniprot21(), augment_prob=0, noise_type='pfam-hmm', full_train=False, softmax_tmp=1.0,
                 hit_function=get_seq_hmm_probabilities_for_hsp):

        df = pd.read_csv(f'{base_pth}/{dataset}/splits/{split}.csv')
        # remove '*' which represents deletions in the wild type
        df.sequence = df['sequence'].apply(lambda s: re.sub(r'[^A-Z]', '', s.upper()))
        if type == 'train':
            if full_train:
                df = df[df.set == 'train']
            else:
                df = df[(df.set == 'train') & (df.validation != True)]
        elif type == 'val':
            df = df[(df.set == 'train') & (df.validation == True)]
        elif type == 'test':
            df = df[df.set == 'test']

        df.reset_index(drop=True, inplace=True)

        self.dataset = df
        self.alphabet = alphabet
        self.noise_type = noise_type
        self.augment_prob = augment_prob
        self.dataset_name = dataset
        self.softmax_tmp = softmax_tmp
        self.hit_function = hit_function

        if noise_type == 'pfam-hmm':
            self.hmm_dict = setup_hmm_dict(os.path.dirname(base_pth) + '/')
            with open(f"{base_pth}/pfam_hmm_hits_output.pkl", "rb") as f:
                self.noise = pickle.load(f)
        elif self.noise_type == 'uniref90':
            self.noise = LMDBDataset('/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/'
                                     'uniref90.fasta.lmdb').noise
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        seq = self.dataset.sequence[idx]
        encoded_seq = torch.from_numpy(self.alphabet.encode(str.encode(seq)))
        target = self.dataset.target[idx]

        if self.augment_prob > 0:

            dont_augment_mask = np.zeros(len(seq))
            if self.dataset_name == "aav":
                prot_id = f"aav-length={len(seq)}"
                # when we have insertions we want to maintain the original sequence
                # there are 28 sequences (because thats the length of the mutated crop)
                # that have 1 deletions, leading to a number of 28 sequences with length 734
                # for these we need to do -1 from the upper range
                if len(seq) == 734:
                    dont_augment_mask[560:588 + (len(seq) - 735)-1] = 1
                else:
                    dont_augment_mask[560:588 + (len(seq) - 735)] = 1

            elif self.dataset_name == "gb1":
                prot_id = 'gb1'
                dont_augment_mask[1:56] = 1
            else:
                # here is about temperature so we can augment everything
                prot_id = f"meltome-seq={seq}"

            encoded_seq = seq_augment(self.noise, encoded_seq, self.augment_prob, self.noise_type, prot_id,
                              self.hmm_dict, self.alphabet, softmax_tmp=self.softmax_tmp,
                                      dont_augment_mask=dont_augment_mask, hit_function=self.hit_function)
        return encoded_seq, target


# --------------------------------------------------ENZYME-------------------------------------------------------------
class EnzymeDataset(Dataset):

    def __init__(self, enzyme_path, split="train", alphabet=Uniprot21(), augment_prob=0, noise_type='pfam-hmm',
                 filter_dataset=False, softmax_tmp=1, dont_augment_prob=0, hit_function=get_seq_hmm_probabilities_for_hsp):
        self.alphabet = alphabet
        self.metadata_path = enzyme_path
        self.augment_prob = augment_prob
        self.uniprot = Uniprot21()
        self.noise_type = noise_type
        self.pdb_fasta = SeqIO.to_dict(SeqIO.parse(f'{enzyme_path}/metadata/pdb_seqres.txt', "fasta"))
        self.name = split
        self.softmax_tmp = softmax_tmp
        self.dont_augment_prob = dont_augment_prob
        self.hit_function = hit_function

        with open(f'{enzyme_path}/metadata/base_split.json') as json_file:
            json_splits = json.load(json_file)
            if split == "all":
                self.prot_ids = json_splits["train"] + json_splits["valid"] + json_splits["test"]
            else:
                self.prot_ids = json_splits[split]

        if filter_dataset:
            # remove chains that are in both train and test, or  test and valid
            self.prot_ids = [p for p in self.prot_ids if p not in MIXED_SPLIT_CHAINS]

        with open(f"{enzyme_path}/metadata/function_labels.json", "r") as f:
            self.labels_all = json.load(f)

        with open(f"{enzyme_path}/metadata/labels_to_idx.json", "r") as f:
            self.labels_to_idx = json.load(f)

        # used just with the hmm noise type
        self.hmm_dict = None
        if self.noise_type == 'pfam-hmm':
            self.hmm_dict = setup_hmm_dict(os.path.dirname(enzyme_path) + '/')
            with open(f"{enzyme_path}/metadata/pfam_hmm_hits_output.pkl", "rb") as f:
                self.noise = pickle.load(f)
        elif self.noise_type == 'uniref90':
            self.noise = LMDBDataset('/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/'
                                     'uniref90.fasta.lmdb').noise

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, index):
        orig_prot_id = self.prot_ids[index]
        prot_id = orig_prot_id if self.prot_ids[index] not in PDB_OBSOLETE_REMAP else PDB_OBSOLETE_REMAP[orig_prot_id]

        prot_record = self.pdb_fasta[prot_id]
        seq = torch.from_numpy(self.alphabet.encode(bytes(prot_record.seq)))

        if self.noise_type is not None:
            dont_augment_mask = get_dont_augment_mask(len(seq), self.dont_augment_prob)
            seq = seq_augment(self.noise, seq, self.augment_prob, self.noise_type, orig_prot_id, self.hmm_dict,
                               self.alphabet, self.softmax_tmp, dont_augment_mask=dont_augment_mask,
                               hit_function=self.hit_function)
        return seq, self.labels_to_idx[self.labels_all[orig_prot_id]], orig_prot_id, prot_record.seq


# ------------------------------------------------FOLD HOMOLOGY DATASET------------------------------------------------
class FoldHomologyDataset(Dataset):
    def __init__(self, dataset_pth='../../data/HomologyTAPE', data_split='train', alphabet=Uniprot21(),
                 augment_prob=0, noise_type='pfam-hmm',  softmax_tmp=1, dont_augment_prob=0,
                 hit_function=get_seq_hmm_probabilities_for_hsp):
        self.labels_to_idx = pd.read_csv(f'{dataset_pth}/class_map.txt', delimiter='\t', header=None,
                                         names=['name', 'id']).set_index('name')['id'].to_dict()
        df = pd.read_csv(f'{dataset_pth}/{data_split}.txt', delimiter='\t', header=None, names=['scopid', 'name'])
        self.prot_ids = df['scopid'].values
        self.labels = df['name'].values
        self.alphabet = alphabet
        self.scop_fasta = SeqIO.to_dict(SeqIO.parse(f'{dataset_pth}/astral-scopdom-seqres-gd-sel-gs-bib-95-1.75.fa',
                                                    "fasta"))
        self.name = data_split.split('_')[1].upper() if '_' in data_split else data_split.upper()
        self.augment_prob = augment_prob
        self.noise_type = noise_type
        self.softmax_tmp = softmax_tmp
        self.dont_augment_prob = dont_augment_prob
        self.hit_function = hit_function

        # used just with the hmm noise type
        self.hmm_dict = None
        if self.noise_type == 'pfam-hmm':
            self.hmm_dict = setup_hmm_dict(f'{os.path.dirname(dataset_pth)}/')
            with open(f"{dataset_pth}/pfam_hmm_hits_output.pkl", "rb") as f:
                self.noise = pickle.load(f)
        elif self.noise_type == 'uniref90':
            self.noise = LMDBDataset('/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/'
                                     'uniref90.fasta.lmdb').noise

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, index):
        id = self.prot_ids[index] if self.prot_ids[index] in self.scop_fasta else 'g' + self.prot_ids[index][1:]
        prot_record = self.scop_fasta[id]
        seq = torch.from_numpy(self.alphabet.encode(bytes(prot_record.seq.upper())))
        if self.noise_type is not None:
            dont_augment_mask = get_dont_augment_mask(len(seq), self.dont_augment_prob)
            seq = seq_augment(self.noise, seq, self.augment_prob, self.noise_type, self.prot_ids[index],
                              self.hmm_dict, self.alphabet, self.softmax_tmp, dont_augment_mask=dont_augment_mask,
                              hit_function=self.hit_function)
        return seq, self.labels_to_idx[self.labels[index]], self.prot_ids[index], prot_record.seq


# -----------------------------------------------DATASETS FOR PROTEIN SEQUENCES---------------------------------------
class LMDBDataset:
    """
    Adapted from: https://github.com/songlab-cal/tape
    Creates a dataset from an lmdb file.
    Args:
        data_file: Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self, data_file, max_length=500, alphabet=Uniprot21(), in_memory=False, delim="|DELIM|",
                 output=sys.stdout):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get("nr_seq".encode()).decode())
            self.noise = torch.from_numpy(pickle.loads(txn.get("marginal_distribution".encode())))
            self.lengths = pickle.loads(txn.get("seq_lengths".encode()))

            print("Number of sequences in the Unirep90 .lmdb dataset: ", self._num_examples,
                  file=output)

        if in_memory:
            cache = [None] * self._num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self.max_length = max_length
        self.alphabet = alphabet
        self.delim = delim

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = txn.get(str(index).encode()).decode()
                if self._in_memory:
                    self._cache[index] = item

        description, seq = item.split(self.delim)
        seq = torch.from_numpy(self.alphabet.encode(seq.encode()))
        if 0 < self.max_length < len(seq):
            # randomly sample a subsequence of length max_length
            j = random.randint(0, len(seq) - self.max_length)
            seq = seq[j:j + self.max_length]

        return seq.long()


class ClozeDataset:
    """
    Wrapper for LMDBDataset
    Args:
        dataset (LMDBDataset): dataset to wrap
        probability (float): number in [0,1] that represents the chance of masking e.g. 0.1 means 10%
        noise (torch.FloatTensor): the background marginal distribution of the dataset, len(noise) == nr_AA
    """

    def __init__(self, dataset, probability, noise):
        self.dataset = dataset
        self.probability = probability
        self.noise = noise

        self.uniprot = Uniprot21()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """"
        Returns:
            list: protein's sequence as a list of integers highlighting the AA number
            list: protein's labels as a list of integers: original values for masked AAs and unknown 20 for the rest """

        item = self.dataset[i]
        n = len(self.noise)  # number of tokens

        # create the random mask... i.e. which positions to infer
        mask = torch.rand(len(item), device=item.device)
        mask = (mask < self.probability).long()  # we mask with probability p

        # keep original AA label for masked positions and assign unknown AA label to all the others
        labels = mask * item + (1 - mask) * (n - 1)

        # sample the masked positions from the noise distribution
        noise = torch.multinomial(self.noise, len(item), replacement=True)

        # replace the masked positions with the sampled noise (21 AA)
        # but translate the noise into the dataset's alphabet (21 or less AA groupings)
        # decoded with uniprot because the noise was calculated with uniprot alphabet
        noise = self.uniprot.decode(noise)
        noise = self.dataset.alphabet.encode(noise)

        x = (1 - mask) * item + mask * noise  # unmasked AA + new values in masked positions

        # eg: x = [4,5,13,19], labels = [4,20,20,13]
        # (only the first and last AA have been replaced with noise AA)
        return x, labels
