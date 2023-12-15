import argparse
import pickle
import pprint
import tqdm
import os
import sys

sys.path.insert(0, '..')

from proemb.datasets import ProcessedSurfaceDataset, ScopSurfaceDataset, InterfaceDataset


def process_scop_surface_dataset(split, base, atom_shape=True, average_over=3, normalise=False):
    base_dataset = ScopSurfaceDataset(f'{base}SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.{split}.fa',
                                      f'{base}SCOPe/pdbstyle-2.06-structures-radius-1.4-no-HETATM')

    surface_train = ProcessedSurfaceDataset(base_dataset, max_length=500000, atom_shape=atom_shape,
                                            average_over=average_over, normalise=normalise)
    processed = []

    for prot in tqdm.tqdm(surface_train):
        processed_dict = {'pid': prot[0], 'seq': prot[1], 'aa_distances': prot[2], 'aa_shapes': prot[3],
                          'aa_mask': prot[4], 'rsasa': prot[5], 'seq_extracted_from_struct': prot[6]}

        processed.append(processed_dict)

    with open(f'astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.{split}-average_over={average_over}-atom-shape={atom_shape}-'
              f'radius-1.4-no-HETATM-surface-processed.pkl', 'wb') as f:
        pickle.dump(processed, f)


def process_interface_dataset(split, base, atom_interface=True, closest_vertices=3, sppider_interface_label=True):
    dataset = InterfaceDataset(f'{base}masif-ppis/lists/{split}.txt', base, masif_interfaces=False,
                               atom_interface=atom_interface, max_length=999999,
                               closest_vertices=closest_vertices, normalise=False,
                               sppider_interface=sppider_interface_label)

    processed = []
    for prot in tqdm.tqdm(dataset, total=len(dataset)):
        processed_dict = {'seq': prot[0], 'aa_mask': prot[1], 'iface': prot[2], 'dist_to_iface': prot[3],
                          'delta_sasa': prot[4], 'seq_extracted_from_struct': prot[5]}

        processed.append(processed_dict)

    with open(f'{split}-closest-vertices={closest_vertices}-atom-interface={atom_interface}-'
              f'sppider_interface_label={sppider_interface_label}-with-delta-sasa-processed.pkl', 'wb') as f:
        pickle.dump(processed, f)


if __name__ == '__main__':

    """
    Processing datasets such data during training we can access the date using a key,value pair
    Doing this during training instead needs loads of workers and memory
    
    --atom-shape: if true, the shape index of a residue is given by the average shape index assigned to each atom (rather than alpha carbon only)
    --average-over: the number of vertexes we average the shape index over to get the shape index of an atom
    --atom-interface: if true, we set the PPI interface based on the vertexes covered during complex formation
    --closest-vertices: the number of closest vertices we consider when setting the PPI interface (atom label)
    --sppider-interface-label: if true, we set the PPI interface based on the SPPIDER framework: at least 
                               4% change and 5A^2  difference in SASA                         
    """

    # base location of the data
    base =  '/scratch/ii1g17/protein-embeddings/data/' if os.environ.get('SLURM_JOB_ID') else \
        '../../data/scratch/protein-embeddings/data/'

    # print all args
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser('Script for generating protein surfaces')
    parser.add_argument('--split', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--atom-shape', action='store_true')
    parser.add_argument('--average-over', type=int, default=3)

    parser.add_argument('--atom-interface', action='store_true')
    parser.add_argument('--sppider-interface-label', action='store_true')
    parser.add_argument('--closest-vertices', type=int, default=3)

    args = parser.parse_args()
    args = vars(args)
    pprint.pprint(args)

    if args['dataset'] == 'interface':
        process_interface_dataset(args['split'], base, args['atom_interface'], args['closest_vertices'],
                                  args['sppider_interface_label'])
    else:
        process_scop_surface_dataset(args['split'], base, args['atom_shape'], args['average_over'])

# run this on iridis as a slurm job
# sbatch -p batch --nodes=1 --job-name=surf-process --ntasks-per-node=1 --cpus-per-task=2 --mem=64G --time=1-00:00:00 --wrap="python process_surface_dataset.py train scop"
