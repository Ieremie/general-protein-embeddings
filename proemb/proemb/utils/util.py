from typing import Tuple

import numpy
import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence


def pack_sequences(X, order=None):
    n = len(X)
    lengths = np.array([len(x) for x in X])
    # order the sequences descending by length
    if order is None:
        order = np.argsort(lengths)[::-1]
        order = np.ascontiguousarray(order)
    m = max(len(x) for x in X)

    X_block = X[0].new(n, m).zero_()

    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i, :len(x)] = x

    lengths = lengths[order]
    X = pack_padded_sequence(X_block, lengths, batch_first=True)

    return X, order


def unpack_sequences(X, order):
    X, lengths = pad_packed_sequence(X, batch_first=True)
    X_block = [None] * len(order)
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i, :lengths[i]]
    return X_block


def get_protein_pairs_weights(y_train_levels, tau):
    # sum along SCOP hierarcy e.g. [1.1.1.1] - > 4
    similarity = y_train_levels.long().numpy().sum(2)
    levels, counts = np.unique(similarity, return_counts=True)
    order = np.argsort(levels)
    counts = counts[order]

    weights = counts ** tau / counts
    weights = weights[similarity].ravel()
    # weights shape = NR_seq * Nr_seq

    return weights


class MultinomialResample:
    def __init__(self, device, p):
        trans = torch.ones(21, 21)
        trans = trans / trans.sum(1, keepdim=True)
        trans = trans.to(device)

        # making the prob of choosing the same one bigger (1-p) and smaller for replacing it
        self.p = (1 - p) * torch.eye(trans.size(0)).to(trans.device) + p * trans

    def __call__(self, x):
        # print(x.size(), x.dtype)
        p = self.p[x]  # get distribution for each x
        return torch.multinomial(p, 1).view(-1)  # sample from distribution (pick one AA as a replacement)


class LargeWeightedRandomSampler(torch.utils.data.sampler.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)), size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def collate_protein_seq(args):
    """
    Sorting seq and labels in a batch in decreasing order
    Args:
       seqs ((torch.FloatTensor, ...) tuple): protein sequences in a batch
       labels ((torch.FloatTensor, ...) tuple): labels of the original masked AAs in a batch
       (same length as the protein seqs)
    Returns:
        torch.FloatTensor: containing the padded sequence, shape: (Batch * MaxL)
        torch.FloatTensor: containing the lengts of the sequences in the batch, shape: (Batch, )
        PackedSequence: contains the labels of the sequences in the batch (original values of masked AAs)
    """

    seqs, labels = zip(*args)
    # from torch tensor to normal list
    seqs = list(seqs)
    labels = list(labels)

    # sort them by decreasing length to work with pytorch ONNX
    seqs = sorted(seqs, key=lambda x: x.size()[0], reverse=True)
    labels = sorted(labels, key=lambda x: x.size()[0], reverse=True)

    seqs_lengths = torch.FloatTensor([len(seq) for seq in seqs])
    # using pad_sequence instead of directly using ge due to multi GPU training
    # padding value same with unknown amino acid A=20, this does not affect training as we are packing the seqs later
    seqs = pad_sequence(seqs, batch_first=True, padding_value=20)
    labels = pack_sequence(labels, enforce_sorted=False)

    return seqs, seqs_lengths, labels


def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)


def collate_lists(args):
    x = [a[0] for a in args]
    y = [a[1] for a in args]
    return x, y


def collate_seq_classification(data: Tuple):
    seqs, labels, names, fasta = zip(*data)
    seqs_lengths = torch.FloatTensor([len(seq) for seq in seqs])
    seqs = pad_sequence(seqs, batch_first=True, padding_value=20)

    return seqs.to(torch.int64), seqs_lengths.to(torch.int64), torch.LongTensor(labels)


def collate_seq_regression(data: Tuple):
    seqs, targets = zip(*data)
    seqs_lengths = torch.FloatTensor([len(seq) for seq in seqs])
    seqs = pad_sequence(seqs, batch_first=True, padding_value=20)

    return seqs.to(torch.int64), seqs_lengths.to(torch.int64), torch.FloatTensor(targets)


def collate_interface(data: Tuple):
    seqs, aa_masks, ifaces, dist_ifaces, delta_sasas, _ = zip(*data)
    sorted_inds = numpy.argsort([len(seq) for seq in seqs])[::-1]
    seqs_packed = pack_sequence([torch.IntTensor(seqs[i]).to(torch.int64) for i in sorted_inds], enforce_sorted=False)
    aa_masks_packed = pack_sequence([torch.BoolTensor(aa_masks[i]) for i in sorted_inds], enforce_sorted=False)
    ifaces_packed = pack_sequence([torch.BoolTensor(ifaces[i]) for i in sorted_inds], enforce_sorted=False)
    dist_ifaces_packed = pack_sequence([torch.FloatTensor(dist_ifaces[i]) for i in sorted_inds], enforce_sorted=False)
    delta_sasas_packed = pack_sequence([torch.FloatTensor(delta_sasas[i]) for i in sorted_inds], enforce_sorted=False)

    return seqs_packed, aa_masks_packed, ifaces_packed, dist_ifaces_packed, delta_sasas_packed


def collate_surface_info(data: Tuple):
    pids, seqs, seqs_aa_distances, seqs_aa_shapes, seqs_mask, seqs_rsasa, _ = zip(*data)

    # sort descending by seq length
    # we need to sort the distances and shapes as well
    sorted_inds = numpy.argsort([len(seq) for seq in seqs])[::-1]

    # we cant enforce sorted as surf info does not the exact same length with the orig seq
    seqs_packed = pack_sequence([torch.IntTensor(seqs[i]).to(torch.int64) for i in sorted_inds], enforce_sorted=False)
    seqs_masks_packed = pack_sequence([torch.BoolTensor(seqs_mask[i]) for i in sorted_inds], enforce_sorted=False)
    aa_distances_packed = pack_sequence([torch.FloatTensor(seqs_aa_distances[i]) for i in sorted_inds],
                                        enforce_sorted=False)
    aa_shape_indexes_packed = pack_sequence([torch.FloatTensor(seqs_aa_shapes[i]) for i in sorted_inds],
                                            enforce_sorted=False)
    seqs_rsasa_packed = pack_sequence([torch.FloatTensor(seqs_rsasa[i]) for i in sorted_inds], enforce_sorted=False)

    return seqs_packed, seqs_masks_packed, aa_distances_packed, aa_shape_indexes_packed, seqs_rsasa_packed
