from __future__ import print_function, division

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from catalyst.data import DistributedSamplerWrapper
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score as average_precision, accuracy_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from proemb import alphabets
from proemb.datasets import ClozeDataset, LMDBDataset, SCOPeDataset, SCOPePairsDataset, ContactMapDataset, \
    AllPairsDataset, ScopSurfaceDataset, ProcessedSurfaceDataset, InterfaceDataset
from proemb.models.lstm import SkipLSTM
from proemb.models.multitask import L2, L1, OrdinalRegression, BilinearContactMap, ProSEMT, SurfacePredictor, \
    InterfacePredictor
from proemb.utils.train_utils import multi_gpu_info, itr_restart, \
    epoch_time, multitask_train_info, save_checkpoint
from proemb.utils.util import LargeWeightedRandomSampler, collate_protein_seq, collate_surface_info, collate_interface
from proemb.utils.util import MultinomialResample, get_protein_pairs_weights, \
    collate_paired_sequences, collate_lists, unpack_sequences, pack_sequences

from contextlib import nullcontext
import pprint

torch_generator = torch.Generator().manual_seed(42)


def eval_scop(model, test_iterator, device, use_multi_gpu):
    y = []
    logits = []
    for x0, x1, y_mb in test_iterator:

        b = len(x0)
        x = x0 + x1
        x, order = pack_sequences(x)
        seq_unpacked, lens_unpacked = pad_packed_sequence(x, batch_first=True)
        seq_unpacked = seq_unpacked.to(device)
        y_mb = y_mb.to(device)
        y.append(y_mb.long())

        padded_output = model(seq_unpacked, lens_unpacked)
        z = pack_padded_sequence(padded_output, lens_unpacked.to('cpu'), batch_first=True)
        z = unpack_sequences(z, order)

        z0 = z[:b]
        z1 = z[b:]
        for i in range(b):
            z_a = z0[i]
            z_b = z1[i]
            logits.append(model.module.score(z_a, z_b)) if use_multi_gpu else logits.append(model.score(z_a, z_b))

    y = torch.cat(y, 0)
    logits = torch.stack(logits, 0)
    # p = torch.stack(logits, 0).data

    log_p = F.logsigmoid(logits).data
    log_m_p = F.logsigmoid(-logits).data
    zeros = log_p.new(log_p.size(0), 1).zero_()
    log_p_ge = torch.cat([zeros, log_p], 1)
    log_p_lt = torch.cat([log_m_p, zeros], 1)
    log_p = log_p_ge + log_p_lt

    loss = F.cross_entropy(log_p, y).item()

    # softmax returns nans when input is all infinity
    log_p = torch.nan_to_num(log_p)
    log_p = log_p.to(torch.float16)

    p = F.softmax(log_p, 1)

    _, y_hard = torch.max(log_p, 1)
    levels = torch.arange(5).to(p.device)
    y_hat = torch.sum(p * levels, 1)

    accuracy = torch.mean((y == y_hard).float()).item()
    mse = torch.mean((y.float() - y_hat) ** 2).item()

    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    r, _ = pearsonr(y_hat, y)
    rho, _ = spearmanr(y_hat, y)

    ## calculate average-precision score for each structural level
    aupr = np.zeros(4, dtype=np.float32)
    for i in range(4):
        target = (y > i).astype(np.float32)
        aupr[i] = average_precision(target, y_hat)

    return loss, accuracy, mse, r, rho, aupr


def batch_similarity_grad(model, seqsA, seqsB, labels, device, use_multi_gpu, weight=1.0):
    b = len(seqsA)
    all_seqs = seqsA + seqsB

    all_seqs, order = pack_sequences(all_seqs)
    seq_unpacked, lens_unpacked = pad_packed_sequence(all_seqs, batch_first=True)
    seq_unpacked = seq_unpacked.to(device)

    labels = labels.to(device)

    padded_output = model(seq_unpacked, lens_unpacked)
    z = pack_padded_sequence(padded_output, lens_unpacked.to('cpu'), batch_first=True)

    # for memory efficiency
    # we backprop to the representations from each loss pair§
    # then backprop through the embedding model
    z_detach = z.data.detach()
    z_detach.requires_grad = True
    z_detach = PackedSequence(z_detach, z.batch_sizes)

    z_unpack = unpack_sequences(z_detach, order)

    z0 = z_unpack[:b]
    z1 = z_unpack[b:]

    logits = torch.zeros_like(labels)

    model = model.module if use_multi_gpu else model
    for i in range(b):
        z_a = z0[i]
        z_b = z1[i]
        li = model.score(z_a, z_b)

        # compare [0.12,0.2,0.14,0.2] to [1,1,1,0] (match up to level 3)
        # we divide by batch size to have the same gradient as if the calculation was done as a batch
        loss = F.binary_cross_entropy_with_logits(li, labels[i]) / b
        loss.backward(retain_graph=True)
        logits[i] = li.detach()

    # now backprop from z
    grad = z_detach.data.grad

    # we update the projection layer of the LM ( -> 100 DIM, which is not used for the cloze task)
    # this should be updated regardless of the weight
    z.data.backward(grad, inputs=list(model.skipLSTM.proj.parameters()), retain_graph=True)

    # we scale down the gradient before backpropagating through the LM model
    # if 0, the LM (LSTM) is not updated based on what happened to the similarity head
    inputs = [p for p in model.skipLSTM.layers.parameters() if p.requires_grad]
    if inputs:
        z.data.backward(grad * weight, inputs=inputs)

    # calculate minibatch performance metrics
    with torch.no_grad():
        # y E [1,1,1,0] matches at up to level 3
        y = torch.sum(labels.long(), 1)

        log_p = F.logsigmoid(logits)
        log_m_p = F.logsigmoid(-logits)
        zeros = log_p.new(b, 1).zero_()
        log_p_ge = torch.cat([zeros, log_p], 1)
        log_p_lt = torch.cat([log_m_p, zeros], 1)
        log_p = log_p_ge + log_p_lt

        # y is single label(1 of 5 classes), log_p is 5 logits
        loss = F.cross_entropy(log_p, y).item()  # how well we predicted the similarity, averaged over batch

        # softmax returns nans when input is all infinity
        log_p = torch.nan_to_num(log_p)
        log_p = log_p.to(torch.float16)

        p = F.softmax(log_p, 1)
        _, y_hard = torch.max(log_p, 1)
        levels = torch.arange(5).to(p.device).float()
        y_hat = torch.sum(p * levels, 1)

        accuracy = torch.mean((y == y_hard).float()).item()
        mse = torch.mean((y.float() - y_hat) ** 2).item()

    return loss, accuracy, mse, b


def cmap_grad(model, x, y, device, use_multi_gpu, weight=1.0):
    b = len(x)

    x, order = pack_sequences(x)
    seq_unpacked, lens_unpacked = pad_packed_sequence(x, batch_first=True)
    seq_unpacked = seq_unpacked.to(device)

    padded_output = model(seq_unpacked, lens_unpacked, apply_proj=False)
    z = pack_padded_sequence(padded_output, lens_unpacked.to('cpu'), batch_first=True)

    # backprop each sequence individually for memory efficiency
    z_detach = z.data.detach()
    z_detach.requires_grad = True
    z_detach = PackedSequence(z_detach, z.batch_sizes)

    z_unpack = unpack_sequences(z_detach, order)

    loss = 0  # loss over minibatch
    tp = 0  # true positives over minibatch
    gp = 0  # number of ground truth positives in minibatch
    pp = 0  # number of predicted positives in minibatch
    total = 0  # total number of residue pairs

    model = model.module if use_multi_gpu else model
    for i in range(b):
        zi = z_unpack[i]
        # logits = (seqL, seqL)

        logits = model.predict(zi.unsqueeze(0)).view(-1)

        yi = y[i].contiguous().view(-1)  # flattened target contacts
        yi = yi.to(device)

        mask = (yi < 0)  # unobserved positions
        logits = logits[~mask]
        yi = yi[~mask]

        cross_entropy_loss = F.binary_cross_entropy_with_logits(logits, yi) / b  # loss for this sequence
        cross_entropy_loss.backward(retain_graph=True)

        # the plotted loss should not be scaled by the weight
        loss += cross_entropy_loss.item()
        total += yi.size(0)  # how long the crop was

        # also calculate the recall and precision
        with torch.no_grad():
            p_hat = torch.sigmoid(logits)

            # added by IOAN
            p_hat = p_hat >= 0.5

            tp += torch.sum(p_hat * yi).item()  # true positive
            gp += yi.sum().item()  # total positive
            pp += p_hat.sum().item()  # predicted positive

    # now, accumulate gradients for the embedding model
    grad = z_detach.data.grad

    # we scale down the gradient before backpropagating through the LM model
    # if 0, the LM is not updated based on what happened to the similarity head
    inputs = [p for p in model.skipLSTM.layers.parameters() if p.requires_grad]
    if inputs:
        z.data.backward(grad * weight, inputs=inputs)
    return loss, tp, gp, pp, total


def predict_cmap(model, x, y, device, use_multi_gpu):
    b = len(x)

    x, order = pack_sequences(x)
    seq_unpacked, lens_unpacked = pad_packed_sequence(x, batch_first=True)
    seq_unpacked = seq_unpacked.to(device)

    padded_output = model(seq_unpacked, lens_unpacked, apply_proj=False)
    z = pack_padded_sequence(padded_output, lens_unpacked.to('cpu'), batch_first=True)
    z = unpack_sequences(z, order)

    logits = []
    y_list = []
    model = model.module if use_multi_gpu else model
    for i in range(b):
        zi = z[i]

        lp = model.predict(zi.unsqueeze(0)).view(-1)
        yi = y[i].contiguous().view(-1)

        yi = yi.to(device)
        mask = (yi < 0)

        lp = lp[~mask]
        yi = yi[~mask]

        logits.append(lp)
        y_list.append(yi)

    return logits, y_list


def eval_cmap(model, test_iterator, d, use_multi_gpu):
    logits = []
    y = []

    for x, y_mb in test_iterator:
        logits_this, y_this = predict_cmap(model, x, y_mb, d, use_multi_gpu)
        logits += logits_this
        y += y_this

    y = torch.cat(y, 0)
    logits = torch.cat(logits, 0)

    loss = F.binary_cross_entropy_with_logits(logits, y).item()

    # this is probably wrong, because we are not transforming the logits to 1 or 0 depending on the threshold
    # original paper from ICLR used the 0 threshold without the sigmoid
    p_hat = torch.sigmoid(logits)

    # added by Ioan
    p_hat = p_hat >= 0.5

    tp = torch.sum(y * p_hat).item()
    pr = tp / (torch.sum(p_hat).item() + 1e-8)
    re = tp / torch.sum(y).item()
    f1 = 2 * pr * re / (pr + re)

    y = y.cpu().numpy()

    # replacing inf with numbers
    logits = torch.nan_to_num(logits)
    logits = logits.to(torch.float16)

    logits = logits.data.cpu().numpy()

    aupr = average_precision(y, logits)

    return loss, pr, re, f1, aupr


def cloze_grad(model, seqs, seqs_lengths, labels, device, weight=1.0, backward=True, use_multi_gpu=True):
    seqs = seqs.to(device)
    labels = labels.data.to(device)

    # we only consider AA that have been masked (this also exclude the AA that have the unknwon token '20')
    mask = (labels < 20)
    # check that we have noised positions...
    loss = 0
    correct = 0
    n = mask.float().sum().item()
    if n > 0:
        padded_output = model(seqs, seqs_lengths, apply_proj=False)
        logits = pack_padded_sequence(padded_output, seqs_lengths.to('cpu'), batch_first=True,
                                      enforce_sorted=False)

        z = logits.data
        # using the original projection layer to project into 21 dim
        model = model.module.skipLSTM if use_multi_gpu else model.skipLSTM
        logits = model.cloze(z)
        logits = logits[mask]
        labels = labels[mask]

        loss = F.cross_entropy(logits, labels)
        _, y_hat = torch.max(logits, 1)

        w_loss = loss * weight

        if backward:
            w_loss.backward()

        loss = loss.item()
        correct = torch.sum((labels == y_hat).float()).item()

    return loss, correct, n


def eval_cloze(model, data_it, device, use_multi_gpu=True):
    losses, accuracies = [], []
    for seqs, seqs_lengths, labels in data_it:
        loss, correct, nr_masked = cloze_grad(model, seqs, seqs_lengths, labels, device, weight=1.0, backward=False,
                                              use_multi_gpu=use_multi_gpu)

        if nr_masked > 0:
            losses.append(loss)
            accuracies.append(correct / nr_masked)
    return np.mean(losses), np.mean(accuracies)


def surface_pass(model, device, batch, use_multi_gpu, dist_weight=1.0, shape_weight=1.0, rsasa_weight=1.0,
                 backward=True, apply_sigmoid=True):
    seqs_packed, seqs_masks_packed, surf_distances_packed, surf_shapes_packed, rsasa_packed = batch

    seqs_unpacked, lens_unpacked = pad_packed_sequence(seqs_packed, batch_first=True)
    padded_output = model(seqs_unpacked.to(device), lens_unpacked, apply_proj=False)

    model = model.module if use_multi_gpu else model

    # we break the computational graph here, so we can update weights
    # of surface head without changing anything to the LM model
    logits_detach = padded_output.detach()
    logits_detach.requires_grad = True

    # projecting LSTM dim -> 1 dim, and remove redundant dim
    logits_distance = model.predict_aa_distance(logits_detach).squeeze(-1)
    logits_shape = model.predict_aa_shape_index(logits_detach).squeeze(-1)
    logits_rsasa = model.predict_rsasa(logits_detach).squeeze(-1)

    if apply_sigmoid:
        logits_distance = torch.sigmoid(logits_distance)
        logits_rsasa = torch.sigmoid(logits_rsasa)
    logits_shape = torch.tanh(logits_shape)

    mse_loss = nn.MSELoss(reduction='none')
    mae_loss = nn.L1Loss(reduction='none')
    # add small epsilon to avoid nan
    eps = 1e-8

    masks_unpacked, _ = pad_packed_sequence(seqs_masks_packed, batch_first=True)
    dist_unpacked, dist_lengths = pad_packed_sequence(surf_distances_packed, batch_first=True)
    rsasa_unpacked, rsasa_lengths = pad_packed_sequence(rsasa_packed, batch_first=True)
    shapes_unpacked, shapes_lengths = pad_packed_sequence(surf_shapes_packed, batch_first=True)

    # TODO: inspect why for some reason nans are generated in the surface shapes
    shapes_unpacked = torch.nan_to_num(shapes_unpacked)

    # unpadding the distances and flattening them
    dist_flat = torch.FloatTensor([v for (vals, l) in zip(dist_unpacked, dist_lengths) for v in vals[:l]])
    rsasa_flat = torch.FloatTensor([v for (vals, l) in zip(rsasa_unpacked, rsasa_lengths) for v in vals[:l]])
    shapes_flat = torch.FloatTensor([v for (vals, l) in zip(shapes_unpacked, shapes_lengths) for v in vals[:l]])

    # mask out AA that do not appear in the structure
    # mask removes the padding as well
    masks_unpacked = masks_unpacked.to(device)
    distance_loss = torch.sqrt(mse_loss(logits_distance[masks_unpacked], dist_flat.to(device)) + eps)
    rsasa_loss = mae_loss(logits_rsasa[masks_unpacked], rsasa_flat.to(device))
    shape_loss = torch.sqrt(mse_loss(logits_shape[masks_unpacked], shapes_flat.to(device)) + eps)

    if backward:
        # backward inside the surface heads
        distance_loss.mean().backward()
        distance_grad = logits_detach.grad

        # reset grad to None
        logits_detach.grad = None
        shape_loss.mean().backward()
        shape_grad = logits_detach.grad

        # reset grad to None
        logits_detach.grad = None
        rsasa_loss.mean().backward()
        rsasa_grad = logits_detach.grad

        # backward to LSTM model depending on weight
        # when 0 it does not influence grads in LSTM
        inputs = [p for p in model.skipLSTM.layers.parameters() if p.requires_grad]
        if inputs:
            padded_output.backward(
                (distance_grad * dist_weight) + (shape_grad * shape_weight) + (rsasa_grad * rsasa_weight),
                inputs=inputs)

    dist_corr, rsasa_corr, shape_corr = [], [], []
    # calculate correlation for each sequence in the batch
    # the mask removes the padding as well, len(mask) == len(seq)
    for i in range(seqs_unpacked.shape[0]):
        dist_corr.append(
            pearsonr(logits_distance[i][masks_unpacked[i]].detach().cpu(), dist_unpacked[i][:dist_lengths[i]])[0])
        rsasa_corr.append(
            pearsonr(logits_rsasa[i][masks_unpacked[i]].detach().cpu(), rsasa_unpacked[i][:rsasa_lengths[i]])[0])
        shape_corr.append(
            pearsonr(logits_shape[i][masks_unpacked[i]].detach().cpu(), shapes_unpacked[i][:shapes_lengths[i]])[0])

    return distance_loss.mean().item(), shape_loss.mean().item(), rsasa_loss.mean().item(), \
        np.mean(dist_corr), np.mean(shape_corr), np.mean(rsasa_corr)


def eval_surface(model, device, data_it, use_multi_gpu, apply_sigmoid=True):
    dist_losses, shape_losses, rsasa_losses, dist_corr, shape_corr, rsasa_corr = [], [], [], [], [], []
    for batch in data_it:
        d_loss, s_loss, rsasa_loss, d_corr, s_corr, sas_corr = surface_pass(model, device, batch, use_multi_gpu,
                                                                            backward=False,
                                                                            apply_sigmoid=apply_sigmoid)

        dist_losses.append(d_loss)
        shape_losses.append(s_loss)
        rsasa_losses.append(rsasa_loss)
        dist_corr.append(d_corr)
        shape_corr.append(s_corr)
        rsasa_corr.append(sas_corr)

    return np.mean(dist_losses), np.mean(shape_losses), np.mean(rsasa_losses), np.mean(dist_corr), \
        np.mean(shape_corr), np.mean(rsasa_corr)


def interface_pass(model, device, batch, use_multi_gpu, backward=True, interface_weight=1.0,
                   interface_dist_weight=1.0, delta_sasa_weight=1.0, apply_sigmoid=True):
    seqs_packed, seqs_masks_packed, ifaces_packed, dist_ifaces_packed, delta_sasas_packed = batch

    seqs_unpacked, lens_unpacked = pad_packed_sequence(seqs_packed, batch_first=True)
    padded_output = model(seqs_unpacked.to(device), lens_unpacked, apply_proj=False)

    model = model.module if use_multi_gpu else model

    # we break the computational graph here, so we can update weights
    # of surface head without changing anything to the LM model
    logits_detach = padded_output.detach()
    logits_detach.requires_grad = True

    # get the logits for the interface prediction
    # projecting LSTM dim -> 1 dim, and remove redundant dim
    logits_iface = model.predict_aa_interface(logits_detach).squeeze(-1)
    logits_dist_iface = model.predict_aa_distance_interface(logits_detach).squeeze(-1)
    logits_delta_sasa = model.predict_delta_sasa(logits_detach).squeeze(-1)

    if apply_sigmoid:
        logits_dist_iface = torch.sigmoid(logits_dist_iface)
        logits_delta_sasa = torch.sigmoid(logits_delta_sasa)

    # unpadding the interfaces and flattening them
    ifaces_unpacked, ifaces_lengths = pad_packed_sequence(ifaces_packed, batch_first=True)
    ifaces_flat = torch.FloatTensor([v for (vals, l) in zip(ifaces_unpacked, ifaces_lengths) for v in vals[:l]])

    dist_ifaces_unpacked, dist_ifaces_lengths = pad_packed_sequence(dist_ifaces_packed, batch_first=True)
    dist_ifaces_flat = torch.FloatTensor(
        [v for (vals, l) in zip(dist_ifaces_unpacked, dist_ifaces_lengths) for v in vals[:l]])

    delta_sasa_unpacked, delta_sasa_lengths = pad_packed_sequence(delta_sasas_packed, batch_first=True)
    delta_sasa_flat = torch.FloatTensor(
        [v for (vals, l) in zip(delta_sasa_unpacked, delta_sasa_lengths) for v in vals[:l]])

    # mask out AA that do not appear in the structure
    # mask removes the padding as well
    masks_unpacked, _ = pad_packed_sequence(seqs_masks_packed, batch_first=True)
    masks_unpacked = masks_unpacked.to(device)

    iface_loss = F.binary_cross_entropy_with_logits(logits_iface[masks_unpacked], ifaces_flat.to(device))
    # we calculate a distance loss to the interface
    eps = 1e-8
    mse_loss = nn.MSELoss(reduction='none')
    iface_dist_loss = torch.sqrt(mse_loss(logits_dist_iface[masks_unpacked], dist_ifaces_flat.to(device)) + eps)

    # calculate delta sasa loss just for interface residues
    interface_residues_mask = (ifaces_flat == 1)
    delta_sasa_loss = torch.sqrt(mse_loss(logits_delta_sasa[masks_unpacked][interface_residues_mask],
                                          delta_sasa_flat.to(device)[interface_residues_mask]) + eps)

    if backward:
        # backward inside the surface heads
        iface_loss.mean().backward()
        iface_grad = logits_detach.grad

        # reset grad to None
        logits_detach.grad = None
        iface_dist_loss.mean().backward()
        iface_dist_grad = logits_detach.grad

        # reset grad to None
        logits_detach.grad = None
        delta_sasa_loss.mean().backward()
        delta_sasa_grad = logits_detach.grad

        # backward to LSTM model depending on weight
        # when 0 it does not influence grads in LSTM
        inputs = [p for p in model.skipLSTM.layers.parameters() if p.requires_grad]
        if inputs:
            padded_output.backward((iface_grad * interface_weight) + (iface_dist_grad * interface_dist_weight) +
                                 (delta_sasa_grad * delta_sasa_weight), inputs=inputs)

    # value between 0 and 1
    out_prob = torch.sigmoid(logits_iface[masks_unpacked])
    auc = roc_auc_score(ifaces_flat.detach().cpu(), out_prob.detach().cpu())

    return iface_loss.mean().item(), iface_dist_loss.mean().item(), delta_sasa_loss.mean().item(), auc, \
        ifaces_flat.detach().cpu(), out_prob.detach().cpu(),


def eval_interface(model, device, data_it, use_multi_gpu, apply_sigmoid=True):
    losses, losses_dist, ground_truths, out_probs, losses_delta_sasa = [], [], [], [], []
    for batch in data_it:
        loss, loss_dist, loss_delta_sasa, _, ground_truth, out_prob = interface_pass(model, device, batch, use_multi_gpu, backward=False,
                                                                    apply_sigmoid=apply_sigmoid)
        losses.append(loss)
        losses_dist.append(loss_dist)
        losses_delta_sasa.append(loss_delta_sasa)
        ground_truths += ground_truth
        out_probs += out_prob

    auc = roc_auc_score(ground_truths, out_probs)
    return np.mean(losses), np.mean(losses_dist), np.mean(losses_delta_sasa), auc


def average_gradients(dist, model, world_size):
    """Calculates the average of the gradients over all gpus.
    """
    for param in model.parameters():
        if param.grad is not None:
            dist.barrier()
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
    return True


def freeze_unfreeze_params(params, freeze=True):
    for param in params:
        param.requires_grad = not freeze


# ---------------------------------API PARSING-------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser('Script for training multitask embedding model')

    data_path = '/scratch/ii1g17/protein-embeddings/'
    scop_path = f'{data_path}data/SCOPe/'
    masif_path = f'{data_path}data/masif-ppis/'
    astral_version = "astral-scopedom-seqres-gd-sel-gs-bib-95-2.06"
    surf_name = '-average_over=3-atom-shape=True-radius-1.4-no-HETATM-surface-processed.pkl'
    interface_name = "training-closest-vertices=3-atom-interface=True-radius-1.4-sppider_interface_label=True-with-delta-sasa-processed.pkl"

    parser.add_argument('--path-train', default=f'{data_path}/uniref-2018july/uniref90/uniref90.fasta.lmdb')
    parser.add_argument('--base', default=f'{data_path}data/')
    parser.add_argument('--path-scop-train', default=f'{scop_path}{astral_version}.train.fa')
    parser.add_argument('--path-scop-test', default=f'{scop_path}{astral_version}.test.sampledpairs.txt')
    parser.add_argument('--path-cmap-test', default=f'{scop_path}{astral_version}.test.fa')

    parser.add_argument('--path-surface', default=f'{scop_path}pdbstyle-2.06-structures-radius-1.4-no-HETATM')
    parser.add_argument('--path-processed-surface-train', default=f'{scop_path}{astral_version}.train{surf_name}')
    parser.add_argument('--path-processed-surface-test', default=f'{scop_path}{astral_version}.test{surf_name}')

    parser.add_argument('--path-interface-train', default=f'{masif_path}lists/training.txt')
    parser.add_argument('--path-processed-interface-train', default=f'{masif_path}lists/{interface_name}')

    parser.add_argument('--rnn-dim', type=int, default=512, help='hidden units of RNNs (default: 512)')
    parser.add_argument('--num-layers', type=int, default=3, help='number of RNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability (default: 0)')
    parser.add_argument('--embedding-dim', type=int, default=100, help='embedding dimension (default: 100)')
    parser.add_argument('--lstm-grad-clip-val', type=int, default=None)

    parser.add_argument('-n', '--num-steps', type=int, default=2000000, help='number ot training steps (default:1 mil)')
    parser.add_argument('--max-length', type=int, default=500, help='sample seq down to (500) during training')
    parser.add_argument('-p', type=float, default=0.1, help='cloze residue masking rate (default: 0.1)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')

    parser.add_argument('--log-freq', type=int, default=10, help='logging information every e.g. 100 batches')
    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--use-multi-gpu', action='store_true', help='using multiple gpus in parallel for training')

    parser.add_argument('--similarity-weight', type=float, default=0)
    parser.add_argument('--contacts-weight', type=float, default=0)
    parser.add_argument('--cloze-weight', type=float, default=1)
    parser.add_argument('--distance-weight', type=float, default=0)
    parser.add_argument('--shape-weight', type=float, default=0)
    parser.add_argument('--rsasa-weight', type=float, default=0)
    parser.add_argument('--interface-weight', type=float, default=0)
    parser.add_argument('--interface-dist-weight', type=float, default=0)
    parser.add_argument('--delta-sasa-weight', type=float, default=0)

    parser.add_argument('--sim', action='store_true', help='grad for sim head')
    parser.add_argument('--cmap', action='store_true', help='grad for cmap head')
    parser.add_argument('--surf', action='store_true', help='grad for surf head')
    parser.add_argument('--cloze', action='store_true', help='grad for cloze head')
    parser.add_argument('--interface', action='store_true', help='grad for interface head')

    parser.add_argument('--similarity-batch-size', type=int, default=16,
                        help='minibatch size for SCOP similarity loss (default: 16)')
    parser.add_argument('--cloze-batch-size', type=int, default=64,
                        help='minibatch size for the cloze loss (default: 64)')
    parser.add_argument('--contacts-batch-size', type=int, default=16,
                        help='minibatch size for contact maps (default: 16)')
    parser.add_argument('--surface-batch-size', type=int, default=16,
                        help='minibatch size for surface (default: 16)')
    parser.add_argument('--interface-batch-size', type=int, default=16,
                        help='minibatch size for interface (default: 16)')

    parser.add_argument('--tau', type=float, default=0.5,
                        help='smoothing on the similarity sampling distribution (default: 0.5)')
    parser.add_argument('--augment', type=float, default=0.05,
                        help='resample amino acids during training with this probability (default: 0.05)')

    parser.add_argument('--hack-offset', type=int, default=0, help='hack to use 4 extra rtx8000')
    parser.add_argument('--save-interval', type=int, default=20000, help='frequency of saving (default:; 100,000)')

    parser.add_argument("--model-checkpoint", type=str, default=None, help="path to model to load")
    parser.add_argument("--pretrained-encoder", type=str, default=None, help="path to encoder to load")

    parser.add_argument('--alphabet-type', type=str, default='uniprot21', help='alphabet to use (default: uniprot21)')
    parser.add_argument('--normalise-surf-info', action='store_true', help='normalise surface distance or rsasa')
    parser.add_argument('--normalise-interface-info', action='store_true', help='normalise interface distance')

    parser.add_argument('--masif-interfaces', action='store_true')
    parser.add_argument('--dropout-surf', type=float, default=0.3, help='dropout probability (default: 0.3)')
    parser.add_argument('--dropout-interface', type=float, default=0, help='dropout probability (default: 0)')
    parser.add_argument('--interface-augment-prob', type=float, default=0)
    parser.add_argument("--interface-augment-type", type=str, default="pfam-hmm")
    parser.add_argument('--backprop-lm-step', type=int, default=5000, help='backprop lm step (default: 100000)')

    parser.add_argument('--interface-num-workers', type=int, default=1,
                        help='number of workers for interface dataloader')
    parser.add_argument('--interface-val-percentage', type=float, default=0.1)
    parser.add_argument('--surface-num-workers', type=int, default=1)
    parser.add_argument('--increased-model-size', action='store_true', help='ignore projection loading')
    parser.add_argument("--ignore-opt-sch", action='store_true')

    parser.add_argument('--save-early-checkpoints', action='store_true')


    args = parser.parse_args()
    # ---------------------------------DISTRIBUTED PARALLEL TRAINING---------------------------------------------------

    output = sys.stdout if args.output is None else open(args.output, 'w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_multi_gpu = args.use_multi_gpu

    rank = 0
    experiment_id = "single-gpu"
    if use_multi_gpu:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"]) + args.hack_offset
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)

        multi_gpu_info(rank, world_size, gpus_per_node, local_rank)
        device = local_rank
        output = open(os.devnull, 'w') if rank != 0 else output

        # we divide by nr_gpus because each GPU gets a section of the data
        args.num_steps = args.num_steps // world_size

        experiment_id = f"multitask/{str(os.environ['SLURM_JOB_NAME'])}/{str(os.environ['SLURM_JOB_ID'])}"

    writer = SummaryWriter(log_dir=f"/scratch/ii1g17/protein-embeddings/runs/{experiment_id}")
    pprint.pprint(args.__dict__, width=1, stream=output)

    # ----------------------------------------DATASETS and DATA LOADERS------------------------------------------------
    augment = MultinomialResample('cpu', args.augment) if args.augment > 0 else None
    alphabet = alphabets.get_alphabet(args.alphabet_type)

    # Shuffle is False by default (test/valid). During training we can't shuffle because we provide a sampler
    # ----------------------------- SIMILARITY----------------------------------------
    if args.sim:
        scop_train = SCOPeDataset(path=args.path_scop_train, alphabet=alphabet)

        y_train_levels = torch.cumprod((scop_train.y.unsqueeze(1) == scop_train.y.unsqueeze(0)).float(), 2)
        scop_train = AllPairsDataset(scop_train.x, y_train_levels, augment=augment)

        weights = get_protein_pairs_weights(y_train_levels, args.tau)
        sampler = LargeWeightedRandomSampler(weights, args.similarity_batch_size * args.num_steps)

        sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank) if use_multi_gpu else None
        sim_train_iterator = itr_restart(DataLoader(scop_train, batch_size=args.similarity_batch_size,
                                                    sampler=sampler, collate_fn=collate_paired_sequences))
        if rank == 0:
            sim_test_iterator = DataLoader(SCOPePairsDataset(path=args.path_scop_test, alphabet=alphabet),
                                           batch_size=args.similarity_batch_size,
                                           collate_fn=collate_paired_sequences)

    # ----------------------------- CONTACTS------------------------------------------
    if args.cmap:
        contacts_train = ContactMapDataset(path=args.path_scop_train, min_length=20, max_length=1000, alphabet=alphabet,
                                           augment=augment)

        sampler = DistributedSampler(contacts_train, num_replicas=world_size, rank=rank) if use_multi_gpu else None
        cmap_train_iterator = itr_restart(DataLoader(contacts_train, batch_size=args.contacts_batch_size,
                                                     collate_fn=collate_lists, sampler=sampler))
        if rank == 0:
            cmap_test_iterator = DataLoader(ContactMapDataset(path=args.path_cmap_test, alphabet=alphabet),
                                            batch_size=args.contacts_batch_size,
                                            collate_fn=collate_lists)
    # ----------------------------- SURFACE---------------------------------------------
    # if not using the processed surface dataset, the SASA calculation takes a lot, needs more workers
    if args.surf:
        surface_train = ProcessedSurfaceDataset(ScopSurfaceDataset(args.path_scop_train, args.path_surface),
                                                alphabet=alphabet, normalise=args.normalise_surf_info,
                                                max_length=args.max_length,
                                                processed_path=args.path_processed_surface_train,
                                                augment=augment)

        sampler = DistributedSampler(surface_train, num_replicas=world_size, rank=rank) if use_multi_gpu else None
        surface_train_iterator = itr_restart(DataLoader(surface_train, batch_size=args.surface_batch_size,
                                                        collate_fn=collate_surface_info, sampler=sampler,
                                                        num_workers=args.surface_num_workers))
        if rank == 0:
            surface_test = ProcessedSurfaceDataset(ScopSurfaceDataset(args.path_cmap_test, args.path_surface),
                                                   alphabet=alphabet, normalise=args.normalise_surf_info,
                                                   processed_path=args.path_processed_surface_test)
            surface_test_iterator = DataLoader(surface_test, batch_size=args.surface_batch_size,
                                               collate_fn=collate_surface_info, num_workers=4)

    # ------------------------------Interface------------------------------------------
    if args.interface:
        interface_train = InterfaceDataset(args.path_interface_train, args.base, args.masif_interfaces, alphabet,
                                           output, max_length=args.max_length, atom_interface=True,
                                           processed_path=args.path_processed_interface_train,
                                           normalise=args.normalise_interface_info,
                                           augment_prob=args.interface_augment_prob,
                                           augment_type=args.interface_augment_type)
        val_size = int(args.interface_val_percentage / 100 * len(interface_train))
        train_size = len(interface_train) - val_size
        interface_train, interface_validation = random_split(interface_train, [train_size, val_size],
                                                             generator=torch_generator)

        sampler = DistributedSampler(interface_train, world_size, rank) if use_multi_gpu else None
        interface_train_iterator = itr_restart(DataLoader(interface_train, args.interface_batch_size,
                                                          collate_fn=collate_interface, sampler=sampler,
                                                          num_workers=args.interface_num_workers))
        if rank == 0:
            interface_val_iterator = DataLoader(interface_validation, args.interface_batch_size,
                                                collate_fn=collate_interface, num_workers=2)

    # ----------------------------- CLOZE---------------------------------------------
    if args.cloze:
        fasta_train = LMDBDataset(args.path_train, max_length=args.max_length,
                                  alphabet=alphabet, output=output)
        cloze_train = ClozeDataset(fasta_train, args.p, noise=fasta_train.noise)

        val_size = int(0.1 / 100 * len(cloze_train))
        train_size = len(cloze_train) - val_size
        cloze_train, cloze_validation = random_split(cloze_train, [train_size, val_size], generator=torch_generator)
        # the split for the lengths is the same as the one in the dataset
        cloze_train_lengths, _ = random_split(fasta_train.lengths, [train_size, val_size], generator=torch_generator)

        weight = np.maximum(np.array(cloze_train_lengths) / args.max_length, 1)
        sampler = LargeWeightedRandomSampler(weight, args.cloze_batch_size * args.num_steps)

        sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank) if use_multi_gpu else sampler
        cloze_iterator = itr_restart(DataLoader(cloze_train, batch_size=args.cloze_batch_size, sampler=sampler,
                                                collate_fn=collate_protein_seq, num_workers=2))

        if rank == 0:
            cloze_val_it = DataLoader(cloze_validation, batch_size=args.cloze_batch_size,
                                      collate_fn=collate_protein_seq, num_workers=2)

    # ----------------------------------------MODEL CREATION----------------------------------------------------------
    encoder = SkipLSTM(21, args.embedding_dim, args.rnn_dim, args.num_layers, dropout=args.dropout)

    scop_predict = OrdinalRegression(args.embedding_dim, 5, compare=L1()) if args.sim else None
    cmap_predict = BilinearContactMap(encoder.cloze.in_features) if args.cmap else None
    surface_predict = SurfacePredictor(encoder.cloze.in_features, 21, args.dropout_surf) if args.surf else None
    interface_predict = InterfacePredictor(encoder.cloze.in_features, 21,
                                           args.dropout_interface) if args.interface else None

    model = ProSEMT(encoder, scop_predict, cmap_predict, surface_predict, interface_predict)

    step = 0
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    # linearly decayed to one tenth of its peak value over the 90% of training duration.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: 1 - s / args.num_steps * 0.9)
    print('Model size: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6), file=output)

    # ----------------------------------------LOAD CHECKPOINT----------------------------------------------------------

    map_location = {'cuda:0': 'cuda:%d' % local_rank} if use_multi_gpu else None
    if args.model_checkpoint is not None:
        print("--------USING CHECKPOINT MODEL-----------", file=output)
        checkpoint = torch.load(args.model_checkpoint, map_location=map_location)
        model.load_state_dict(checkpoint['model'], strict=False)
        if not args.ignore_opt_sch:
            optimiser.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        step = checkpoint['step']

    frozen_params = []
    if args.pretrained_encoder is not None:
        print("--------USING PRETRAINED ENCODER-----------", file=output)
        checkpoint = torch.load(args.pretrained_encoder, map_location=map_location)
        # keep only the encoder
        encoder_checkpoint = {k: v for k, v in checkpoint['model'].items() if k.startswith('skipLSTM')
                              and not k.startswith('skipLSTM.proj')}

        # proj head needs to be reinitialised as it is not the same size
        if args.increased_model_size:
            step = checkpoint['step']
            encoder_checkpoint = {k: v for k, v in encoder_checkpoint.items() if not k.startswith('skipLSTM.cloze')}

        model.load_state_dict(encoder_checkpoint, strict=False)
        # freeze the loaded bit until the unfreeze step is reached
        frozen_params = [param for name, param in model.named_parameters() if name in encoder_checkpoint]
        freeze_unfreeze_params(frozen_params, freeze=True)

    # ----------------------------------------DISTRIBUTED----------------------------------------------------------
    if use_multi_gpu:
        model = DDP(model, device_ids=[local_rank])
    model.train()

    # ----------------------------------------------TRAINING-----------------------------------------------------------
    weights = [args.similarity_weight, args.cloze_weight, args.contacts_weight, args.distance_weight,
               args.shape_weight, args.rsasa_weight, args.interface_weight, args.interface_dist_weight,
               args.delta_sasa_weight]
    args.__dict__.update(zip(weights, [w / sum(weights) for w in weights]))
    print('similarity weight, cloze weight, contacts weight, distance weight, shape weight, rsasa weight,'
          ' interface weight \n', weights, file=output)

    start_time = time.time()
    for i in range(step, args.num_steps):

        if i - step == args.backprop_lm_step and args.pretrained_encoder is not None:
            print('Unfreezing encoder layers', file=output)
            freeze_unfreeze_params(frozen_params, freeze=False)

        with model.no_sync() if use_multi_gpu else nullcontext() as gs:
            # ----------------------------------------------SIMILARITY-------------------------------------------------
            if args.sim:
                x0, x1, y = next(sim_train_iterator)
                # we go through this data, but we don't use the loss to contribute towards the grad optimisation
                # we just want to assign gradients such that the parameters are updated (if loss weight is 0)
                sim_loss, sim_accuracy, sim_mse, batch_size = batch_similarity_grad(model, x0, x1, y, device,
                                                                                    use_multi_gpu,
                                                                                    weight=args.similarity_weight)
            # ----------------------------------------------CONTACTS---------------------------------------------------
            if args.cmap:
                c_x, c_y = next(cmap_train_iterator)
                cmap_loss, true_positives, ground_positives, predicted_positives, batch_size = \
                    cmap_grad(model, c_x, c_y, device, use_multi_gpu, weight=args.contacts_weight)
                cmap_precision = true_positives / (predicted_positives + 1e-8)

            # ----------------------------------------------SURFACE -------------------------------------------------
            if args.surf:
                batch = next(surface_train_iterator)
                d_loss, s_loss, sas_loss, d_corr, s_corr, sas_corr = surface_pass(model, device, batch, use_multi_gpu,
                                                                                  dist_weight=args.distance_weight,
                                                                                  shape_weight=args.shape_weight,
                                                                                  rsasa_weight=args.rsasa_weight,
                                                                                  apply_sigmoid=args.normalise_surf_info)

            # ----------------------------------------------INTERFACE--------------------------------------------------
            if args.interface:
                batch = next(interface_train_iterator)
                interface_loss, interface_dist_loss, delta_sasa_loss, interface_auc, _, _ = interface_pass(model, device, batch,
                                                                                          use_multi_gpu,
                                                                                          backward=True,
                                                                                          interface_weight=args.interface_weight,
                                                                                          interface_dist_weight=args.interface_dist_weight,
                                                                                          delta_sasa_weight=args.delta_sasa_weight,
                                                                                          apply_sigmoid=args.normalise_interface_info)
            # ----------------------------------------------LANGUAGE MODEL---------------------------------------------
            if args.cloze:
                seqs, seqs_lengths, labels = next(cloze_iterator)
                cloze_loss, correct, nr_masked = cloze_grad(model, seqs, seqs_lengths, labels, device,
                                                            weight=args.cloze_weight, use_multi_gpu=args.use_multi_gpu)
                if nr_masked > 0:
                    cloze_acc = correct / nr_masked

        # ----------------------------------------------PARAMETER UPDATES----------------------------------------------

        '''gradients are not syncronized because of multitask and reentrant backwards, so no_syc() was used
           we have to syncronize them manually (no time wasted)
        this also makes processing wait when testing is done on 1 process because they have reached a barrier
        but the one doing the testing (rank 0) has not encountered the barrier yet'''
        if args.use_multi_gpu:
            average_gradients(dist, model, world_size)

        if args.lstm_grad_clip_val is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.lstm_grad_clip_val)

        optimiser.step()
        scheduler.step()
        optimiser.zero_grad()

        #  theta should grow with probability, projected gradient for bounding ordinal regression parameters
        if args.sim:
            model.module.scop_predict.clip() if use_multi_gpu else model.scop_predict.clip()
        # ----------------------------------------------TRAINING UPDATES----------------------------------------------
        if i % args.log_freq == 0 and rank == 0:
            epoch_mins, epoch_secs = epoch_time(start_time, time.time())
            start_time = time.time()
            print(f'{i}/{args.num_steps} training {i / args.num_steps:.1%},\
             {epoch_mins}m {epoch_secs}s', file=output)

            if args.sim:
                multitask_train_info(writer, sim_loss, sim_accuracy, i, data="train", name="SIMILARITY")
            if args.cmap:
                multitask_train_info(writer, cmap_loss, cmap_precision, i, data="train", name="CMAP")
            if args.cloze and nr_masked > 0:
                multitask_train_info(writer, np.exp(cloze_loss), cloze_acc, i, data="train", name="LM")

            if args.surf:
                writer.add_scalars("SURF-dist" + '/loss', {"train": d_loss}, i)
                writer.add_scalars("SURF-dist" + '/corr', {"train": d_corr}, i)
                writer.add_scalars("SURF-shape" + '/loss', {"train": s_loss}, i)
                writer.add_scalars("SURF-shape" + '/corr', {"train": s_corr}, i)
                writer.add_scalars("SURF-rsasa" + '/loss', {"train": sas_loss}, i)
                writer.add_scalars("SURF-rsasa" + '/corr', {"train": sas_corr}, i)

            if args.interface:
                writer.add_scalars("INTERFACE" + '/AUC', {"train": interface_auc}, i)
                writer.add_scalars("INTERFACE" + '/loss', {"train": interface_loss}, i)
                writer.add_scalars("INTERFACE" + '/dist_loss', {"train": interface_dist_loss}, i)
                writer.add_scalars("INTERFACE" + '/delta_sasa_loss', {"train": delta_sasa_loss}, i)

        # evaluate and save model
        checkpoints = [0, 10, 50, 100, 250, 500, 750, 1000, 2500, 5000] if args.save_early_checkpoints else []
        if (i % args.save_interval == 0 or i in checkpoints) and rank == 0:
            # ----------------------------------------------TESTING----------------------------------------------------
            # eval and save model
            model.eval()
            with torch.no_grad():
                if args.sim:
                    loss, accuracy, mse, r, rho, aupr = eval_scop(model, sim_test_iterator, device, use_multi_gpu)
                    print('\nSIMILARITY test\n', '\t'.join(['loss', 'mse', 'accuracy', 'r', 'rho', 'class', 'fold',
                                                            'superfamily', 'family']), file=output)
                    print('\t'.join(['{:.5f}'.format(x) for x in [loss, mse, accuracy, r, rho, aupr[0], aupr[1],
                                                                  aupr[2], aupr[3]]]), file=output)
                    multitask_train_info(writer, loss, accuracy, i, data="test", name="SIMILARITY")

                if args.cmap:
                    cmap_loss_t, cmap_pr, cmap_re, cmap_f1, cmap_aupr = \
                        eval_cmap(model, cmap_test_iterator, device, use_multi_gpu)
                    print('\nCONTACT test\n', '\t'.join(['rrc_loss', 'rrc_pr', 'rrc_re', 'rrc_f1', 'rrc_aupr']),
                          file=output)
                    print('\t'.join(['{:.5f}'.format(x) for x in [cmap_loss_t, cmap_pr, cmap_re, cmap_f1, cmap_aupr]]),
                          file=output)
                    multitask_train_info(writer, cmap_loss_t, cmap_pr, i, data="test", name="CMAP")

                if args.cloze:
                    cloze_val_loss, cloze_val_acc = eval_cloze(model, cloze_val_it, device)
                    multitask_train_info(writer, np.exp(cloze_val_loss), cloze_val_acc, i, data="val", name="LM")

                if args.surf:
                    d_loss, s_loss, sas_loss, d_corr, s_corr, sas_corr = eval_surface(model, device,
                                                                                      surface_test_iterator,
                                                                                      use_multi_gpu,
                                                                                      apply_sigmoid=args.normalise_surf_info)

                    writer.add_scalars("SURF-dist" + '/loss', {"test": d_loss}, i)
                    writer.add_scalars("SURF-shape" + '/loss', {"test": s_loss}, i)
                    writer.add_scalars("SURF-rsasa" + '/loss', {"test": sas_loss}, i)
                    writer.add_scalars("SURF-rsasa" + '/corr', {"test": sas_corr}, i)
                    writer.add_scalars("SURF-dist" + '/corr', {"test": d_corr}, i)
                    writer.add_scalars("SURF-shape" + '/corr', {"test": s_corr}, i)

                if args.interface:
                    interface_loss, interface_dist_los, delta_sasa_loss, auc = eval_interface(model, device, interface_val_iterator,
                                                                             use_multi_gpu,
                                                                             apply_sigmoid=args.normalise_interface_info)
                    writer.add_scalars("INTERFACE" + '/AUC', {"val": auc}, i)
                    writer.add_scalars("INTERFACE" + '/loss', {"val": interface_loss}, i)
                    writer.add_scalars("INTERFACE" + '/dist_loss', {"val": interface_dist_los}, i)
                    writer.add_scalars("INTERFACE" + '/delta_sasa_loss', {"val": delta_sasa_loss}, i)

            # ----------------------------------------------SAVING MODEL-------------------------------------------
            save_checkpoint(model, optimiser, i, experiment_id, use_multi_gpu, scheduler)

            # flip back to train mode
            model.train()

    if rank == 0:
        save_checkpoint(model, optimiser, args.num_steps, experiment_id, use_multi_gpu, scheduler)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()