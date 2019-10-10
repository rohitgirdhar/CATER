# coding: utf-8
import argparse
import time
import os.path as osp
import torch
import torch.nn as nn
import torch.onnx
import subprocess
import h5py
import numpy as np
import pickle as pkl
import random
import sys

import model

# Imported from @achalddave's AP implementations
ap_impl_path = osp.join(osp.dirname(osp.realpath(__file__)), '../video-nonlocal-net/external_lib/average-precision/python/')  # noQA
sys.path.insert(0, ap_impl_path)
from ap import compute_multiple_aps

# Common localization evaluation codes
loc_eval_path = osp.join(osp.dirname(osp.realpath(__file__)), '../video-nonlocal-net/lib/utils/')  # noQA
sys.path.insert(0, loc_eval_path)
from localization_eval import compute_top_k_acc, l1_dist_labels


NRUNS = 3   # Average the predictions over this many runs
eval_batch_size = 10

parser = argparse.ArgumentParser(description='PyTorch video lstm')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Path where the h5 files are stored.'
                         'Should be the path to .tar file.')
parser.add_argument('--lbl-dir', type=str, default=None,
                    help='Path where the GT files for this model are stored. '
                         'Must be set for NL models, since labels are not '
                         'stored in the PKL files.')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net '
                    '(RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--nclasses', type=int, default=36,
                    help='Number of classes to classify into.')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--multilabel', action='store_true',
                    help='Set, to use multilabel loss and evaluation.')
args = parser.parse_args()

main_data_dir = args.data_dir
if osp.isfile(main_data_dir):
    main_data_dir = osp.dirname(main_data_dir)
args.save = osp.join(main_data_dir, 'lstm', 'model.pt')
subprocess.call('mkdir -p {}'.format(osp.dirname(args.save)),
                shell=True)

###############################################################################
# Load data
###############################################################################


def compute_map(labels, preds):
    ap = compute_multiple_aps(labels, preds)
    map = np.mean([el for el in ap if el >= 0])
    return map


def top_k_acc(labels, preds, k):
    """Compute accuracy.
    Args:
        labels: (N,) for softmax case, (N, C) for sigmoid case
        preds: (N, C)
    """
    if args.multilabel:
        return compute_map(labels, preds)
    return compute_top_k_acc(labels, preds, k)


def evaluate_everything(labels, preds):
    """Writing this as the common evaluation code for all
    localization setups.
    Args:
        labels: (N,) for softmax case, (N, C) for sigmoid case
        preds: (N, C)
    Returns:
        A dictionary with top-1, top-5 etc accuracies.
    """
    final_res = {
        'top_1': top_k_acc(labels, preds, 1),
        'top_5': top_k_acc(labels, preds, 5),
        'l1_dist': l1_dist_labels(labels, preds) if not args.multilabel else -1,
    }
    return final_res


def read_data_tsn(data_file):
    with h5py.File(data_file, 'r') as fin:
        features = fin['feats'].value
        labels = fin['labels'].value
        output = fin['output'].value
    output = np.mean(output, axis=(1, 2))
    accs = evaluate_everything(labels, output)
    print('Average pool accuracy of this model: {} is {}'.format(
        data_file, accs))
    # Slice out the center crop, in case I tested for multiple crops
    return (
        torch.FloatTensor(features[:, :, features.shape[2] // 2, :]).cuda(),
        torch.LongTensor(labels).cuda())


def read_labels_singlelabel(lbl_file):
    labels = []
    with open(lbl_file, 'r') as fin:
        for line in fin:
            labels.append(int(line.split()[-1]))
    return np.array(labels)


def read_labels_multilabel(lbl_file):
    labels = []
    with open(lbl_file, 'r') as fin:
        for line in fin:
            lbl = np.zeros((args.nclasses,))
            for el in line.split()[-1].strip().split(','):
                lbl[int(el)] = 1.0
            labels.append(lbl)
    return np.array(labels)


def read_data_nl(data_file, lbl_file):
    if args.multilabel:
        labels = read_labels_multilabel(lbl_file)
    else:
        labels = read_labels_singlelabel(lbl_file)
    NUM_SPATIAL_CROPS = 3
    NUM_TEMPORAL_CROPS = 10
    FEAT_DIM = 2048
    outputs = np.zeros((len(labels), NUM_SPATIAL_CROPS, NUM_TEMPORAL_CROPS,
                        args.nclasses))
    features = np.zeros((len(labels), NUM_SPATIAL_CROPS, NUM_TEMPORAL_CROPS,
                         FEAT_DIM))
    with open(data_file, 'rb') as fin:
        data = pkl.load(fin, encoding='latin1')  # Written using python2 code
    for el in data:
        vid_id = el[0]
        softmax_feat = el[1]
        temporal_crop_id = el[2]
        spatial_crop_id = el[3]
        logits = el[4]
        assert spatial_crop_id in range(NUM_SPATIAL_CROPS)
        assert temporal_crop_id in range(NUM_TEMPORAL_CROPS)
        outputs[vid_id, spatial_crop_id, temporal_crop_id, ...] = np.array(
            softmax_feat)
        features[vid_id, spatial_crop_id, temporal_crop_id, ...] = np.array(
            logits)
    print('Average all-crop pool accuracy of model: {}'.format(data_file))
    outputs_mean = np.mean(outputs, axis=(1, 2))
    print('Top 1: {}'.format(top_k_acc(labels, outputs_mean, 1)))
    print('Top 5: {}'.format(top_k_acc(labels, outputs_mean, 5)))
    # Keep only the center crop for the remaining
    outputs = outputs[:, 1, ...]
    features = features[:, 1, ...]
    print('Average center-crop pool accuracy of model: {}'.format(data_file))
    outputs_mean = np.mean(outputs, axis=1)
    print('Top 1: {}'.format(top_k_acc(labels, outputs_mean, 1)))
    print('Top 5: {}'.format(top_k_acc(labels, outputs_mean, 5)))
    print('First center-crop accuracy of model: {}'.format(data_file))
    outputs_mean = outputs[:, 0, ...]
    print('Top 1: {}'.format(top_k_acc(labels, outputs_mean, 1)))
    print('Top 5: {}'.format(top_k_acc(labels, outputs_mean, 5)))
    print('Last center-crop accuracy of model: {}'.format(data_file))
    outputs_mean = outputs[:, -1, ...]
    print('Top 1: {}'.format(top_k_acc(labels, outputs_mean, 1)))
    print('Top 5: {}'.format(top_k_acc(labels, outputs_mean, 5)))
    # Slice out the center crop, in case I tested for multiple crops
    if args.multilabel:
        lbl_tensor = torch.FloatTensor(labels).cuda()
    else:
        lbl_tensor = torch.LongTensor(labels).cuda()
    return (torch.FloatTensor(features).cuda(),
            lbl_tensor)


def read_data(data_dir):
    if osp.exists(args.data_dir + '_val_feats.h5'):
        print('This looks like TSN outputs, reading it so.')
        val_data = read_data_tsn(args.data_dir + '_val_feats.h5')
        train_data = read_data_tsn(args.data_dir + '_train_feats.h5')
    elif osp.exists(osp.join(
            args.data_dir, 'results_probs_test_fullLbl.pkl')):
        print('This looks like NL outputs, reading it so.')
        assert args.lbl_dir is not None, (
            'lbl_dir must be set for NL models, since the labels are not '
            'stored in the PKL file.')
        val_data = read_data_nl(
            osp.join(args.data_dir, 'results_probs_test_fullLbl.pkl'),
            osp.join(args.lbl_dir, 'val.txt'))
        train_data = read_data_nl(
            osp.join(args.data_dir, 'results_probs_train_fullLbl.pkl'),
            osp.join(args.lbl_dir, 'train.txt'))
    else:
        raise NotImplementedError('Dunno how to read data directory {}'.format(
            data_dir))
    return train_data, val_data


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
       to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bsz, random=False):
    """For simplicity, it will pick all temporal locations."""
    if not random:
        st_pos = i
        end_pos = min(i + bsz, source[0].size(0))
        pos = np.arange(st_pos, end_pos)
    else:
        # Create a batch randomly
        pos = np.random.permutation(np.arange(source[0].size(0)))[:bsz]
    data = source[0][pos, :, :]
    data = data.transpose(0, 1)
    target = source[1][pos]
    return pos, data, target


def evaluate(model_instance, data_source):
    # Turn on evaluation mode which disables dropout.
    model_instance.eval()
    # total_loss = 0.
    vids = []
    preds = []
    labels = []
    with torch.no_grad():
        for i in range(
                0, data_source[0].size(0) - 1,
                eval_batch_size):
            idx, data, targets = get_batch(
                data_source, i, eval_batch_size)
            vids += idx.tolist()
            # hidden is different for each batch in this case
            hidden = model_instance.init_hidden(eval_batch_size)
            output, hidden = model_instance(data, hidden)
            output_flat = output[output.size(0) - 1]
            if args.multilabel:
                output_flat = nn.Sigmoid()(output_flat)
            else:
                output_flat = nn.Softmax(dim=1)(output_flat)
            # output_flat = output.view(-1, args.nclasses)
            labels.append(targets.cpu().numpy())
            preds.append(output_flat.cpu().numpy())
            # total_loss += len(data) * criterion(output_flat, targets).item()
            # hidden = repackage_hidden(hidden)
    vids = np.array(vids)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    # acc = np.mean(labels == preds.argmax(axis=-1))
    final_res = evaluate_everything(labels, preds)
    print('Top-1: {}'.format(final_res['top_1']))
    print('Top-5: {}'.format(final_res['top_5']))
    return final_res['top_1'], final_res, vids, labels, preds


def train(lr, model_instance, criterion, train_data):
    # Turn on training mode which enables dropout.
    model_instance.train()
    total_loss = 0.
    # start_time = time.time()
    for batch, i in enumerate(range(
            0, train_data[0].size(0) - args.batch_size + 1,
            args.batch_size)):
        _, data, targets = get_batch(train_data, i, args.batch_size,
                                     random=True)
        # Each batch is a different time series, so should not share the
        # hidden representation across time
        hidden = model_instance.init_hidden(args.batch_size)
        # Starting each batch, we detach the hidden state from how it
        # was previously produced. If we didn't, the model would try
        # backpropagating all the way to start of the dataset.
        # hidden = repackage_hidden(hidden)
        model_instance.zero_grad()
        output, hidden = model_instance(data, hidden)
        loss = criterion(output[output.size(0) - 1], targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem
        # in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model_instance.parameters(), args.clip)
        for p in model_instance.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()


def train_lstm():
    ###########################################################################
    # Build the model
    ###########################################################################
    train_data, val_data = read_data(args.data_dir)
    model_instance = model.RNNModel(
        args.model, args.nclasses, train_data[0].size(-1),
        args.nhid, args.nlayers,
        args.dropout).cuda()

    if args.multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    lr = args.lr
    best_val_acc = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(lr, model_instance, criterion, train_data)
            val_acc, final_res, _, _, _ = evaluate(model_instance, val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'valid acc {:5.4f} | '.format(
                      epoch, (time.time() - epoch_start_time), val_acc))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so
            # far.
            if not best_val_acc or val_acc > best_val_acc:
                with open(args.save, 'wb') as f:
                    torch.save(model_instance, f)
                best_val_acc = val_acc
            else:
                # Anneal the learning rate if no improvement has been
                # seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    # Run final test, and store the predictions
    _, _, idx, _, preds = evaluate(model_instance, val_data)
    preds_to_store = []
    for idx_i, pred_i in zip(idx.tolist(), preds.tolist()):
        preds_to_store.append((idx_i, pred_i))
    with open(osp.join(osp.dirname(
            args.save), 'results_probs_lstm.pkl'), 'wb') as f:
        pkl.dump(preds_to_store, f, protocol=2)
    return final_res


def main():
    # Train LSTM multiple times since there is random variation, and then
    # avg the numbers over the runs
    all_final_res = []
    for run_id in range(NRUNS):
        print('>> Running run {}'.format(run_id))
        random.seed(42+run_id)
        torch.manual_seed(42+run_id)
        np.random.seed(42+run_id)
        final_res = train_lstm()
        all_final_res.append(final_res)

    print('Averaged over {} runs'.format(NRUNS))
    for key in all_final_res[0].keys():
        print('{}: {}'.format(
            key, np.mean([el[key] for el in all_final_res])))


if __name__ == '__main__':
    main()
