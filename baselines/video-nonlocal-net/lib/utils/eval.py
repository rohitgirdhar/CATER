from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np
import sys
from collections import defaultdict

from core.config import config as cfg
from utils.localization_eval import all_localization_accuracies

# Imported from @achalddave's AP implementations
from ap import compute_multiple_aps

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def evaluate_result(results, hard=False):
    if cfg.MODEL.MULTILABEL:
        return evaluate_result_multiLabel(results, hard=hard)
    else:
        return evaluate_result_singleLabel(results)


def read_groundtruth(filename_gt, singleLabel=False):
    f = open(filename_gt, 'r')
    labels = []
    for line in f:
        rows = line.split()
        # handle mulit-label case
        label = [int(el) for el in rows[1].split(',')]
        if singleLabel:
            assert len(label) == 1
            label = label[0]
        labels.append(label)
    f.close()
    return labels


def evaluate_result_singleLabel(results):
    gt_labels = read_groundtruth(cfg.FILENAME_GT, singleLabel=True)
    probs = {}
    counts = {}
    for entry in results:
        vid = entry[0]
        prob = np.array(entry[1])
        if vid in probs:
            probs[vid] += prob
            counts[vid] += 1
        else:
            probs[vid] = prob
            counts[vid] = 1
    # Average over the clips
    # top_1 = 0.
    # top_5 = 0.
    nclasses = max(gt_labels) + 1  # since it starts from 0.
    preds = np.full((len(probs), nclasses), np.nan, dtype=np.float32)
    labels = np.full((len(probs),), np.nan, dtype=np.int32)
    for vid in probs.keys():
        probs[vid] /= counts[vid]
        gt_lbl = gt_labels[vid]
        preds[vid, :] = probs[vid]
        labels[vid] = gt_lbl
    assert not np.any(np.isnan(preds))
    assert not np.any(np.isnan(labels))
    #     top_preds = np.argsort(probs[vid])[::-1]
    #     top_1 += (gt_lbl in top_preds[:1])
    #     top_5 += (gt_lbl in top_preds[:5])
    # res = {
    #     'top_1': top_1 / len(probs),
    #     'top_5': top_5 / len(probs),
    # }
    # return res
    # Moving to a new way of computing accuracies, that is shared across all codes
    return all_localization_accuracies(labels, preds)


# This code is written way too badly, better just redo it
# def evaluate_result_singleLabel(results):
#     gt_labels = read_groundtruth(cfg.FILENAME_GT, singleLabel=True)
#
#     sample_num = cfg.TEST.DATASET_SIZE
#     class_num = cfg.MODEL.NUM_CLASSES
#     sample_video_times = cfg.TEST.NUM_TEST_CLIPS
#     counts = np.zeros(sample_num, dtype=np.int32)
#     probs = np.zeros((sample_num, class_num))
#
#     # assert(len(gt_labels) == sample_num)
#
#     """
#     clip_accuracy: the (e.g.) 10*19761 clips' average accuracy
#     clip1_accuracy: the 1st clip's accuracy (starting from frame 0)
#     """
#     clip_accuracy = 0
#     clip1_accuracy = 0
#     clip1_count = 0
#     seen_inds = defaultdict(int)
#
#     # evaluate
#     for entry in results:
#         vid = entry[0]
#         prob = np.array(entry[1])
#         probs[vid] += prob[0: class_num]
#         counts[vid] += 1
#
#         idx = prob.argmax()
#         if idx == gt_labels[vid]:
#             # clip accuracy
#             clip_accuracy += 1
#
#         # clip1 accuracy
#         seen_inds[vid] += 1
#         if seen_inds[vid] == 1:
#             clip1_count += 1
#             if idx == gt_labels[vid]:
#                 clip1_accuracy += 1
#
#     # sanity checkcnt = 0
#     max_clips = 0
#     min_clips = sys.maxsize
#     count_empty = 0
#     count_corrupted = 0
#     for i in range(sample_num):
#         max_clips = max(max_clips, counts[i])
#         min_clips = min(min_clips, counts[i])
#         if counts[i] != sample_video_times:
#             count_corrupted += 1
#             logger.warning('Id: {} count: {}'.format(i, counts[i]))
#         if counts[i] == 0:
#             count_empty += 1
#
#     logger.info('Num of empty videos: {}'.format(count_empty))
#     logger.info('Num of corrupted videos: {}'.format(count_corrupted))
#     logger.info('Max num of clips in a video: {}'.format(max_clips))
#     logger.info('Min num of clips in a video: {}'.format(min_clips))
#
#     # clip1 accuracy for sanity (# print clip1 first as it is lowest)
#     logger.info('Clip1 accuracy: {:.2f} percent ({}/{})'.format(
#         100. * clip1_accuracy / clip1_count, clip1_accuracy, clip1_count))
#
#     # clip accuracy for sanity
#     final_clip_acc = 100. * clip_accuracy / len(results)
#     logger.info('Clip accuracy: {:.2f} percent ({}/{})'.format(
#         final_clip_acc, clip_accuracy, len(results)))
#
#     # compute accuracy
#     accuracy = 0
#     accuracy_top5 = 0
#
#     for i in range(sample_num):
#         prob = probs[i]
#
#         # top-1
#         idx = prob.argmax()
#         if idx == gt_labels[i] and counts[i] > 0:
#             accuracy = accuracy + 1
#
#         ids = np.argsort(prob)[::-1]
#         for j in range(5):
#             if ids[j] == gt_labels[i] and counts[i] > 0:
#                 accuracy_top5 = accuracy_top5 + 1
#                 break
#
#     accuracy = float(accuracy) / float(sample_num)
#     accuracy_top5 = float(accuracy_top5) / float(sample_num)
#     final_top1 = accuracy * 100
#     final_top5 = accuracy_top5 * 100
#
#     logger.info('-' * 80)
#     logger.info('top-1 accuracy: {:.2f} percent'.format(final_top1))
#     logger.info('top-5 accuracy: {:.2f} percent'.format(final_top5))
#     logger.info('-' * 80)
#     return {'clip': final_clip_acc,
#             'top-1': final_top1,
#             'top-5': final_top5}


def get_labels_and_prob(results, hard=False):
    gt_labels = read_groundtruth(cfg.FILENAME_GT)

    sample_num = cfg.TEST.DATASET_SIZE
    class_num = cfg.MODEL.NUM_CLASSES
    sample_video_times = cfg.TEST.NUM_TEST_CLIPS
    counts = np.zeros(sample_num, dtype=np.int32)
    probs = np.zeros((sample_num, class_num))

    assert(len(gt_labels) == sample_num)

    # evaluate
    for entry in results:
        vid = entry[0]
        prob = np.array(entry[1])
        probs[vid] += prob[0: class_num]
        counts[vid] += 1

    # sanity checkcnt = 0
    max_clips = 0
    min_clips = sys.maxsize
    count_empty = 0
    count_corrupted = 0
    for i in range(sample_num):
        max_clips = max(max_clips, counts[i])
        min_clips = min(min_clips, counts[i])
        if counts[i] != sample_video_times:
            count_corrupted += 1
            logger.warning('Id: {} count: {}'.format(i, counts[i]))
        if counts[i] == 0:
            count_empty += 1

    logger.info('Num of empty videos: {}'.format(count_empty))
    logger.info('Num of corrupted videos: {}'.format(count_corrupted))
    logger.info('Max num of clips in a video: {}'.format(max_clips))
    logger.info('Min num of clips in a video: {}'.format(min_clips))

    # convert GT labels into a 0-1 matrix as well
    labels = np.zeros((sample_num, class_num))
    for i, lbl in enumerate(gt_labels):
        labels[i, lbl] = 1

    return labels, probs


def evaluate_result_multiLabel(results, hard=False):
    if hard:
        raise NotImplementedError('Something I was trying '
                                  'to implement to make '
                                  'numbers lower.')
    labels, probs = get_labels_and_prob(results, hard=hard)
    ap = compute_multiple_aps(labels, probs)
    map = np.mean([el for el in ap if el >= 0])

    logger.info('-' * 80)
    # logger.info('AP per class: {}'.format(ap))
    # ignore the -1s, or class not present, for mAP
    logger.info('mAP: {}'.format(map))
    logger.info('-' * 80)
    return {'map': map, 'per_class_ap': ap}
