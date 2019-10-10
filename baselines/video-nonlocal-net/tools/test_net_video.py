# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np
import argparse
import sys
import pickle
import datetime
import os
import os.path as osp
import math
import cv2
from collections import defaultdict
import scipy.ndimage

from caffe2.python import workspace

from core.config import config as cfg
from core.config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg, count_lines)
from models import model_builder_video

import utils.misc as misc
import utils.checkpoints as checkpoints
from utils.timer import Timer
from utils.eval import evaluate_result
from utils.label_id import label_id_to_parts


FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def write_video(frames, outfpath):
    # Following instructions from
    # https://github.com/ContinuumIO/anaconda-issues/issues/223#issuecomment-285523938  # noQA
    # Define the codec and create VideoWriter object
    misc.mkdir_p(osp.dirname(outfpath))
    if 0:  # Tried writing video with opencv, clearly didn't work
        fourcc = cv2.VideoWriter_fourcc(
            chr(ord('X')), chr(ord('V')), chr(ord('I')), chr(ord('D')))
        out = cv2.VideoWriter(outfpath, fourcc, 20.0,
                              frames.shape[-3:-1][::-1])
        for i in range(frames.shape[0]):
            out.write(frames[i].astype(np.uint8))
        out.release()
        logger.warning('Writen to %s', outfpath)
    else:
        frame_dir = outfpath + '_frames'
        misc.mkdir_p(frame_dir)
        for i in range(frames.shape[0]):
            cv2.imwrite(osp.join(frame_dir, '{0:06d}.jpg'.format(i)),
                        frames[i].astype(np.uint8))
        # convert to video and delete the frames
        misc.run_cmd('ffmpeg -loglevel panic -i {0}/%06d.jpg -crf {2} {1}'
                     .format(frame_dir, outfpath, 24/cfg.TEST.SAMPLE_RATE))
        misc.run_cmd('rm -r {}'.format(frame_dir))


def gen_store_vis(frames, fc7_feats, outfpath):
    # Convert to BxTxHxWxC, and BGR
    frames = frames.transpose((1, 2, 3, 0))[..., ::-1]
    fc7 = fc7_feats.transpose((1, 2, 3, 0))
    fc7_norms = np.linalg.norm(fc7, axis=-1, keepdims=True)
    zoom_ratio = np.array(frames.shape) // np.array(fc7_norms.shape)
    zoom_ratio[-1] = 1.0
    fc7_norms_zoomed = scipy.ndimage.zoom(fc7_norms, zoom_ratio, order=1)
    fc7_norms_zoomed /= fc7_norms_zoomed.max()
    fc7_norms_zoomed *= 255.0
    res = frames * 0.5
    res[..., -1] += fc7_norms_zoomed[..., 0] * 0.5
    res = res.astype(np.uint8)
    write_video(res, outfpath + '.mp4')


def test_net_one_section(full_label_fname=None, store_vis=False):
    """
    To save test-time memory, we perform multi-clip test in multiple
    "sections":
    e.g., 10-clip test can be done in 2 sections of 5-clip test
    Args:
        full_label_id: If set uses this LMDB file, and assumes the full labels
            are being provided
        store_vis: Store visualization of what the model learned, CAM
            style stuff
    """
    timer = Timer()
    results = []
    seen_inds = defaultdict(int)

    logger.warning('Testing started...')  # for monitoring cluster jobs
    test_model = model_builder_video.ModelBuilder(
        name='{}_test'.format(cfg.MODEL.MODEL_NAME), train=False,
        use_cudnn=True, cudnn_exhaustive_search=True,
        split=cfg.TEST.DATA_TYPE,
        split_dir_name=(
            full_label_fname if full_label_fname is not None
            else cfg.TEST.DATA_TYPE))

    test_model.build_model()

    if cfg.PROF_DAG:
        test_model.net.Proto().type = 'prof_dag'
    else:
        test_model.net.Proto().type = 'dag'

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    misc.save_net_proto(test_model.net)
    misc.save_net_proto(test_model.param_init_net)

    total_test_net_iters = int(math.ceil(float(
        cfg.TEST.DATASET_SIZE * cfg.TEST.NUM_TEST_CLIPS) /
        cfg.TEST.BATCH_SIZE))

    if cfg.TEST.PARAMS_FILE:
        checkpoints.load_model_from_params_file_for_test(
            test_model, cfg.TEST.PARAMS_FILE)
    else:
        cfg.TEST.PARAMS_FILE = checkpoints.get_checkpoint_resume_file()
        checkpoints.load_model_from_params_file_for_test(
            test_model, cfg.TEST.PARAMS_FILE)
        logging.info('No params file specified for testing but found the last '
                     'trained one {}'.format(cfg.TEST.PARAMS_FILE))
        # raise Exception('No params files specified for testing model.')

    for test_iter in range(total_test_net_iters):
        timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        timer.toc()

        if test_iter == 0:
            misc.print_net(test_model)
            os.system('nvidia-smi')

        test_debug = False
        if test_debug is True:
            save_path = 'temp_save/'
            data_blob = workspace.FetchBlob('gpu_0/data')
            label_blob = workspace.FetchBlob('gpu_0/labels')
            print(label_blob)
            data_blob = data_blob * cfg.MODEL.STD + cfg.MODEL.MEAN
            for i in range(data_blob.shape[0]):
                for j in range(4):
                    temp_img = data_blob[i, :, j, :, :]
                    temp_img = temp_img.transpose([1, 2, 0])
                    temp_img = temp_img.astype(np.uint8)
                    fname = save_path + 'ori_' + str(test_iter) \
                        + '_' + str(i) + '_' + str(j) + '.jpg'
                    cv2.imwrite(fname, temp_img)

        """
        When testing, we assume all samples in the same gpu are of the same id.
        ^ This comment is from the original code. Anyway not sure why it should
        be the case.. we are extracting out the labels for each element of the
        batch anyway... Where is this assumption being used?
        ^ Checked with Xiaolong, ignore this.
        """
        video_ids_list = []  # for logging
        for gpu_id in range(cfg.NUM_GPUS):
            prefix = 'gpu_{}/'.format(gpu_id)

            # Note that this is called softmax_gpu, but could also be
            # sigmoid.
            softmax_gpu = workspace.FetchBlob(prefix + 'activation')
            softmax_gpu = softmax_gpu.reshape((softmax_gpu.shape[0], -1))
            # Mean the fc7 over time and space, to get a compact feature
            # This has already been passed through AvgPool op, but might not
            # have averaged all the way
            fc7 = np.mean(workspace.FetchBlob(
                prefix + 'fc7'), axis=(-1, -2, -3))
            # IMP! The label blob at test time contains the "index" to the
            # video, and not the video class. This is how the lmdb gen scripts
            # are set up. @xiaolonw needs it to get predictions for each video
            # and then re-reads the label file to get the actual class labels
            # to compute the test accuracy.
            video_id_gpu = workspace.FetchBlob(prefix + 'labels')
            temporal_crop_id = [None] * len(video_id_gpu)
            spatial_crop_id = [None] * len(video_id_gpu)
            if full_label_fname is not None:
                video_id_gpu, temporal_crop_id, spatial_crop_id = (
                    label_id_to_parts(video_id_gpu))

            for i in range(len(video_id_gpu)):
                seen_inds[video_id_gpu[i]] += 1

            video_ids_list.append(video_id_gpu[0])
            # print(video_id_gpu)

            if store_vis:
                save_dir = osp.join(
                    cfg.CHECKPOINT.DIR, 'vis_{}'.format(full_label_fname))
                data_blob = workspace.FetchBlob(prefix + 'data')
                label_blob = workspace.FetchBlob(prefix + 'labels')
                fc7_full = workspace.FetchBlob(prefix + 'fc7_beforeAvg')
                data_blob = data_blob * cfg.MODEL.STD + cfg.MODEL.MEAN
                for i in range(data_blob.shape[0]):
                    if temporal_crop_id[i] != 0 or spatial_crop_id[i] != 1:
                        # Only visualizing the first center clip
                        continue
                    gen_store_vis(
                        frames=data_blob[i],
                        fc7_feats=fc7_full[i],
                        outfpath=osp.join(save_dir, str(video_id_gpu[i]))
                    )

            # collect results
            for i in range(softmax_gpu.shape[0]):
                probs = softmax_gpu[i].tolist()
                vid = video_id_gpu[i]
                if seen_inds[vid] > cfg.TEST.NUM_TEST_CLIPS:
                    logger.warning('Video id {} have been seen. Skip.'.format(
                        vid,))
                    continue

                save_pairs = [vid, probs, temporal_crop_id[i],
                              spatial_crop_id[i], fc7[i]]
                results.append(save_pairs)

        # ---- log
        eta = timer.average_time * (total_test_net_iters - test_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta)))
        logger.info(('{}/{} iter ({}/{} videos):' +
                    ' Time: {:.3f} (ETA: {}). ID: {}').format(
                        test_iter, total_test_net_iters,
                        len(seen_inds), cfg.TEST.DATASET_SIZE,
                        timer.diff, eta,
                        video_ids_list,))

    return results


def test_net(full_label_fname=None, store_vis=False):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)

    cfg.TEST.DATA_TYPE = 'test'
    if cfg.TEST.TEST_FULLY_CONV is True:
        cfg.TRAIN.CROP_SIZE = cfg.TRAIN.JITTER_SCALES[0]
        cfg.TEST.USE_MULTI_CROP = 1
    elif cfg.TEST.TEST_FULLY_CONV_FLIP is True:
        cfg.TRAIN.CROP_SIZE = cfg.TRAIN.JITTER_SCALES[0]
        cfg.TEST.USE_MULTI_CROP = 2
    else:
        cfg.TRAIN.CROP_SIZE = 224

    # ------------------------------------------------------------------------
    logger.info('Setting test crop_size to: {}'.format(
        cfg.TRAIN.CROP_SIZE))

    print_cfg()
    # ------------------------------------------------------------------------

    results = []
    workspace.ResetWorkspace()  # for memory
    logger.info("Done ResetWorkspace...")

    # save temporary file
    fname = 'results_probs.pkl'
    if full_label_fname is not None:
        fname = 'results_probs_{}.pkl'.format(full_label_fname)
        # Set the dataset size and GT path based on the full lbl fname
        if full_label_fname == 'test_fullLbl':
            cfg.FILENAME_GT = osp.join(
                osp.dirname(cfg.FILENAME_GT), 'val.txt')
        elif full_label_fname == 'train_fullLbl':
            cfg.FILENAME_GT = osp.join(
                osp.dirname(cfg.FILENAME_GT), 'train.txt')
        else:
            raise NotImplementedError('Unknown full label fname {}'.format(
                full_label_fname))
        cfg.TEST.DATASET_SIZE = count_lines(cfg.FILENAME_GT)
    pkl_path = os.path.join(cfg.CHECKPOINT.DIR, fname)

    if os.path.exists(pkl_path) and not cfg.TEST.FORCE_RECOMPUTE_RESULTS:
        logger.warning('READING PRE-COMPUTED RESULTS! Delete the {} file '
                       'or set TEST.FORCE_RECOMPUTE_RESULTS True '
                       'to recompute the test results'.format(pkl_path))
        with open(pkl_path, 'r') as fin:
            results = pickle.load(fin)
    else:
        results = test_net_one_section(full_label_fname=full_label_fname,
                                       store_vis=store_vis)

    with open(pkl_path, 'w') as f:
        pickle.dump(results, f)
    logger.info('Temporary file saved to: {}'.format(pkl_path))

    # evaluate
    if cfg.FILENAME_GT is not None:
        logger.info('Overall perf (full label: %s): %s', full_label_fname,
                    evaluate_result(results))
        logger.info('=========================================')
        logger.info('=============Random Baseline=============')
        logger.info('=========================================')
        N_RUNS = 10
        rand_res = []
        for _ in range(N_RUNS):
            rand_res.append(evaluate_result(
                [(el[0], np.random.random(len(el[1]),))
                 for el in results]))
        logger.info('Emperical random baseline, averaged over {} runs'.format(
            N_RUNS))
        for metric in rand_res[0].keys():
            logger.info('{}: {}'.format(
                metric, np.mean([el[metric] for el in rand_res])))


def main():
    parser = argparse.ArgumentParser(
        description='Classification model testing')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
    parser.add_argument('--store_vis', type=bool,
                        default=False,  # Just set here when running...
                        help='Store a CAM style visualization.')
    parser.add_argument('opts', help='see configs.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)

    assert_and_infer_cfg()

    # run testing for both
    # Only storing full fc7 features for test because that's the only one
    # we will analyze
    test_net(full_label_fname='test_fullLbl',
             store_vis=args.store_vis)
    test_net(full_label_fname='train_fullLbl')


if __name__ == '__main__':
    main()
