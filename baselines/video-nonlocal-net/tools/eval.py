from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np
import argparse
import sys
import pickle
import os

from core.config import config as cfg
from core.config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg)

from utils.eval import evaluate_result


FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def test_net():
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

    # save temporary file
    pkl_path = os.path.join(cfg.CHECKPOINT.DIR, "results_probs.pkl")
    assert os.path.exists(pkl_path)
    with open(pkl_path, 'r') as fin:
        results = pickle.load(fin)

    # evaluate
    if cfg.FILENAME_GT is not None:
        final_res = evaluate_result(results)
        for metric in final_res.keys():
            if metric != 'per_class_ap':  # Too many things in this one
                print('{}: {}'.format(metric, final_res[metric]))
        pkl_out_path = os.path.join(cfg.CHECKPOINT.DIR , 'results_scores.pkl')
        print('Storing scores in {}'.format(pkl_out_path))
        with open(pkl_out_path, 'w') as fout:
            pickle.dump(final_res, fout)
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
            if metric != 'per_class_ap':  # Too many things in this one
                print('{}: {}'.format(
                    metric, np.mean([el[metric] for el in rand_res])))


def main():
    parser = argparse.ArgumentParser(
        description='Classification model testing')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
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

    test_net()


if __name__ == '__main__':
    main()
