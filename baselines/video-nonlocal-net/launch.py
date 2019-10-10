##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import subprocess
import os
import errno
import re
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='Launch a config')
    parser.add_argument(
        '--cfg', '-c', dest='cfg_file', required=True,
        help='Config file to run')
    parser.add_argument(
        '--tool', '-t', dest='tool', default='train',
        help='Tool to run')
    parser.add_argument(
        '--gpus', '-g', default=None, type=str,
        help='GPUs to run on. By default, use all.')
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def mkdir_p(path):
    """
    Make all directories in `path`. Ignore errors if a directory exists.
    Equivalent to `mkdir -p` in the command line, hence the name.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _run_lstm_cmd(cfg_file, cuda_vis):
    with open(cfg_file, 'r') as fin:
        txt = fin.read()
        matches = re.findall(r'FILENAME_GT: (.+)', txt)
        lbl_dir = osp.dirname(matches[0])
        matches = re.findall(r'MULTILABEL: (\w+).*', txt)
        if matches and matches[-1].strip() == 'True':
            multilabel = True
        else:
            multilabel = False
        matches = re.findall(r'NUM_CLASSES: (\d+).*', txt)
        num_classes = int(matches[-1])
    cmd = [
        'source activate pytorch0.4;',
        '{}'.format(cuda_vis),
        'python ../lstm/main.py ',
        '--data-dir outputs/{}'.format(cfg_file),  # noQA
        '--lbl-dir {}'.format(lbl_dir),
        '--nclasses {}'.format(num_classes),
        '--multilabel' if multilabel else '',
    ]
    cmd = ' '.join(cmd)
    print('running {}'.format(cmd))
    subprocess.call(cmd, shell=True)


def _run_cmd(cfg_file, other_opts, output_dir, tool_name, tool_path, cuda_vis):
    if tool_name == 'lstm':
        return _run_lstm_cmd(cfg_file, cuda_vis)
    cmd = '''
        source activate {env}; \
        {cuda_vis} \
        PYTHONPATH=`pwd`/lib/:$PYTHONPATH \
        LD_LIBRARY_PATH=`pwd`/../caffe2/build/install/lib/:$LD_LIBRARY_PATH \
        PYTHONPATH=`pwd`/../caffe2/build/install/:$PYTHONPATH \
        PYTHONPATH=`pwd`/../caffe2/build/install/lib/python2.7/site-packages/:$PYTHONPATH \
        LD_LIBRARY_PATH=/home/rgirdhar/Software/basic/python/anaconda2/envs/caffe2_Nov_2018/lib/:$LD_LIBRARY_PATH \
        PYTHONPATH=`pwd`/external_lib/average-precision/python:$PYTHONPATH \
        python tools/{tool_path} \
        --config_file {cfg_file} \
        {other_opts} 2>&1 | tee {output_dir}/log_{tool_path}.txt
        '''.format(env=('caffe2_Nov_2018_vis' if tool_name == 'vis' else
                        'caffe2_Nov_2018'),
                   cuda_vis=cuda_vis,
                   cfg_file=cfg_file,
                   tool_path=tool_path,
                   output_dir=output_dir,
                   other_opts=other_opts)
    print('Running {}'.format(cmd))
    subprocess.call(cmd, shell=True)


def main():
    args = parse_args()
    other_opts = ''
    output_dir = 'outputs/{}'.format(args.cfg_file)
    other_opts += 'CHECKPOINT.DIR {} '.format(output_dir)
    mkdir_p(output_dir)
    if args.opts is not None:
        other_opts += ' '.join(args.opts) + ' '
    if args.tool == 'train':
        tool_name = 'train_net_video.py'
    elif args.tool == 'test':
        tool_name = 'test_net_video.py'
        other_opts += 'TEST.TEST_FULLY_CONV True '
    elif args.tool == 'vis':
        tool_name = 'vis_results.py'
    elif args.tool == 'eval':
        tool_name = 'eval.py'
    elif args.tool == 'lstm':
        tool_name = 'lstm'
    else:
        raise NotImplementedError('Unknown tool name {}'.format(args.tool))

    cuda_vis = ''
    if args.gpus is not None:
        print('Running with fewer GPUs. Might need to reduce batch size.')
        other_opts += 'NUM_GPUS {} '.format(len(args.gpus.split(',')))
        cuda_vis = 'CUDA_VISIBLE_DEVICES="{}"'.format(args.gpus)
    _run_cmd(args.cfg_file, other_opts, output_dir, args.tool, tool_name,
             cuda_vis)


if __name__ == '__main__':
    main()
