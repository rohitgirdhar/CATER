import subprocess
import os
import os.path as osp
import errno
import numpy as np


# Set this to the dataset/task you need to convert to LMDB
DIR_TO_PROCESS = '/scratch/rgirdhar/release/all_actions/lists/localize/'


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


def make_lmdb(txt_fpath, lmdb_outpath, multicrop=False, full_label=False):
    """
    Args:
        multicrop (bool): Gen lmdb for 3-crop 10
        (hard-coded in the lmdb gen file)
    """
    mkdir_p(osp.dirname(lmdb_outpath))
    # remove in case it exists
    subprocess.call('rm -r {}'.format(lmdb_outpath), shell=True)
    # File
    if multicrop:
        python_fname = 'create_video_lmdb_test_multicrop.py'
    else:
        python_fname = 'create_video_lmdb.py'
    # Use xiaolong's code to gen the lmdbs
    subprocess.call(
        'source activate caffe2_Nov_2018;'
        'PYTHONPATH=`pwd`/lib/:$PYTHONPATH '
        'LD_LIBRARY_PATH=`pwd`/../caffe2/build/install/lib/:$LD_LIBRARY_PATH '
        'PYTHONPATH=`pwd`/../caffe2/build/install/:$PYTHONPATH '
        'PYTHONPATH=`pwd`/../caffe2/build/install/lib/python2.7/site-packages/:$PYTHONPATH '
        'LD_LIBRARY_PATH=/home/rgirdhar/Software/basic/python/anaconda2/envs/caffe2_Nov_2018/lib/:$LD_LIBRARY_PATH '
        'PYTHONPATH=`pwd`/external_lib/average-precision/python:$PYTHONPATH; '
        'cd process_data/kinetics/; '
        'python {prog_name} '
        '{full_label} '
        '--dataset_dir {data_dir} --list_file {txt_fpath}'.format(
            prog_name=python_fname,
            full_label='--use_full_label_id' if full_label else '',
            data_dir=lmdb_outpath, txt_fpath=txt_fpath), shell=True)


def shuf_repeat(lst, count):
    """ Xiaolong's code expects LMDBs with the train list shuffled and
    repeated, so creating that here to avoid multiple steps. """
    final_list = []
    ordering = range(len(lst))
    for _ in range(count):
        np.random.shuffle(ordering)
        final_list += [lst[i] for i in ordering]
    assert len(final_list) == count * len(lst)
    return final_list


def write_to_file(vid_lbl, fname):
    with open(fname, 'w') as fout:
        for vname, lbl in vid_lbl:
            fout.write('{} {}\n'.format(vname, lbl))


def read_from_file(fname):
    res = []
    with open(fname, 'r') as fin:
        for line in fin:
            res.append(line.split())
    return res


def main():
    train_txt_infpath = osp.join(DIR_TO_PROCESS, 'train.txt')
    train_txt_outfpath = osp.join(DIR_TO_PROCESS, 'train_fullpath.txt')
    train_shuf_txt_outfpath = osp.join(DIR_TO_PROCESS, 'train_shuf_fullpath.txt')
    val_txt_infpath = osp.join(DIR_TO_PROCESS, 'val.txt')
    val_txt_outfpath = osp.join(DIR_TO_PROCESS, 'val_fullpath.txt')
    # Add the full path
    vid_lbl_train = read_from_file(train_txt_infpath)
    vid_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(DIR_TO_PROCESS))), 'videos')
    write_to_file([(osp.join(vid_dir, el[0]), el[1]) for 
                   el in vid_lbl_train], train_txt_outfpath)
    write_to_file([(osp.join(vid_dir, el[0]), el[1]) for 
                   el in shuf_repeat(vid_lbl_train, 100)], train_shuf_txt_outfpath)
    write_to_file([(osp.join(vid_dir, el[0]), el[1]) for 
                   el in read_from_file(val_txt_infpath)], val_txt_outfpath)
    # Write out the LMDBs
    mkdir_p(osp.join(DIR_TO_PROCESS, 'lmdb'))
    make_lmdb(train_shuf_txt_outfpath, osp.join(DIR_TO_PROCESS, 'lmdb/train'))
    make_lmdb(val_txt_outfpath, osp.join(DIR_TO_PROCESS, 'lmdb/val'))
    # The same file, but stored in the test-style lmdb format
    make_lmdb(val_txt_outfpath, osp.join(DIR_TO_PROCESS, 'lmdb/test'),
                multicrop=True)
    make_lmdb(val_txt_outfpath,
                osp.join(DIR_TO_PROCESS, 'lmdb/test_fullLbl'),
                multicrop=True, full_label=True)
    # This is for training the LSTM
    make_lmdb(train_txt_outfpath,
                osp.join(DIR_TO_PROCESS, 'lmdb/train_fullLbl'),
                multicrop=True, full_label=True)


if __name__ == '__main__':
    main()