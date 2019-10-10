# Run with caffe2_Nov_2018 conda env

import glob
import os.path as osp
import json
from gen_utils import mkdir_p
import numpy as np
from tqdm import tqdm
import logging
import subprocess
from functools import partial
import re
import cPickle as pkl
from collections import defaultdict, OrderedDict
from itertools import permutations, product
import multiprocessing as mp
import math
import copy

OUTPUT_DATA_DIR = 'output'  # Change path to where the videos are stored
LIST_DIR = 'lists'
USE_TRAIN_TEST_SPLIT_FROM = None
MAX_TOT_VIDEOS = 5500
np.random.seed(42)
NUM_ROWS = 3
NUM_COLS = 3
ACTION_CLASSES = [
    # object, movement
    ('sphere', '_slide'),
    ('sphere', '_pick_place'),
    ('spl', '_slide'),
    ('spl', '_pick_place'),
    ('spl', '_rotate'),
    ('cylinder', '_pick_place'),
    ('cylinder', '_slide'),
    ('cylinder', '_rotate'),
    ('cube', '_slide'),
    ('cube', '_pick_place'),
    ('cube', '_rotate'),
    ('cone', '_contain'),
    ('cone', '_pick_place'),
    ('cone', '_slide'),
]
_BEFORE = 'before'
_AFTER = 'after'
_DURING = 'during'
ORDERING = [
    _BEFORE,
    _DURING,
    _AFTER,
]


def localize_dataset(data, num_rows=NUM_ROWS, num_cols=NUM_COLS):
    """ data is a dictionary with video name to metadata (JSON file). """
    # NOTE that this code is now also being used by proj_utils to go from
    # 2D coordinates on plane to class ID. So change that too if this gets
    # changed.
    fnames = []
    lbls = []
    for fname, metadata in data.items():
        fnames.append(fname)
        objects = metadata['objects']
        object = [el for el in objects if el['shape'] == 'spl'][0]
        pos = object['locations'][str(len(object['locations']) - 1)]
        if num_rows != NUM_ROWS or num_cols != NUM_COLS:
            # In this case, need to scale the pos values to scale to the new num_rows etc
            pos[0] *= num_cols * 1.0 / NUM_COLS
            pos[1] *= num_rows * 1.0 / NUM_ROWS
        # Without math.floor it would screw up on negative axis
        x, y = (int(math.floor(pos[0])) + num_cols,
                int(math.floor(pos[1])) + num_rows)
        cls_id = y * (2 * num_cols) + x
        lbls.append(cls_id)
    return fnames, lbls, {'classes': range(num_cols * num_rows * 4)}


def actions_or_not_dataset(data, action_classes):
    fnames = []
    lbls = []
    for fname, metadata in data.items():
        fnames.append(fname)
        movements = metadata['movements']
        objects = metadata['objects']
        name_to_type = {el['instance']: el['shape'] for el in objects}
        shape_to_actions = defaultdict(lambda: [])
        for name, motions in movements.items():
            shape_to_actions[name_to_type[name]] += [
                el[0] for el in motions]
        this_lbl = []
        # iterate over the classes and check if any of that is true for this
        # case
        for action_id, (shape, movement) in enumerate(action_classes):
            if movement in shape_to_actions[shape]:
                this_lbl.append(action_id)
        lbls.append(','.join([str(el) for el in this_lbl]))
    return fnames, lbls, {'classes': action_classes}


def get_ordering(act1_time, act2_time):
    if act1_time[1] <= act2_time[0]:
        # act1 should finish before act2 starts
        return _BEFORE
    elif act2_time[1] <= act1_time[0]:
        # act1 should start after act2 ends
        return _AFTER
    else:
        # We define everything else as "during", though it might be partly
        # overlapped etc
        return _DURING


def satisfy_action_class(action_class, actions_set):
    action_class_ents, action_class_ord = action_class
    assert len(action_class_ents) == len(actions_set), \
        'Must be same number of actions'
    # The action set must contain the exact objects in exact motion
    for action_class_ent, action in zip(action_class_ents, actions_set):
        if (action_class_ent[0] != action[0] or
                action_class_ent[1] != action[1][0]):
            return False
    # Now need to make sure the intervals make sense w.r.t the relation
    for i in range(len(action_class_ord)):
        if not get_ordering(
                actions_set[i][1][2:],
                actions_set[i + 1][1][2:]) == action_class_ord[i]:
            return False
    return True


def compute_active_labels(data_el, classes, n):
    fname, metadata = data_el
    movements = metadata['movements']
    objects = metadata['objects']
    name_to_type = {el['instance']: el['shape'] for el in objects}
    all_actions = []
    for name, motions in movements.items():
        for motion in motions:
            all_actions.append((
                name_to_type[name],
                motion))
    # Consider all n-length permutations of all_actions, and check if it
    # fits any of the classes
    this_lbl = set()
    for (cls_id, action_class), actions_set in product(
            enumerate(classes), permutations(all_actions, n)):
        if satisfy_action_class(action_class, actions_set):
            this_lbl.add(cls_id)
    return fname, this_lbl


def action_order_unique(classes):
    def reverse(el):
        if el == ('during',):
            return el
        elif el == ('before',):
            return ('after',)
        elif el == ('after',):
            return ('before',)
        else:
            raise ValueError('This should not happen')
    classes_uniq = []
    for el in classes:
        if el not in classes_uniq and ((el[0][1], el[0][0]), reverse(el[1])) not in classes_uniq:
            classes_uniq.append(el)
    return classes_uniq


def actions_order_dataset(data, n=2, unique=False):
    # NOTE: When an object is contained, and the containing object slides,
    # I consider that as a slide for the contained object as well.
    fnames = []
    lbls = []
    action_sets = list(product(ACTION_CLASSES, repeat=n))
    # all orderings
    orderings = list(product(ORDERING, repeat=(n-1)))
    # all actions and orderings
    classes = list(product(action_sets, orderings))
    if unique:
        # Remove classes such as "X before Y" when "Y after X" already exists in the data
        classes = action_order_unique(classes)
    print('Action orders classes {}'.format(len(classes)))
    # Now check for all combinations in the video and check if it fits any of
    # the classes
    num_labels_active = 0
    # Compute the labels in parallel, since it is pretty slow
    pool = mp.Pool(16)
    all_labels = list(tqdm(
        pool.imap(partial(compute_active_labels, classes=classes, n=n),
                  data.items()),
        total=len(data), desc='Computing action order labels'))
    pool.terminate()
    for fname, this_lbl in all_labels:
        fnames.append(fname)
        num_labels_active += len(this_lbl)
        lbls.append(','.join([str(el) for el in sorted(list(this_lbl))]))
    num_labels_active /= len(data)
    print('Found {} active labels avg out of {} classes'.format(
        num_labels_active, len(classes)))
    return fnames, lbls, {'classes': classes}


def write_to_file(vid_lbl, fname):
    with open(fname, 'w') as fout:
        for vname, lbl in vid_lbl:
            fout.write('{} {}\n'.format(vname, lbl))


def check_avi_broken(fpath):
    """ Check if the AVI file is broken, i.e. does not have index. This
    indicates a video that was not fully rendered and must be ignored for the
    final training/testing. """
    if osp.exists(fpath + '.lock'):
        # For any properly rendered video, the lock file must be deleted.
        return True
    # TODO(rgirdhar): Also might want to check for empty, overly small files?
    # So far the AVI without index is able to catch most bad/partially rendered
    # files.
    try:
        output = subprocess.check_output(
            'ffmpeg -i {}'.format(fpath), shell=True, stderr=subprocess.STDOUT,
            universal_newlines=True)
    except subprocess.CalledProcessError as exc:
        output = exc.output
    prog = re.compile('.*AVI without index.*', flags=re.DOTALL)
    if prog.match(output):
        return True
    return False


def read_data(scene_files):
    data = {}
    for scene_file in tqdm(scene_files, desc='Reading metadata'):
        # TODO(rgirdhar): Make sure to only keep the videos that are readable
        try:
            with open(scene_file, 'r') as fin:
                metadata = json.load(fin)
            vid_name = osp.splitext(scene_file.replace(
                '/scenes/', '/images/'))[0] + '.avi'
            # Ignore videos that were not rendered correctly
            if check_avi_broken(vid_name):
                continue
            data[vid_name] = metadata
            if len(data) > MAX_TOT_VIDEOS:
                # Since we don't make a list of others, might as well stop here
                break
        except Exception as e:
            logging.error('Unable to read {} due to {}'.format(
                scene_file, e))
    return data


def get_data_subset_from_filenames(split_fpath, data):
    fnames = []
    with open(split_fpath, 'r') as fin:
        for line in fin:
            fnames.append(line.split()[0])
    subset = []
    for fn in fnames:
        subset.append((fn, data[fn]))
    return subset


def sort_data_for_train_test_split(data):
    if not USE_TRAIN_TEST_SPLIT_FROM:
        print('Keeping {} of these videos'.format(MAX_TOT_VIDEOS))
        assert MAX_TOT_VIDEOS <= len(data), 'Data does not contain enough elts.'
        data = list(data.items())[:MAX_TOT_VIDEOS]
        np.random.shuffle(data)
        cut_point = int(0.7 * len(data))
        return data[:cut_point], data[cut_point:]
    # Else, use the train/val list from USE_TRAIN_TEST_SPLIT_FROM
    train_data = get_data_subset_from_filenames(osp.join(USE_TRAIN_TEST_SPLIT_FROM, 'train.txt'), data)
    val_data = get_data_subset_from_filenames(osp.join(USE_TRAIN_TEST_SPLIT_FROM, 'val.txt'), data)
    return train_data, val_data



def main():
    scene_files = glob.glob(osp.join(OUTPUT_DATA_DIR, 'scenes/*.json'))
    output_dir = osp.join(OUTPUT_DATA_DIR, LIST_DIR)
    data_cache_fpath = osp.join(OUTPUT_DATA_DIR, 'good_videos.pkl')
    if osp.exists(data_cache_fpath):
        print('Found pre-computed file of good rendered data {}'.format(
            data_cache_fpath))
        with open(data_cache_fpath, 'r') as fin:
            data = pkl.load(fin)
        print('...Read.')
    else:
        data = read_data(scene_files)
        with open(data_cache_fpath, 'w') as fout:
            pkl.dump(data, fout)
    print('Found {} good videos out of {}'.format(len(data), len(scene_files)))
    train_data, val_data = sort_data_for_train_test_split(data)
    train_data = OrderedDict(train_data)
    val_data = OrderedDict(val_data)
    dataset_gen_fns = OrderedDict([
        ('actions_order_uniq', partial(actions_order_dataset, unique=True)),
        ('localize_4x4', partial(localize_dataset, num_rows=2, num_cols=2)),
        ('localize_8x8', partial(localize_dataset, num_rows=4, num_cols=4)),
        ('localize', localize_dataset),
        ('actions_present', partial(
            actions_or_not_dataset,
            action_classes=ACTION_CLASSES)),
    ])
    for dset_name, dset_fn in dataset_gen_fns.items():
        this_output_dir = osp.join(output_dir, dset_name)
        train_txt_outfpath = osp.join(this_output_dir, 'train.txt')
        train_shuf_txt_outfpath = osp.join(
            this_output_dir, 'train_shuffle_rep.txt')
        val_txt_outfpath = osp.join(this_output_dir, 'val.txt')
        mkdir_p(this_output_dir)
        if not (osp.exists(train_shuf_txt_outfpath) and
                osp.exists(train_txt_outfpath) and
                osp.exists(val_txt_outfpath)):
            train_fnames, train_labels, train_metadata = dset_fn(copy.deepcopy(train_data))
            val_fnames, val_labels, val_metadata = dset_fn(copy.deepcopy(val_data))
            # remove datapoints that don't have a valid label
            vid_lbl_train = [(fname, label) for fname, label in
                             zip(train_fnames, train_labels) if len(str(label)) > 0]
            vid_lbl_val = [(fname, label) for fname, label in
                           zip(val_fnames, val_labels) if len(str(label)) > 0]
            # split into train and test
            train_metadata.update(val_metadata)
            with open(osp.join(this_output_dir, 'metadata.pkl'), 'w') as fout:
                pkl.dump(train_metadata, fout)
            write_to_file(vid_lbl_train, train_txt_outfpath)
            write_to_file(vid_lbl_val, val_txt_outfpath)


if __name__ == '__main__':
    main()
