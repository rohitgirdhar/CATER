import json
import numpy as np
import sys; sys.path.append('DaSiamRPN/code/')  # noQA
import cv2
import torch
from tqdm import tqdm
import torch.multiprocessing as mp
import os.path as osp
import pickle as pkl

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track

import proj_utils

# Common localization evaluation codes
loc_eval_path = osp.join(osp.dirname(osp.realpath(__file__)), '../video-nonlocal-net/lib/utils/')  # noQA
sys.path.insert(0, loc_eval_path)
from localization_eval import all_localization_accuracies


DEBUG_STORE_VIDEO = False
DEBUG_STORE_VIDEO_PATH = '/tmp/tracker/frames/'
RESULTS_DIR = '/tmp/tracker'

model_wts = torch.load('DaSiamRPN/SiamRPNVOT.model')
val_fpath = '/scratch/rgirdhar/release/all_actions/lists/localize/val.txt'
vid_dir = '/scratch/rgirdhar/release/all_actions/videos/'
NROWS = 3
NCOLS = 3


def one_hot(lst, num_classes):
    res = np.zeros((len(lst), num_classes))
    for i, el in enumerate(lst):
        if el >= num_classes:
            print('Found class {} outside num_classes {}'.format(el, num_classes))
        else:
            res[i, el] = 1.0
    return res


def get_snitch_start_end_coord(vid_path):
    scene_path = vid_path.replace('/videos/', '/scenes/').replace(
        '.avi', '.json')
    with open(scene_path) as f:
        data = json.load(f)
    spl = [el for el in data['objects'] if el['shape'] == 'spl'][0]
    start_loc = spl['locations']['0']
    end_loc = max([(int(key), val) for key, val
                   in spl['locations'].items()])[-1]
    return start_loc, end_loc


def un_normalize_img_coords(cx, cy, w, h):
    """From the -1 1 to w h repr."""
    cx = int((cx + 1) * w / 2)
    cy = int((cy + 1) * h / 2)
    return cx, cy


def track_and_predict(vid_path, net):
    """
    Returns:
        pred: Tracks the snitch, and return the class for the final
            coordinates of the snitch
        pred_gt: Uses the end point location of the snitch, projects to 2D,
            then computes the class coordinates. This is only returned as a
            sanity check, to make sure the projection etc functions are
            working reasonably -- i.e. using this output should get 100%
            accuracy.
    """
    start_coord, end_coord = get_snitch_start_end_coord(vid_path)
    start_coord_2d = proj_utils.project_3d_point(np.array([start_coord]))[0]
    end_coord_2d = proj_utils.project_3d_point(np.array([end_coord]))[0]
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise 'Unable to open video {}'.format(vid_path)
    frame_id = 0
    state = None  # State to be used by the tracker, will be init in the loop
    pred = None
    vid_out = None
    flag, frame = cap.read()
    out_vid_path = osp.join(DEBUG_STORE_VIDEO_PATH, osp.basename(vid_path))
    while flag:
        frame_id += 1
        if frame_id == 1:
            # tracker init
            h_frame, w_frame, _ = frame.shape
            cx, cy = start_coord_2d.tolist()
            cx, cy = un_normalize_img_coords(cx, cy, w_frame, h_frame)
            target_pos, target_sz = np.array([cx, cy]), np.array([30, 30])
            state = SiamRPN_init(frame, target_pos, target_sz, net)
            if DEBUG_STORE_VIDEO:
                vid_out = cv2.VideoWriter(
                    out_vid_path,
                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                    (w_frame, h_frame))
        else:
            state = SiamRPN_track(state, frame)  # track
            if DEBUG_STORE_VIDEO:
                cx, cy = state['target_pos']
                w_box, h_box = state['target_sz']
                cv2.rectangle(
                    frame,
                    (int(cx - w_box/2), int(cy - h_box/2)),
                    (int(cx + w_box/2), int(cy + h_box/2)), (0, 255, 255), 3)
                # Drawing the GT rectangle as well
                cx_gt, cy_gt = un_normalize_img_coords(
                    end_coord_2d[0], end_coord_2d[1], w_frame, h_frame)
                cv2.rectangle(
                    frame,
                    (int(cx_gt - 10), int(cy_gt - 10)),
                    (int(cx_gt + 10), int(cy_gt + 10)), (255, 0, 0), 3)
                vid_out.write(frame)
        flag, frame = cap.read()
    # Converting back to -1 1 coordinates, as that will be used to project
    # back to the CATER plane
    cx, cy = state['target_pos']
    pred = proj_utils.get_class_prediction(
        (cx * 2 / w_frame - 1), (cy * 2 / h_frame - 1),
        nrows=NROWS, ncols=NCOLS)
    pred_gt = proj_utils.get_class_prediction(
        end_coord_2d[0], end_coord_2d[1],
        nrows=NROWS, ncols=NCOLS)
    cap.release()
    if vid_out is not None:
        print('Written debugging video to {}'.format(out_vid_path))
        vid_out.release()
    return pred, pred_gt


def compute_preds(vid_line):
    # Most nodes has 4 GPUs, run on a random one
    torch.cuda.set_device(np.random.randint(4))
    # load net  <-- have to do for each as I'm running in mp
    net = SiamRPNvot()
    net.load_state_dict(model_wts)
    net.eval().cuda()
    vid_fpath, lbl = vid_line.split()
    pred, pred_gt = track_and_predict(vid_fpath, net)
    return pred, pred_gt, int(lbl), vid_fpath


def store_failure_cases(vid_fpaths, preds, lbls):
    with open(osp.join(
            DEBUG_STORE_VIDEO_PATH, 'failure_cases.txt'), 'w') as fout:
        for vid_fpath, pred, lbl in zip(vid_fpaths, preds, lbls):
            fout.write('{} {} {}\n'.format(vid_fpath.strip(), lbl, pred))


def store_preds(preds):
    N_CLASSES = 4 * NROWS * NCOLS
    res = np.zeros((len(preds), N_CLASSES))
    for i, pred in enumerate(preds):
        if pred >= 0 and pred < N_CLASSES:
            res[i, pred] = 1
        else:
            print('Found a prediction outside the class boundaries')
            # Usually due to the homography putting it slightly outside..
            # ignore.. hopefully not too many
            pass
    final = list(zip(range(len(preds)), res))
    with open(osp.join(RESULTS_DIR, 'results_probs.pkl'), 'wb') as fout:
        pkl.dump(final, fout, protocol=1)


def main():
    mp.set_start_method('spawn', force=True)
    with open(val_fpath, 'r') as fin:
        vid_list = [osp.join(vid_dir, el.strip()) for el in fin.readlines()]
    # vid_list = vid_list[:10]  # DEBUG!!!!
    pool = mp.Pool(processes=8)
    all_preds = list(tqdm(pool.imap(compute_preds, vid_list),
                          desc='Evaluating', total=len(vid_list)))
    pool.close()
    pool.join()
    preds, preds_gt, lbls, vid_fpaths = zip(*all_preds)
    nclasses = max(lbls) + 1  # lbls are 0 indexed
    preds_1hot = one_hot(preds, nclasses)
    preds_gt_1hot = one_hot(preds_gt, nclasses)
    lbls = np.array(lbls)
    # print(preds, preds_gt, lbls)
    corr = (np.array(preds) == np.array(lbls))
    all_acc = all_localization_accuracies(lbls, preds_1hot)
    assert np.isclose(all_acc['top_1'], np.mean(corr))
    print('Tracking baseline accuracies: {}'.format(all_acc))
    store_preds(preds)
    store_failure_cases(vid_fpaths, preds, lbls)
    gt_acc = np.mean(np.array(np.array(preds_gt) == np.array(lbls)))
    all_acc_gt = all_localization_accuracies(lbls, preds_gt_1hot)
    assert all_acc_gt['top_1'] > 0.99
    assert np.isclose(all_acc_gt['top_1'], gt_acc)
    print('Tracking baseline accuracy (GT end loc, this should be 1.0): {}'.format(all_acc_gt))


if __name__ == '__main__':
    main()
