# Typically run using
# python launch.py -c configs_clevr/019_NIPS18/008_I3D_NL_localize_imagenetPretrained_32f_8SR.yaml -t vis  # noQA

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import os.path as osp
import matplotlib; matplotlib.use('Agg')  # noQA
from mpl_toolkits.mplot3d import Axes3D  # noQA  # Needed for 3D projections
import numpy as np
import logging
import argparse
import sys
import cPickle as pkl
import cv2
import json
import subprocess
from moviepy.editor import (
    VideoFileClip, clips_array, CompositeVideoClip, TextClip,
    ImageSequenceClip, concatenate_videoclips, ImageClip)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.eval import evaluate_result, get_labels_and_prob
from core.config import config as cfg
from core.config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg)

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

GRID_SIZE = 3
OUT_WD = 320
OUT_HT = 240
MOVIEPY_FONT = 'Utopia'
PLT_EXT = '.pdf'

# For dark mode
# PLT_EXT = '.svg'
# plt.style.use('dark_background')


def chunkify(lst, n):
    return [lst[i::n] for i in xrange(n)]


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def add_snitch(frame, loc, color=(255, 215, 0)):
    cv2.circle(frame,
               (int((loc[0] + GRID_SIZE) * OUT_WD / (2 * GRID_SIZE)),
                int((GRID_SIZE - loc[1]) * OUT_HT / (2 * GRID_SIZE))),
               5, color, 3)


def mkdir_p(dpath):
    subprocess.call('mkdir -p {}'.format(dpath), shell=True)


def location_predictions_to_map(prob, video_meta):
    """ Probability values for each location. """
    # This is how the matrix classes are defined finally
    # 30, 31, 32, 33, 34, 35
    # ...
    # 0, 1, 2, 3, 4, 5
    prob = np.flipud(prob.reshape((2 * GRID_SIZE, -1, 1)) * 256.0).astype(
        np.uint8)
    # Lighter version of the prob map
    prob_cmap = 0.7 * cv2.applyColorMap(prob, cv2.COLORMAP_BONE)
    prob_cmap = cv2.resize(prob_cmap, (OUT_WD, OUT_HT),
                           interpolation=cv2.INTER_NEAREST)
    # Draw X,Y axis
    cv2.arrowedLine(
        prob_cmap,
        (prob_cmap.shape[1] // 2, prob_cmap.shape[0]),
        (prob_cmap.shape[1] // 2, 0),
        (255, 0, 0), 3)
    cv2.arrowedLine(
        prob_cmap,
        (0, prob_cmap.shape[0] // 2),
        (prob_cmap.shape[1], prob_cmap.shape[0] // 2),
        (255, 0, 0), 3)
    # Draw the locations of spl object, from the video metadata
    spl_locs = [el for el in video_meta['objects'] if el['shape'] == 'spl']
    assert len(spl_locs) == 1
    spl_locs = spl_locs[0]['locations']
    nframes = len(spl_locs)
    spl_locs = [spl_locs[str(el)] for el in range(nframes)]
    all_frames = []
    for loc_i in range(len(spl_locs)):
        frame = prob_cmap.copy()
        add_snitch(frame, spl_locs[loc_i], (255, 215, 0))
        # Have a trail, make lighter circles for the previous positions
        for prev_loc_i in range(0, loc_i):
            add_snitch(frame, spl_locs[prev_loc_i],
                       ((128 + (255 - 128) * prev_loc_i / loc_i),
                        (128 + (215 - 128) * prev_loc_i / loc_i),
                        0))
        all_frames.append(frame.astype(np.uint8))
    return all_frames


def video_collage(vid_names, ordering, probs, correct_class_prob, labels):
    sel_clips = []
    for i in range(6):
        vname = vid_names[ordering[i]]
        # Load all the metadata for this video
        with open(vname.replace(
                'images/', 'scenes/').replace('.avi', '.json'), 'r') as fin:
            metadata = json.load(fin)
        gt = np.argmax(labels[ordering[i]])
        pred = np.argmax(probs[ordering[i]])
        pred_map = location_predictions_to_map(probs[ordering[i]], metadata)
        print(vname, gt, pred, correct_class_prob[ordering[i]])
        main_video = VideoFileClip(vname).margin(3)
        sel_clips.append(CompositeVideoClip([
            clips_array([[
                main_video,
                ImageSequenceClip(pred_map, fps=main_video.fps),
            ]]),
            TextClip(
                "GT {} Pred {}".format(gt, pred),
                font=MOVIEPY_FONT)
            .set_pos((10, 10))
            .set_duration(main_video.duration),
        ]))
    return clips_array(chunkify(sel_clips, 3))


def add_text_slide(text, slide_size, fontsize=40, duration_secs=3,
                   color='white', bg_color='transparent'):
    # Moviepy leads to aliased fonts, to fix need to render at high res and
    # resize down
    SCALE_UP_FONT = 3
    return TextClip(
        text,
        color=color,
        bg_color=bg_color,
        size=tuple([el * SCALE_UP_FONT for el in slide_size]),
        fontsize=fontsize * SCALE_UP_FONT,
        font=MOVIEPY_FONT).set_pos((
            'center')).set_duration(duration_secs).resize(1 / SCALE_UP_FONT)


def add_loc_task_instructions(slide_size):
    return ImageClip(
        'assets/instructions.png',
        duration=10,
    ).set_pos('center').resize(slide_size)


def last_frame_with_txt(vid, txt, duration):
    """Take the last frame from vid, show it for duration with txt overlay."""
    frame = list(vid.iter_frames())[-1]
    clip = ImageClip(frame, duration=duration)
    return CompositeVideoClip([
        clip,
        TextClip(txt, font=MOVIEPY_FONT, color='black', bg_color='white',
                 fontsize=40)
        .set_pos((10, 10)).set_duration(duration)])


def location_make_best_worst_video(labels, probs):
    correct_class_prob = []
    for label, prob in zip(labels, probs):
        correct_class_prob.append(prob[np.argmax(label)])
    # Get the video fnames
    with open(cfg.FILENAME_GT, 'r') as fin:
        vid_names = np.array([el.split()[0] for el in fin])
    ordering = np.argsort(correct_class_prob)
    # Remove an NaNs
    ordering = [el for el in ordering if not np.isnan(correct_class_prob[el])]
    worse_egs = video_collage(
        vid_names, ordering, probs, correct_class_prob, labels)
    best_egs = video_collage(
        vid_names, ordering[::-1], probs, correct_class_prob, labels)
    final_video = concatenate_videoclips([
        add_text_slide(
            '''Task 3: Localize Snitch\nHardest and easiest video analysis
            ''', best_egs.size, duration_secs=2),
        add_loc_task_instructions(best_egs.size),
        add_text_slide('Hard cases: Videos for which prediction\n '
                       'probability for correct class was lowest',
                       best_egs.size, duration_secs=5),
        worse_egs,
        last_frame_with_txt(worse_egs, 'Videos where snitch is not visible, or moves right at the end,\n are the hardest for the model.', duration=5),  # noQA
        add_text_slide('Easy cases: Videos for which prediction\n '
                       'probability for correct class was highest',
                       best_egs.size, duration_secs=5),
        best_egs,
        last_frame_with_txt(best_egs, 'Videos where snitch is visible and moves little towards the end,\n are the easiest for the model.', duration=5),  # noQA
    ], method='compose')
    return final_video, worse_egs, best_egs


def add_title_slide(slide_size):
    return add_text_slide(
        '''CATER: A diagnostic dataset for\nCompositional Actions and TEmporal Reasoning''',  # noQA
        slide_size, fontsize=50)


def random_videos_collage(lbls_file, slide_size):
    with open(lbls_file, 'r') as fin:
        all_files = [el.split()[0] for el in fin.readlines()]
    # Take the top 8 for now
    res = clips_array(chunkify(
        [VideoFileClip(el).margin(3) for el in all_files[:12]], 3)).resize(
            slide_size)
    return res


def dataset_intro(slide_size):
    return concatenate_videoclips([
        add_text_slide('Sample videos', slide_size, duration_secs=2),
        random_videos_collage('/data/rgirdhar/Data2/Projects/2018/003_RenderAction/clevr-vid/outputs/003_Fixes/lists_01/localize/train.txt', slide_size),  # noQA
        add_text_slide('Sample videos with camera motion', slide_size,
                       duration_secs=2),
        random_videos_collage('/data/rgirdhar/Data2/Projects/2018/003_RenderAction/clevr-vid/outputs/003_Fixes_withCameraMotion/lists_01/localize/train.txt', slide_size),  # noQA
    ], method='compose')


def pred_att_maps(att_dir, slide_size):
    # SEL = [784, 930, 998, 555]
    # title_slide = add_text_slide(
    #     'Where does the model look for task 3?\n '
    #     'We visualize network attention as the L2 norm of last layer features, superimposed on the video.\n '  # noQA
    #     'We find the network focuses on the snitch, especially towards the end of the clip. ',  # noQA
    #     slide_size, duration_secs=5)
    # Decided to make this one in keynote and just add it here.
    # Stored at https://drive.google.com/file/d/1jasu-TLufHlLFfOdQ16e752dXQ_tV7_u/view?usp=sharing  # noQA

    # Also adding the tracking visualization, and piggy backing on this
    # function. Stored in the same folder as the attention_vis (NIPS'18 in
    # google drive)
    tracking_vis = VideoFileClip(osp.join(osp.dirname(att_dir),
                                          'tracking_vis.m4v'))
    att_vis = VideoFileClip(osp.join(osp.dirname(att_dir), 'attention_vis.m4v'))
    return concatenate_videoclips([tracking_vis, att_vis], method='compose')


def make_supp_video(labels, probs, att_dir, outdir):
    """
    Generates the suppl video.
    Args:
        labels: The GT
        probs: The softmax predictions
        att_dir: The directory where the l2 norm attention maps are stored.
            These are generated by running the test code (like so:
            `python launch.py -c configs_clevr/019_NIPS18/008_I3D_NL_localize_imagenetPretrained_32f_8SR.yaml -t test TEST.FORCE_RECOMPUTE_RESULTS True`)  # noQA
            but before that set store_vis = True in the arguments
            Set to None to not generate this in the output.
        outdir: The directory to store the final predictions in.
    """
    loc_task_res, worst_clip, best_clip = location_make_best_worst_video(
        labels, probs)
    subclips = [
        add_title_slide(loc_task_res.size),
        dataset_intro(loc_task_res.size),
        loc_task_res,
    ]
    if att_dir is not None:
        subclips.append(pred_att_maps(att_dir, loc_task_res.size))
    final_video = concatenate_videoclips(subclips, method='compose')
    worst_clip.save_frame(osp.join(
        outdir, 'supp_worst.png'), worst_clip.duration-0.1)
    best_clip.save_frame(osp.join(
        outdir, 'supp_best.png'), best_clip.duration-0.1)
    final_video.write_videofile(osp.join(outdir, 'supp.mp4'))


def count_objects(metadata):
    return len(metadata['objects'])


def snitch_displacement(metadata):
    spl = [el for el in metadata['objects'] if el['shape'] == 'spl'][0]
    init = spl['locations']['0']
    final = spl['locations'][str(len(spl['locations']) - 1)]
    dx, dy = init[0] - final[0], init[1] - final[1]
    return np.sqrt(dx * dx + dy * dy)


def snitch_contained_time(metadata):
    # for each object, see the end frame when it contained spl, and then the
    # the first frame when it released (pick_placed). Sum over all the cones.
    # This will automatically cover for the heirarchical containment
    total_contain_time = 0
    for _, movements in metadata['movements'].items():
        contain_start = None
        for movement in movements:
            if movement[0] == '_contain' and movement[1] == 'Spl_0':
                # Should not be containing anything already
                assert contain_start is None
                contain_start = movement[-1]
            elif contain_start is not None and movement[0] == '_pick_place':
                assert movement[2] > contain_start
                total_contain_time += (movement[2] - contain_start)
                contain_start = None
    return total_contain_time


def snitch_contained_at_end(metadata, three_class=False):
    """Return 1 if snitch is contained at the end."""
    ever_contained = False
    res = None
    for _, movements in metadata['movements'].items():
        contain_start = None
        for movement in movements:
            if movement[0] == '_contain' and movement[1] == 'Spl_0':
                # Should not be containing anything already
                assert contain_start is None
                contain_start = True
                ever_contained = True
            elif contain_start is not None and movement[0] == '_pick_place':
                assert movement[2] > contain_start
                contain_start = None
        # Means this object never "uncontained" the snitch
        if contain_start is not None:
            res = 'At end'
            break
    if res is None:
        if ever_contained:
            res = 'In between'
        else:
            res = 'Never'
    if three_class:
        return res
    else:
        return 'Contained' if res == 'At end' else 'Open'


def snitch_last_contained(metadata):
    """Return the frame when snitch was last contained."""
    last_contain = 0
    for _, movements in metadata['movements'].items():
        contain_start = False
        for movement in movements:
            if movement[0] == '_contain' and movement[1] == 'Spl_0':
                # Should not be containing anything already
                contain_start = True
            elif contain_start and movement[0] == '_pick_place':
                last_contain = max(movement[-2], last_contain)
                contain_start = False
        if contain_start:
            # It never ended
            last_contain = 300  # Max frames
    return last_contain


def snitch_last_moved_frame(metadata):
    """Return the frame_id at which snitch last moved."""
    # Get position of snitch at every frame
    snitch_pos = [el for el in metadata['objects']
                  if el['instance'] == 'Spl_0'][0]['locations']
    # Reset the keys to integers.. for some reason I stored them as string
    snitch_pos = {int(key): val for key, val in snitch_pos.items()}
    frame_ids = snitch_pos.keys()
    # Last frame ID
    last_frame_id = max(frame_ids)
    # Go down descending order to see when it was last at a different location
    for frame_id in sorted(frame_ids)[::-1]:
        if not np.all(np.isclose(
                snitch_pos[frame_id], snitch_pos[last_frame_id])):
            break
    return frame_id


def plot_acc_binned(per_sample_accuracy, param, param_name, outdir, nbins=10,
                    exact_values=False, put_xlabel=False):
    """
    Args:
        per_sample_accuracy: Dict with keys = methods, and values = whether or
            not this method got that video correctly predicted.
        nbins: Set to a list to use those bins edges
        exact_values: The bin edges are exact values
    """
    for psa in per_sample_accuracy.values():
        assert len(psa) == len(param)
    # _, bin_edges = np.histogram(param, bins=nbins)
    if not isinstance(nbins, list):
        bin_edges = sorted(param)[::len(param)//(nbins+1)][1:]
        if not isinstance(bin_edges[0], int):
            bin_edges = [float('{:0.02f}'.format(el)) for el in bin_edges]
        # Remove the bin with limit 0, as it usually gets too few assigned
        bin_edges = [el for el in bin_edges if el != 0]
        bin_edges = sorted(list(set(bin_edges)))
    else:
        bin_edges = nbins
    bin_lists = {method: [[] for _ in range(len(bin_edges))]
                 for method in per_sample_accuracy.keys()}
    for method, psa in per_sample_accuracy.items():
        for acc, p in zip(psa, param):
            if exact_values:
                bin_id = bin_edges.index(p)
            else:
                bin_id = np.argmin([
                    x - p if x >= p else float('inf') for x in bin_edges])
            bin_lists[method][bin_id].append(acc)
    # print([len(el) for el in bin_lists])
    plt.clf()
    # sns.barplot(x=bin_edges, y=[np.mean(el) for el in bin_lists],
    #             color="salmon")  # palette="BuGn_d")
    # Show the % of dataset in this situation
    sns.barplot(x=bin_edges,
                y=[len(el) / len(param) for el in bin_lists.values()[0]],
                color="#FFC0CB")  # palette="BuGn_d")
    plt.ylabel('Accuracy/Ratio of test set')
    if put_xlabel:
        plt.xlabel(param_name)
    xlocs, xlabels = plt.xticks()
    if not exact_values:
        plt.xticks(xlocs, [
            "<={}".format(el.get_text()) if i == 0 else "{}-{}".format(
                xlabels[i-1].get_text(), xlabels[i].get_text())
            for i, el in enumerate(xlabels)])
    colors = sns.color_palette('colorblind')
    markers = ['o', 's', '^']
    methods_plotted = []
    # Reversing because otherwise colors were not matching for 1 graph and
    # combined graph cases for the slides
    for i, (method, bl) in enumerate(bin_lists.items()[::-1]):
        # Also plot a line showing the actual full accuracy
        total_acc = np.mean(flatten_list(bl))
        plt.axhline(y=total_acc, linestyle='dashed', color=colors[i])
        # Actual accuracy plot
        # kwargs = {'linewidth': 0 if exact_values else 2}
        # Plot the line all the time, looks better
        kwargs = {'linewidth': 2}
        plt.plot(
            [np.mean(el) for el in bl], label=method,
            color=colors[i], marker=markers[i], markersize=12, **kwargs)
        methods_plotted.append(method)
    if len(bin_lists) > 1:
        plt.legend()
    outfpath = osp.join(outdir, param_name.replace(' ', '_') + PLT_EXT)
    plt.savefig(outfpath, bbox_inches='tight', transparent=True)
    print('Saved {}'.format(outfpath))


def plot_acc_binned_loc(fig, acc, locations, param_name, outdir):
    """Show a 3D plot with accuracy at every given position."""
    plt.clf()
    acc = np.array(acc)
    max_locs = max(locations) + 1
    top = np.zeros((max_locs,))
    nrows = int(np.sqrt(max_locs))
    assert nrows * nrows == max_locs  # Must be a perfect square
    for loc in range(max_locs):
        pos = np.where(locations == loc)[0].astype(np.int32)
        top[loc] = np.mean(acc[pos])
    # top = np.flipud(np.reshape(top, [nrows, -1]))
    bottom = np.zeros_like(top)
    width = depth = 1
    _xx, _yy = np.meshgrid(np.arange(nrows), np.arange(nrows))
    x, y = _xx.ravel(), _yy.ravel()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x, y, bottom, width, depth, top)
    outfpath = osp.join(outdir, param_name.replace(' ', '_') + PLT_EXT)
    plt.savefig(outfpath, bbox_inches='tight', transparent=True)
    print('Saved {}'.format(outfpath))


def location_plot_perf_diagnostics(labels, probs, outdir):
    # Compute the statistics for the dataset
    fig = plt.figure(figsize=(8, 4))
    if isinstance(labels, dict):
        per_sample_accuracy = {
            name: (labels[name].argmax(axis=1) ==
                   probs[name].argmax(axis=1)).tolist()
            for name in labels.keys()
        }
    else:
        per_sample_accuracy = {'': (labels.argmax(axis=1) ==
                                    probs.argmax(axis=1)).tolist()}
        # This one only works iwth the single results case
        plot_acc_binned_loc(
            fig, per_sample_accuracy[''], labels.argmax(axis=1),
            'End location of snitch',
            outdir)
    # First, read the metadata for each of the test video
    val_metadata = []
    with open(osp.join(cfg.FILENAME_GT), 'r') as fin:
        for line in tqdm(fin.readlines(), desc='Reading metadata'):
            json_fname = line.split()[0].replace(
                'images', 'scenes').replace('.avi', '.json')
            with open(json_fname, 'r') as fin:
                val_metadata.append(json.load(fin))
    # It's unclear why in the following the perf went up and then down.
    # Maybe vis only the hard videos (where it was covered in the end, and
    # see how long they were covered and if that has anything to do with the
    # perf.) Maybe one possible explanation is that by being covered for long,
    # the previous frames are ambiguous, and then in the last clip makes it
    # unabmiguous. While if it was visible earlier, then it will keep
    # confusing the model.
    plot_acc_binned(per_sample_accuracy,
                    [snitch_last_contained(el) for
                     el in val_metadata],
                    'Last frame contained',
                    outdir, nbins=7)
    plot_acc_binned(per_sample_accuracy,
                    [snitch_contained_at_end(el, three_class=False) for
                     el in val_metadata],
                    'Contained in end',
                    outdir,
                    # nbins=['Open', 'Contained'],
                    nbins=['Open', 'Contained'],
                    exact_values=True)
    plot_acc_binned(per_sample_accuracy,
                    [snitch_contained_at_end(el, three_class=True) for
                     el in val_metadata],
                    'Contained',
                    outdir,
                    # nbins=['Open', 'Contained'],
                    nbins=['Never', 'In between', 'At end'],
                    exact_values=True)
    plot_acc_binned(per_sample_accuracy,
                    [snitch_last_moved_frame(el) for
                     el in val_metadata],
                    'Frame at which snitch last moved',
                    outdir, nbins=5)
    plot_acc_binned(per_sample_accuracy,
                    [snitch_contained_time(el) for
                     el in val_metadata],
                    'Number of frames where snitch is contained',
                    outdir, nbins=5)
    plot_acc_binned(per_sample_accuracy,
                    [snitch_displacement(el) for el in val_metadata],
                    'Snitch Displacement', outdir, nbins=5)
    plot_acc_binned(per_sample_accuracy,
                    [count_objects(el) for el in val_metadata],
                    'Objects Count', outdir, nbins=5)


def visualize_location(args):
    # Load the predictions
    load_fpaths = {
        'tracker': osp.join(cfg.CHECKPOINT.DIR, 'tracker/results_probs.pkl'),
        'lstm': osp.join(cfg.CHECKPOINT.DIR, 'lstm/results_probs_lstm.pkl'),
        'avg': osp.join(cfg.CHECKPOINT.DIR, 'results_probs.pkl'),
    }
    all_labels = {}
    all_probs = {}
    for name, load_fpath in load_fpaths.items():
        if not osp.exists(load_fpath):
            print('load_fpath {} NOT EXIST!!'.format(load_fpath))
            continue
        outdir = osp.dirname(load_fpath)
        with open(load_fpath, 'r') as fin:
            results = pkl.load(fin)
        # Sanity check
        logger.info('Sanity check: making sure the results match up')
        print(evaluate_result(results))
        # Find the videos with highest and lower x-entropy
        labels, probs = get_labels_and_prob(results)
        # normalize by to get a distribution. The predictions should ideally
        # be softmaxes, so if probs consists of a sum of multiple predictions,
        # this should fix it
        probs /= probs.sum(axis=1, keepdims=True)
        # probs /= cfg.TEST.NUM_CLIPS
        all_probs[name] = probs
        all_labels[name] = labels
        location_plot_perf_diagnostics(labels, probs, outdir)
        # Doing this here mostly just for the JPG files of the hard/easy cases
        # we use in the paper.
        make_supp_video(labels, probs, None, outdir)
    outdir = osp.join(cfg.CHECKPOINT.DIR, 'combined')
    mkdir_p(outdir)
    location_plot_perf_diagnostics(all_labels, all_probs, outdir)
    make_supp_video(
        all_labels['lstm'], all_probs['lstm'],
        # These predictions are generated by test code, by setting store_vis
        # argument in the flags to true
        osp.join(cfg.CHECKPOINT.DIR, 'vis_test_fullLbl'),
        outdir)


def main():
    parser = argparse.ArgumentParser(description='Visualize predictions')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
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
    print_cfg()
    visualize_location(args)


if __name__ == '__main__':
    main()
