from __future__ import print_function

import random
import bpy
import numpy as np
import itertools
import math
import logging

PICK_HEIGHT = 2
MAX_TRIALS = 100  # Max number of times to try to find a good op that works
MOVEMENT_MIN = 20
MOVEMENT_MAX = 30
# Upper bound on the number of objects that move in a given segment. This
# can be set using the argparse. Lower numbers mean sparser videos.
MAX_MOTIONS = 999999


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def sanitize_locations(locations):
    """ The location maybe vectors etc, need a serializable nice format. """
    res = {}
    for frame, location in enumerate(locations):
        res[frame] = [location[0], location[1], location[2]]
    return res


def random_objects_movements(
        objects, blender_objects, args, total_frames, min_dist, record,
        max_motions=MAX_MOTIONS):
    bpy.ops.screen.frame_jump(end=False)
    # from https://blender.stackexchange.com/a/70478
    all_obj_locations = []
    # add all objects initial locations, to make sure to not move over
    # stationary objects. By default everything is stationary.
    all_obj_locations = [
        list(itertools.repeat(obj.location.copy(), args.num_frames + 1))
        for obj in blender_objects]
    objects = zip(objects, blender_objects)  # tie them together
    # Make a list, to specify what objects are "tied together" -- must move
    # together. The first object is containing everything after it.
    objects = [[el] for el in objects]
    all_obj_locations = [[el] for el in all_obj_locations]
    # Make sure we start from a sane world
    assert_no_collisions(all_obj_locations, objects, min_dist, record)
    # Now start by time intervals, and add a single object or multiple object
    # action sequence.
    cur_frame = 0
    while cur_frame <= total_frames - MOVEMENT_MAX:
        ops = [add_movements_multiObj_try, add_movements_singleObj]
        op = random.choice(ops)
        end_frame = op(
            objects, cur_frame, all_obj_locations, min_dist, total_frames,
            record, max_motions=max_motions)
        cur_frame = end_frame + 1
        logging.debug('objects now: {}'.format(
            [[e[0]['instance'] for e in el] for el in objects]))
        assert_top_obj_is_cone(objects)
    # Add the locations
    objects = flatten_list(objects)
    all_obj_locations = flatten_list(all_obj_locations)
    assert len(objects) == len(all_obj_locations)
    for obid in range(len(objects)):
        objects[obid][0]['locations'] = sanitize_locations(
            all_obj_locations[obid])


def assert_top_obj_is_cone(objects):
    for object in objects:
        if len(object) > 1:
            assert object[0][0]['shape'] == 'cone', \
                'Only cones are allowed to contain other objects'


def add_movements_multiObj_try(objects, start_frame, all_obj_locations,
                               min_dist, total_frames, record,
                               max_motions=MAX_MOTIONS):
    for _ in range(100):  # try 10 times to find
        # Pick random pairs, and see if first can contain the secondself.
        # If so, then go ahead and contain
        i1, i2 = random.sample(range(len(objects)), 2)
        assert i1 != i2, \
            'This should never happen, random.sample is without replacement'
        frames_this_move = random.randint(MOVEMENT_MIN, MOVEMENT_MAX)
        new_start_frame = start_frame + random.randint(0, 10)
        new_end_frame = min(new_start_frame + frames_this_move, total_frames)
        if not _can_contain(objects[i1], objects[i2],
                            # other objects, to check collisions
                            [el for i, el in enumerate(objects)
                             if i not in [i1, i2]],
                            # other objects locations over frames
                            [el for i, el in enumerate(all_obj_locations)
                             if i not in [i1, i2]],
                            new_end_frame,
                            min_dist):
            continue
        # Ideally need to do 2 steps: simulate motion and then actually move,
        # but for simple case it is fine since I check for collisions at the
        # end points, and at a time, only one object is moved.
        # The new objects are added after, and by then the contain op has
        # already been added
        obj_locations = _contain(
            objects[i1][0][1], objects[i2][0][1],
            start_frame=new_start_frame, end_frame=new_end_frame)
        record.insert(objects[i1][0][1], _contain, objects[i2][0][1],
                      new_start_frame, new_end_frame)
        logging.debug('Moved {} to {}'.format(
            objects[i1][0][1], objects[i2][0][1]))
        for k in range(len(objects[i1])):
            all_obj_locations[i1][k][new_start_frame: new_end_frame + 1] = \
                obj_locations[:]
            all_obj_locations[i1][k][new_end_frame:] = \
                itertools.repeat(
                    obj_locations[-1],
                    len(all_obj_locations[i1][k]) - new_end_frame)
        assert_no_collisions(all_obj_locations, objects, min_dist, record)

        # Combine the objects. The first element of the list of objects is the
        # TOP-MOST object in the heirarchy!! [IMP]
        objects[i1] += objects[i2]
        objects.pop(i2)
        all_obj_locations[i1] += all_obj_locations[i2]
        all_obj_locations.pop(i2)
        # This is the idx of the final object. Do not touch it when adding
        # single motions
        affected_idx = i1 if i1 < i2 else i1 - 1
        # The following should always be true
        for i in range(len(objects)):
            assert len(objects[i]) == len(all_obj_locations[i]), \
                '{} vs {}'.format(len(objects[i]), len(all_obj_locations[i]))
        assert_no_collisions(all_obj_locations, objects, min_dist, record)

        # add single object movements for the rest of the objects in this time
        new_end_frame_singleObjMotion = add_movements_singleObj(
            objects,
            start_frame,
            all_obj_locations,
            min_dist, total_frames, record, ignore_obids=[affected_idx],
            max_motions=(max_motions - 1))
        # The following should always be true
        for i in range(len(objects)):
            assert len(objects[i]) == len(all_obj_locations[i]), \
                '{} vs {}'.format(len(objects[i]), len(all_obj_locations[i]))
        assert_no_collisions(all_obj_locations, objects, min_dist, record)
        return max(new_end_frame, new_end_frame_singleObjMotion)
    return start_frame - 1


def _can_contain(ob1, ob2, other_objects, all_obj_locations, end_frame,
                 min_dist):
    """ Return true if ob1 can contain ob2. """
    assert len(other_objects) == len(all_obj_locations)
    # Only cones do the contains, and can contain spl or smaller sphere/cones,
    # cylinders/cubes are too large
    if (len(ob1) == 1 and ob1[0][0]['sized'] > ob2[0][0]['sized'] and
            ob1[0][0]['shape'] == 'cone' and
            ob2[0][0]['shape'] in ['cone', 'sphere', 'spl']):
        # Also make sure the moved object will not collide with anything
        # there
        collisions = [
            _obj_overlap(
                # ob2 location since the ob1 will be moved to ob2's location
                # but will have the size of ob1,
                (ob2[0][1].location[0], ob2[0][1].location[1],
                 ob1[0][1].location[2]),
                ob1[0][0]['sized'],
                # top objects location at the end point, and its size
                other_locations[0][end_frame], other_obj[0][0]['sized'],
                min_dist)
            for other_obj, other_locations in
            zip(other_objects, all_obj_locations)]
        if not any(collisions):
            return True
    return False


def _contain(blend_top_ob1, blend_top_ob2, start_frame, end_frame,
             pos_only=False):
    return _pick_place(
        # the 0th element is the outermost in hierarchical nesting
        blend_top_ob1, blend_top_ob1.location.copy(),
        start_frame, end_frame,
        x=blend_top_ob2.location[0], y=blend_top_ob2.location[1],
        pos_only=pos_only)


def add_movements_singleObj(objects, start_frame, all_obj_locations, min_dist,
                            total_frames, record, ignore_obids=(),
                            max_motions=MAX_MOTIONS):
    # order to iterate through the frames in
    obj_order = np.random.permutation(len(objects))
    # Remove any object IDs in dont_touch. They have either already been
    # moved this round, or for whatever reason we don't want to move.
    obj_order = [el for el in obj_order if el not in ignore_obids]
    # Only apply the motions to this many objects. This makes the motions
    # sparser, and the random performance for tasks 1 and 2 lower.
    obj_order = obj_order[:max_motions]
    logging.debug(obj_order)
    logging.debug('Moving in order {}'.format([
        objects[i][0][0]['shape'] for i in obj_order]))
    last_frame_added = -1
    splits = []
    for obid in obj_order:
        frames_this_move = random.randint(MOVEMENT_MIN, MOVEMENT_MAX)
        new_start_frame = start_frame + random.randint(0, 10)
        new_end_frame = min(new_start_frame + frames_this_move, total_frames)
        last_frame_added = max(new_end_frame, last_frame_added)
        if new_end_frame <= new_start_frame:
            logging.error('>>> This should not happen')
            # most likely won't be able to get anything else, just die
            return total_frames
        obj_locations_per_obj, split = add_movements(
            objects[obid],
            record,
            start_frame=new_start_frame, end_frame=new_end_frame,
            # Do not add the current object in "other" object locations, as
            # then it will always be "colliding" with itself.
            # Also need to compare to all the elements in the list and not the
            # top most only, as the top might have been moved out in an earlier
            # action.
            other_obj_locs=flatten_list([
                el for i, el in enumerate(all_obj_locations) if i != obid]),
            # Though we only need the outer-most element for size, but just so
            # the sizes match to other_obj_locs, taking all objs
            other_obj_sizes=flatten_list([[e[0]['sized'] for e in el] for i, el
                                          in enumerate(objects) if i != obid]),
            min_dist=min_dist)
        splits.append(split)
        for i in range(len(objects[obid])):
            all_obj_locations[obid][i][new_start_frame: new_end_frame + 1] = \
                obj_locations_per_obj[i]
            # Now make all positions after the last frame to the new last
            # position, since it will sit there unless moved. Think this is
            # what was leading to collisions with moved objects
            all_obj_locations[obid][i][new_end_frame:] = \
                itertools.repeat(
                    obj_locations_per_obj[i][-1],
                    len(all_obj_locations[obid][i]) - new_end_frame)
        assert_no_collisions(all_obj_locations, objects, min_dist, record,
                             ignore_obids=ignore_obids)
    # split the objects that were split
    obids_split = [obid for i, obid in enumerate(obj_order) if splits[i]]
    final_objects = []
    final_all_obj_locations = []
    for obid, (object, all_obj_location) in enumerate(zip(
            objects, all_obj_locations)):
        if obid in obids_split:
            final_objects.append([object[0]])
            final_objects.append(object[1:])
            final_all_obj_locations.append([all_obj_location[0]])
            final_all_obj_locations.append(all_obj_location[1:])
        else:
            final_objects.append(object)
            final_all_obj_locations.append(all_obj_location)
    objects[:] = final_objects[:]
    all_obj_locations[:] = final_all_obj_locations[:]
    assert_no_collisions(all_obj_locations, objects, min_dist, record,
                         ignore_obids=ignore_obids)
    return last_frame_added


def assert_no_collisions(obj_locs, objs, min_dist, record, ignore_obids=()):
    # only consider the top-most objects, since anything inside will be
    # colliding, by definition
    obj_locs = [obj_loc[0] for obj_loc in obj_locs]
    objs = [obj[0] for obj in objs]
    assert len(objs) == len(obj_locs)
    for i in range(len(objs)):
        # Check if i is colliding with anything
        if i in ignore_obids:
            # There is a special case when the multi-objects are being moved
            # and I also want to move other objects. So, in those case, the
            # other objects have not yet been merged together, so I don't want
            # to incur a collision on the contains op.
            continue
        for j in range(len(objs)):
            if i == j:
                continue
            assert len(obj_locs[i]) == len(obj_locs[j]), 'Number of frames ' \
                '{} and {} should be same ({} vs {})'.format(
                    objs[i][1], objs[j][1], len(obj_locs[i]), len(obj_locs[j]))
            overlap_frames = []
            for frame_id in range(len(obj_locs[i])):
                overlap = _obj_overlap(
                    obj_locs[i][frame_id], objs[i][0]['sized'],
                    obj_locs[j][frame_id], objs[j][0]['sized'], min_dist)
                if not overlap:
                    continue
                # check if the overlap happened when the objects were contained
                # in each other. In that case it is fine
                if record.was_contained(objs[i][1], objs[j][1], frame_id) or \
                   record.was_contained(objs[j][1], objs[i][1], frame_id):
                    continue
                # else, raise the error
                overlap_frames.append(frame_id)
            if len(overlap_frames) > 0:
                logging.error(
                    'WARNING: Overlap detected between {} (size {} loc {}) '
                    'and {} (size {} loc {}) at frame {}'.format(
                        objs[i][1], objs[i][0]['sized'], obj_locs[i][frame_id],
                        objs[j][1], objs[j][0]['sized'], obj_locs[j][frame_id],
                        overlap_frames))
                raise AssertionError('Overlap')


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def add_movements(objs, record, start_frame, end_frame,
                  other_obj_locs=(), other_obj_sizes=(), min_dist=0):
    """
    objs can contain multiple objects nested in each other. The first one is
    the outermost.
    """
    all_actions = [  # action, and whether it will split the objects or not
        ([_slide], False),
        ([_pick_place], False),
        ([_rotate], False),
    ]
    if len(objs) > 1:
        # pick_place for the top obj
        all_actions = [
            ([_slide] * len(objs), False),
            ([_pick_place] + [_no_op] * (len(objs) - 1), True),
        ]
    elif objs[0][0]['shape'] in ['cone', 'sphere']:  # only 1 obj
        all_actions = [
            ([_slide], False),
            ([_pick_place], False),
        ]  # no rotate for these
    # add current locations as a keyframe
    _add_keyframe([blend_obj for obj, blend_obj in objs], start_frame)
    # TODO(rgirdhar): assert all objects are the same location
    num_trials = 0
    while True:  # try to find a movement that does not collide with others
        action, split = random.choice(all_actions)
        # Some ops need end points
        kwargs = {}
        # If there is any slide/pick_place in the actions
        if len(intersection(action, [_slide, _pick_place])) > 0:
            kwargs.update(
                {'x': random.uniform(-3, 3), 'y': random.uniform(-3, 3)})
        # pos_only makes sure that we do not effect any of the actions,
        # but only compute the positions to compute overlaps etc.
        obj_pos_all_subObj = [list(obj_action(
            None, blend_obj.location.copy(), pos_only=True,
            start_frame=start_frame, end_frame=end_frame, **kwargs))
            for (_, blend_obj), obj_action in zip(objs, action)]
        for obj_pos in obj_pos_all_subObj:
            assert len(obj_pos) == end_frame - start_frame + 1, \
                'pos didnt match for action {}. {} vs {}'.format(
                    action, len(obj_pos), end_frame - start_frame + 1)
        clean = [_no_object_overlaps(
            obj_pos, obj['sized'], other_obj_locs, other_obj_sizes,
            start_frame, end_frame, min_dist)
            for (obj, _), obj_pos in zip(objs, obj_pos_all_subObj)]
        if split:
            # In this case, clean should also check if the final positions are
            # sufficiently far apart or not.
            clean.append(not _obj_overlap(
                obj_pos_all_subObj[0][-1], objs[0][0]['sized'],
                obj_pos_all_subObj[1][-1], objs[1][0]['sized'], min_dist))
        if not all(clean) and num_trials > MAX_TRIALS:
            logging.debug('Hit the max_trials')
            action = [_no_op] * len(objs)
            split = False
            if 'x' in kwargs:  # no_op does not take these
                del kwargs['x'], kwargs['y']
            clean = [True]
        if all(clean):
            all_obj_pos = []  # for each sub-object
            for (_, blend_obj), obj_action in zip(objs, action):
                all_obj_pos.append(list(obj_action(
                    blend_obj,
                    # Only take X/Y from the covering object
                    (blend_obj.location[0], blend_obj.location[1],
                     blend_obj.location[2]),
                    start_frame=start_frame, end_frame=end_frame,
                    **kwargs)))
                record.insert(blend_obj, obj_action, None,
                              start_frame, end_frame)
            break
        num_trials += 1
    bpy.ops.screen.frame_jump(end=False)
    # +1 because the frame numbering starts at 0, and frame number total_frames
    # is the last frame
    assert len(all_obj_pos[0]) == end_frame - start_frame + 1, \
        '{} vs {}'.format(len(all_obj_pos[0]), end_frame - start_frame + 1)
    return all_obj_pos, split


def _no_object_overlaps(pos, size, other_obj_locs, other_obj_sizes,
                        start_frame, end_frame, min_dist):
    assert len(other_obj_locs) == len(other_obj_sizes)
    assert len(pos) == (end_frame - start_frame + 1)
    for i, frame_id in enumerate(range(start_frame, len(other_obj_locs[0]))):
        if frame_id in range(start_frame, end_frame + 1):
            new_loc = pos[i]
        else:
            # This is for the case when this object will stay at this place.
            # We need to make sure nothing comes in at this point either.
            new_loc = pos[-1]
        for obj_locs, obj_size in zip(other_obj_locs, other_obj_sizes):
            obj_loc = obj_locs[frame_id]
            if _obj_overlap(new_loc, size, obj_loc, obj_size, min_dist):
                return False
    return True


def _obj_overlap(loc1, size1, loc2, size2, min_dist):
    dx, dy, dz = loc1[0] - loc2[0], loc1[1] - loc2[1], loc1[2] - loc2[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist - size1 - size2 < min_dist:
        return True
    return False


def _no_op(obj, init_loc, start_frame, end_frame, pos_only=False, **kwargs):
    # Adding **kwargs to read x,y etc random extra keyword args and
    # ignore them. They get passed when I'm splitting objects, and the top
    # one is being pick_placed, while the rest are no-op
    if not pos_only:
        _add_keyframe(obj, end_frame)
    # No movement
    return itertools.repeat(init_loc, end_frame - start_frame + 1)


def _rotate(obj, init_loc, start_frame, end_frame, angle=np.array([0, 90, 0]),
            num_keyframes=1, pos_only=False):
    pos = itertools.repeat(init_loc, end_frame - start_frame + 1)
    if pos_only:
        return pos
    _add_keyframe(obj, start_frame, 'rotation_euler')
    rot = obj.rotation_euler.copy()
    tot_frames = end_frame - start_frame
    # Need to convert to radians (imp!)
    # https://blender.stackexchange.com/a/43089
    angle = (angle / 180) * 3.14
    # add multiple keyframes if needed
    for frame_id in range(1, num_keyframes + 1):
        ratio = frame_id / num_keyframes
        dAngle = ratio * angle
        new_rot = (rot[0] + dAngle[0], rot[1] + dAngle[1], rot[2] + dAngle[2])
        obj.rotation_euler = new_rot
        _add_keyframe(obj, start_frame + ratio * tot_frames, 'rotation_euler')
    # No movement
    return pos


def _slide(obj, init_loc, start_frame, end_frame,
           x=None, y=None, pos_only=False):
    if not pos_only:
        _add_keyframe(obj, start_frame)
    new_loc = (x, y, init_loc[2])
    return move_to_location(obj, init_loc, new_loc, start_frame, end_frame,
                            pos_only=pos_only)


def move_to_location(obj, init_loc, new_loc, start_frame, end_frame,
                     pos_only=False):
    # Compute linear interpolation between these two points
    pts = []
    for i in range(3):
        pts.append(np.interp(
            range(start_frame, end_frame + 1),
            [start_frame, end_frame],
            [init_loc[i], new_loc[i]]).reshape((-1,)).tolist())
    res = list(zip(*pts))
    assert len(res) == (end_frame - start_frame + 1), 'Must match {} to {}' \
        .format(len(res), end_frame - start_frame + 1)
    if pos_only:
        return res
    # Now effect it
    obj.location = new_loc
    _add_keyframe(obj, end_frame)
    return res


def _pick_place(obj, init_loc, start_frame, end_frame,
                x=None, y=None, pos_only=False):
    pos = []
    if not pos_only:
        _add_keyframe(obj, start_frame)
    loc = init_loc
    tot_frames = end_frame - start_frame + 1

    # pick up
    new_loc = (loc[0], loc[1], loc[2] + PICK_HEIGHT)
    end_frame_1 = start_frame + int(0.2 * tot_frames)
    pos += move_to_location(obj, init_loc, new_loc, start_frame, end_frame_1,
                            pos_only=pos_only)

    # slide
    end_frame_2 = start_frame + int(0.8 * tot_frames)
    pos += _slide(obj, pos[-1], end_frame_1 + 1, end_frame_2, x=x, y=y,
                  pos_only=pos_only)

    # place
    final_loc = pos[-1]
    new_loc = (final_loc[0], final_loc[1], loc[2])
    pos += move_to_location(obj, final_loc, new_loc, end_frame_2 + 1,
                            end_frame, pos_only=pos_only)

    return pos


def _move_block(objs, id, delta=None, pos=None):
    assert delta is None or pos is None, 'Both can not be defined together'
    if delta is not None:
        pos = objs[id].location
        pos = (pos[0] + delta[0], pos[1] + delta[1], pos[2] + delta[2])
    objs[id].location = pos


def _add_keyframe(blender_objects, frame_id, data_path='location'):
    # If only a single object
    if not isinstance(blender_objects, list):
        blender_objects = [blender_objects]
    for obj in blender_objects:
        obj.keyframe_insert(data_path=data_path, frame=frame_id)