from __future__ import print_function

import itertools
import logging


class MovementRecord:
    def __init__(self, objects, total_frames):
        """
        Args:
            objects (list of blender objects)
        """
        # initialize a list for each object
        self.total_frames = total_frames
        self.timeline = {}
        self.contains = {}
        for obj in objects:
            self.timeline[obj] = []
            # store for each frame, what is contained
            self.contains[obj] = list(itertools.repeat(None, total_frames + 1))

    def insert(self, obj, action, other_obj, start_frame, end_frame):
        """
        Args:
            obj: The blender object that was acted upon
            action: The function/action op that was executed
            other_obj: Other blender obj involved in this.
                Only useful for contains op
            start_frame: The start_frame of the action
            end_frame: The end_frame of the action
        """
        self.timeline[obj].append((
            action,
            other_obj,
            start_frame, end_frame))
        logging.debug('Recorded action for {}: {}'.format(
            obj, self.human_readable_interval(self.timeline[obj][-1])))
        # If contains, also update the contains data structure
        if action.__name__ == '_contain':
            # Assume it will be contained for ever, until the same object hits
            # a pick_place
            assert obj != other_obj, '{} can not contain itself!'.format(
                obj)  # will lead to infinite recursion when checking
            for frame_id in range(start_frame, self.total_frames + 1):
                # Nothing should already be contained
                assert self.contains[obj][frame_id] is None, \
                    '{} already contains {} at frame {}. ' \
                    'Cant contain {} (btw {} and {}) now also..?' \
                    'This may be because I use generous timing for contains ' \
                    'op, i.e. since the the cone picks up, it is counted as ' \
                    'contains. Anyway, ignore this setup.'.format(
                        obj, self.contains[obj][frame_id], frame_id,
                        other_obj, start_frame, end_frame)
                self.contains[obj][frame_id] = other_obj
            logging.debug('{} contains {} as of {}'.format(
                obj, other_obj, start_frame))
        elif action.__name__ == '_pick_place':
            # If something was contained, it will no longer be
            for frame_id in range(end_frame, self.total_frames + 1):
                self.contains[obj][frame_id] = None

    def get_dict(self):
        """
        Return the record in a human readable dictionary format
        """
        res = {}
        for ob, intervals in self.timeline.items():
            res[ob.name] = [
                self.human_readable_interval(interval) for interval
                in intervals]
        return res

    def human_readable_interval(self, interval):
        return (
            interval[0].__name__,
            interval[1].name if interval[1] else None,
            interval[2], interval[3])

    def was_contained(self, ob1, ob2, frame_id):
        """ Return true/false, based on whether ob2 was contained in ob1. """
        if ob1 is None:
            return False
        if ob1 == ob2:
            return True
        return self.was_contained(self.contains[ob1][frame_id], ob2, frame_id)
