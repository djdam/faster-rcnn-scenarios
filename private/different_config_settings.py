#!/usr/bin/env python
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Dennis Dam
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""
from os.path import join, dirname, abspath
import sys

this_dir = dirname(abspath(__file__))
sys.path.insert(0,join(this_dir,'..','src'))

import _init_paths
from copy import deepcopy
from default import default_scenario

scenarios_dir = join(this_dir, 'scenarios')

def create_scenarios():

    thresholds = [
        dict(id="bbox_thresh_03", TRAIN=dict(BBOX_THRESH=0.3)),
        dict(id="bbox_thresh_07", TRAIN=dict(BBOX_THRESH=0.7)),
        dict(id="bbox_thresh_09", TRAIN=dict(BBOX_THRESH=0.9)),
    ]

    rpn_overlap = [
        dict(id="rpn_overlap_03_05", TRAIN=dict(RPN_NEGATIVE_OVERLAP=0.3, RPN_POSITIVE_OVERLAP=0.5)),
        dict(id="rpn_overlap_05_07",  TRAIN=dict(RPN_NEGATIVE_OVERLAP=0.5, RPN_POSITIVE_OVERLAP=0.7)),
        dict(id="rpn_overlap_07_09",  TRAIN=dict(RPN_NEGATIVE_OVERLAP=0.7, RPN_POSITIVE_OVERLAP=0.9)),
    ]

    # rpn_min_size = [dict(id="rpn_min_size_%s"%i, TRAIN=dict(RPN_MIN_SIZE=i), TEST=dict(RPN_MIN_SIZE=i)) for i in [1,5,10]]

    def create(snr):
        s = deepcopy(default_scenario)
        s.name(snr["id"])
        if snr.get("TRAIN", None) != None:
            s.config["TRAIN"].update(snr["TRAIN"])
        if snr.get("TEST", None) != None:
            s.config["TEST"].update(snr["TEST"])
        return s

    return [create(snr) for snr in thresholds + rpn_overlap]