#!/usr/bin/env python
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Dennis Dam
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""
from os import chmod
from os.path import join, dirname, abspath
import sys

this_dir = dirname(abspath(__file__))
sys.path.insert(0,join(this_dir,'..','src'))

from _init_paths import faster_rcnn_root, caffe_root
from copy import deepcopy
from default import default_scenario

scenarios_dir = join(this_dir, 'scenarios')

def avg(list):
    return float(sum(list)) / len(list)

def create_scenarios():

    SMALL_SCALES=[4, 8, 16]
    DEFAULT_SCALES=[8, 16, 32]
    BROAD_RANGE_OF_SCALES=[0.25,0.5,0.75,1,2,4,8,16,32]
    FOUR_SCALES_XS=[2, 4, 8, 16]
    FOUR_SCALES_MD=[4, 8, 16, 32]
    FOUR_SCALES_LG=[8, 16, 32, 64]
    FIVE_SCALES=[4, 8, 16, 32, 64]
    LARGE_SCALES=[16, 32, 48]
    DEFAULT_RATIOS=[0.5,1,2]
    MORE_RATIOS=[0.25,0.5,0.75,1]
    MORE_RATIOS+=[1/i for i in MORE_RATIOS if i != 1]
    EVEN_MORE_RATIOS=[0.125, 0.25, 0.5, 1, 2, 4, 8]
    EXTREME_AMOUNT_OF_RATIOS = [1/16.0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]

    STATS_AMOUNT_OF_RATIOS = [0.2, 0.35, 0.5, 0.65, 0.8, 1, 1/0.8, 1/0.65, 1/0.5, 1/0.35, 10]
    STATS_LESS_AMOUNT_OF_RATIOS = [0.2, 0.5, 1, 2, 10]

    def create(snr):
        s = deepcopy(default_scenario)
        s.name(snr["id"])
        if snr.get('scales', None) != None:
            s.rpn_config.anchor_scales = snr['scales']
            s.fast_rcnn_config.anchor_scales = s.rpn_config.anchor_scales
        if snr.get('ratios', None) != None:
            s.rpn_config.anchor_ratios = snr['ratios']
            s.fast_rcnn_config.anchor_ratios = s.rpn_config.anchor_ratios
        if snr.get("feat_stride", None) != None:
            s.rpn_config.anchor_feat_stride = snr.get("feat_stride")
            s.fast_rcnn_config.anchor_feat_stride = snr.get("feat_stride")

        return s

    scales_and_ratios=[
        # dict(id="feat_stride_2", feat_stride=2),
        dict(id="feat_stride_8", feat_stride=8,scales=[1,2,4,8]),
        dict(id="feat_stride_4", feat_stride=4, scales=[1,2,4,8]),
        dict(id="feat_stride_2", feat_stride=2, scales=[1,2,4,8]),
    ]

    return [create(snr) for snr in scales_and_ratios]