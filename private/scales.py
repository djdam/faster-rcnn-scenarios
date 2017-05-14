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

def create_scenarios():
    small_scales = deepcopy(default_scenario)
    small_scales.name("scales_4_8_16")
    small_scales.rpn_config.anchor_scales = [4, 8, 16]
    small_scales.fast_rcnn_config.anchor_scales = small_scales.rpn_config.anchor_scales

    default_scales = deepcopy(default_scenario)
    default_scales.name("scales_8_16_32")
    default_scales.rpn_config.anchor_scales = [8, 16, 32]
    default_scales.fast_rcnn_config.anchor_scales = default_scales.rpn_config.anchor_scales

    large_scales = deepcopy(default_scenario)
    large_scales.name("scales_16_32_64")
    large_scales.rpn_config.anchor_scales = [16, 32, 64]
    large_scales.fast_rcnn_config.anchor_scales = large_scales.rpn_config.anchor_scales

    return small_scales, default_scales, large_scales