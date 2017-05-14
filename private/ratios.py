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
    five_ratios = deepcopy(default_scenario)
    five_ratios.name("ratios_five")
    five_ratios.rpn_config.anchor_ratios = [0.25, 0.5, 1, 2, 4]
    five_ratios.fast_rcnn_config.anchor_ratios = five_ratios.rpn_config.anchor_ratios

    seven_ratios = deepcopy(default_scenario)
    seven_ratios.name("ratios_seven")
    seven_ratios.rpn_config.anchor_ratios =  [1/8.0, 0.25, 0.5, 1, 2, 4, 8]
    seven_ratios.fast_rcnn_config.anchor_ratios = seven_ratios.rpn_config.anchor_ratios

    return five_ratios, seven_ratios