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

import _init_paths
from _init_paths import faster_rcnn_root, caffe_root
from scenario import Scenario
from network import RegionProposalNetworkConfig as RpnConfig, FastRcnnNetworkConfig as FastRcnnConfig
from solver import SolverConfig
from copy import deepcopy
import yaml

scenarios_dir = join(this_dir, 'scenarios')

def create_scenarios():
    alt_opt_cfg = join(faster_rcnn_root, "experiments/cfgs/faster_rcnn_alt_opt.yml")
    base_scenario = Scenario(
        scenarios_dir=scenarios_dir,
        scenario="scales_2_4_8",
        train_imdb="technicaldrawings_single-numbers_train",
        test_imdb="technicaldrawings_single-numbers_val",
        weights_path=join(faster_rcnn_root, "data/imagenet_models/ZF.v2.caffemodel"),  # you have to download this first
        gpu_id=0,
        max_iters=[1, 1, 1, 1],  # max iters
        rpn_config=RpnConfig(num_classes=2, anchor_scales=[8, 16, 32], anchor_feat_stride=16),
        fast_rcnn_config=FastRcnnConfig(num_classes=2),
        solver_config=SolverConfig(),
        config=yaml.load(open(alt_opt_cfg))
    )

    small_scales = deepcopy(base_scenario)
    small_scales.name("scales_4_8_16")
    small_scales.rpn_config.anchor_scales = [4, 8, 16]

    default_scales = deepcopy(base_scenario)
    default_scales.name("scales_8_16_32")
    default_scales.rpn_config.anchor_scales = [8, 16, 32]

    large_scales = deepcopy(base_scenario)
    large_scales.name("scales_16_32_64")
    large_scales.rpn_config.anchor_scales = [16, 32, 64]

    return small_scales, default_scales, large_scales

def generate():
    small, default, large=create_scenarios()
    for scenario in [small,default,large]:
        print "*"*160
        print 'Generating scenario:',scenario.scenario
        print "*"*160
        print scenario
        scenario.generate()

    run_all_script_path = join(scenarios_dir, 'run_all.sh')
    run_all_script = open(run_all_script_path, 'wb')
    run_all_script.write("""
        {small} || true
        {default} || true
        {large} || true
        """.format(
        small=small.script_path,
        default=default.script_path,
        large=large.script_path)
    )
    chmod(run_all_script_path, 0755)

if __name__ == '__main__':
    generate()