from os import chmod
from os.path import join, dirname, abspath
import sys

import _init_paths
from _init_paths import faster_rcnn_root, caffe_root
from scenario import Scenario
from network import RegionProposalNetworkConfig as RpnConfig, FastRcnnNetworkConfig as FastRcnnConfig
from solver import SolverConfig
from copy import deepcopy
import yaml


this_dir = dirname(abspath(__file__))
sys.path.insert(0,join(this_dir,'..','src'))

scenarios_dir = join(this_dir, 'scenarios')

default_cfg = join(this_dir, 'default-cfg.yml')

STATS_SCALES = [2, 4, 8, 16, 32]
DEFAULT_SCALES = [8, 16, 32]
DEFAULT_RATIOS=[0.5,1,2]
MORE_RATIOS=[0.25, 0.5, 1, 2, 4]
EVEN_MORE_RATIOS=[0.125, 0.25, 0.5, 1, 2, 4, 8]
STATS_AMOUNT_OF_RATIOS = [0.2, 0.5, 0.8, 1, 1/0.8, 1/0.5, 10]

default_scenario=Scenario(
        scenarios_dir=scenarios_dir,
        scenario="scales_2_4_8",
        train_imdb="technicaldrawings_numbers_train",
        test_imdb="technicaldrawings_numbers_val",
        weights_path=join(faster_rcnn_root, "data/imagenet_models/ZF.v2.caffemodel"),  # you have to download this first
        gpu_id=0,
        # max_iters=[1, 1, 1, 1],  # max iters
        max_iters=[1000, 1000, 1000, 1000],  # max iters
        rpn_config=RpnConfig(num_classes=2, anchor_scales=DEFAULT_SCALES, anchor_feat_stride=16, anchor_ratios=DEFAULT_RATIOS),
        fast_rcnn_config=FastRcnnConfig(num_classes=2, anchor_scales=DEFAULT_SCALES, anchor_feat_stride=16, anchor_ratios=DEFAULT_RATIOS),
        solver_config=SolverConfig(step_size=3500),
        config=yaml.load(open(default_cfg))
    )