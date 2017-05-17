#!/usr/bin/env python
import _init_paths
from train import train_fast_rcnn, dir_exists_or_create
import argparse
import sys
from fast_rcnn.config import cfg, cfg_from_file
from scenario import Scenario
import os
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')

    parser.add_argument('--scenario_file', help='Path scenario file (e.g. /home/user/scenario.yml)')
    parser.add_argument('--weights', help='weights)', type=str)

    # parser.add_argument('output_dir', help='.pkl file of stage 1 RPN)', type=str)
    # parser.add_argument('imdb', help='imdb name', type=str)
    # parser.add_argument('init_model', help='init_model', type=str)
    # parser.add_argument('solver', help='solver', type=str)
    # parser.add_argument('cfg', help='cfg', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

if __name__ == '__main__':

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    os.chdir(_init_paths.faster_rcnn_root)
    args=parse_args()
    scenario = Scenario().load(args.scenario_file)
    if scenario.config_path is not None:
        print 'loading config from ', scenario.config_path
        cfg_from_file(scenario.config_path)

    cfg.GPU_ID = scenario.gpu_id
    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    _init_caffe(cfg)


    output_dir = os.path.join(scenario.scen_dir, 'output')
    dir_exists_or_create(output_dir)
    train_fast_rcnn(
        imdb_name=scenario.train_imdb,
        init_model=args.weights,
        solver=scenario.models['stage1_fast_rcnn_solver'],
        cfg=cfg,
        rpn_file=args.roi_pkl,
        output_dir=output_dir,
        max_iters=scenario.max_iters[1]
    )