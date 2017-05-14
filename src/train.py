#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Originally written by Rob Girshick. Adapted by Dennis Dam to
# use scenarios.
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import argparse
import numpy as np
import sys, os
import multiprocessing as mp
from multiprocessing import Pool, TimeoutError
import cPickle
import shutil
from scenario import Scenario
import pprint
import GPUtil

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('scenario_file',
                        help='Path scenario file (e.g. /home/user/scenario.p)')
    parser.add_argument('--gpus', dest='gpus',
                        help='Number of GPU cores)',
                        default=1, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name, rpn_file=None):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb

# ------------------------------------------------------------------------------
# Pycaffe doesn't reliably free GPU memory when instantiated nets are discarded
# (e.g. "del net" in Python code) . To work around this issue, each training
# stage is executed in a separate process using multiprocessing.Process.
# ------------------------------------------------------------------------------

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

def train_rpn(queue=None, imdb_name=None, init_model=None, solver=None,
              max_iters=None, cfg=None, output_dir=None):
    """Train a Region Proposal Network in a separate training process.
    """

    # Not using any proposals, just ground-truth boxes
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.BBOX_REG = False  # applies only to Fast R-CNN bbox regression
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name)
    print 'roidb len: {}'.format(len(roidb))
    if output_dir==None:
        output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    print 'len roidb=',len(roidb)
    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    rpn_model_path = model_paths[-1]
    # Send final model path through the multiprocessing queue
    queue.put({'model_path': rpn_model_path})

def rpn_generate_kw_wrapper(kwargs):
    return rpn_generate(**kwargs)

def rpn_generate(queue=None, imdb_name=None, rpn_model_path=None, cfg=None,
                 rpn_test_prototxt=None, output_dir=None, part_id=None):
    """Use a trained RPN to generate proposals.
    """

    cfg.TEST.RPN_PRE_NMS_TOP_N = -1     # no pre NMS filtering
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000  # limit top boxes after NMS
    print 'RPN model: {}'.format(rpn_model_path)
    print('Using config:')
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    # NOTE: the matlab implementation computes proposals on flipped images, too.
    # We compute them on the image once and then flip the already computed
    # proposals. This might cause a minor loss in mAP (less proposal jittering).
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

    # Load RPN and configure output directory
    rpn_net = caffe.Net(rpn_test_prototxt, rpn_model_path, caffe.TEST)
    if output_dir==None:
        output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Generate proposals on the imdb
    rpn_proposals = imdb_proposals(rpn_net, imdb)
    # Write proposals to disk and send the proposal file path through the
    # multiprocessing queue
    rpn_net_name = os.path.splitext(os.path.basename(rpn_model_path))[0]
    rpn_proposals_path = os.path.join(
        output_dir, rpn_net_name + ('_'+part_id if part_id != None else '')+'_proposals.pkl')
    with open(rpn_proposals_path, 'wb') as f:
        cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
    print 'Wrote RPN proposals to {}'.format(rpn_proposals_path)
    queue.put({'proposal_path': rpn_proposals_path, 'rpn_net': rpn_net_name})


def train_fast_rcnn(queue=None, imdb_name=None, init_model=None, solver=None,
                    max_iters=None, cfg=None, rpn_file=None, output_dir=None):
    """Train a Fast R-CNN using proposals generated by an RPN.
    """

    cfg.TRAIN.HAS_RPN = False           # not generating prosals on-the-fly
    cfg.TRAIN.PROPOSAL_METHOD = 'rpn'   # use pre-computed RPN proposals instead
    cfg.TRAIN.IMS_PER_BATCH = 2
    print 'Init model: {}'.format(init_model)
    print 'RPN proposals: {}'.format(rpn_file)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name, rpn_file=rpn_file)
    if output_dir==None:
        output_dir = get_output_dir(imdb)

    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Train Fast R-CNN
    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    fast_rcnn_model_path = model_paths[-1]
    # Send Fast R-CNN model path over the multiprocessing queue
    queue.put({'model_path': fast_rcnn_model_path})

def dir_exists_or_create(path):
    # create scenario dir if it not exists
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Guard against race condition
            pass


def join_pkls(proposal_paths, output_dir, rpn_net_name):
    rpn_proposals=[]
    for ppath in proposal_paths:
        f= open(ppath, 'r')
        rpn_proposals+=cPickle.load(f)

        rpn_proposals_path=os.path.join(output_dir, rpn_net_name+'_proposals.pkl')
    with open(rpn_proposals_path, 'wb') as f:
        cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
    return rpn_proposals_path


if __name__ == '__main__':

    # first change current dir to py-faster-rcnn dir, or else scripts will break:
    os.chdir(_init_paths.faster_rcnn_root)
    print dir(mp)
    args = parse_args()

    print(args)

    scenario=Scenario().load(args.scenario_file)

    print "Using scenario:"
    pprint.pprint(scenario.__dict__)

    output_dir = os.path.join(scenario.scen_dir, 'output')
    dir_exists_or_create(output_dir)

    cfg.GPU_ID = scenario.gpu_id

    # --------------------------------------------------------------------------
    # Pycaffe doesn't reliably free GPU memory when instantiated nets are
    # discarded (e.g. "del net" in Python code). To work around this issue, each
    # training stage is executed in a separate process using
    # multiprocessing.Process.
    # --------------------------------------------------------------------------

    # queue for communicated results between processes
    mp_queue = mp.Queue()
    # solves, iters, etc. for each training stage
    print '#'*25
    print 'USING SCENARIO:'
    print scenario.scenario
    print '#'*25
    max_iters = scenario.max_iters
    cpu_count=mp.cpu_count()
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    if scenario.config_path is not None:
        cfg_from_file(scenario.config_path)
    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=scenario.train_imdb,
            init_model=scenario.weights_path,
            solver=scenario.models['stage1_rpn_solver'],
            max_iters=max_iters[0],
            cfg=cfg, output_dir=output_dir)
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    multi_gpu=False # disable for now
    if multi_gpu:


        parts = min(len(GPUtil.getGPUs()),cpu_count)

        print 'Number of parts is',parts
        pool=Pool(processes=parts)

        def gpu_conf(cfg, gpu_id=None):

            if gpu_id==None:
                DEVICE_ID_LIST = GPUtil.getFirstAvailable()
                if (len(DEVICE_ID_LIST) > 0):
                    cfg.GPU_ID = DEVICE_ID_LIST[0]  # grab first element from list
            else:
                cfg.GPU_ID=gpu_id

            return cfg

        configs=[
            dict(
                imdb_name='%s_part_%dof%d' % (scenario.train_imdb, part_id, parts),
                rpn_model_path=str(rpn_stage1_out['model_path']),
                cfg=gpu_conf(cfg, part_id-1),
                rpn_test_prototxt=scenario.models['rpn_test'],
                output_dir=output_dir,
                part_id=part_id

            ) for part_id in range(1,parts+1)
        ]
        pprint.pprint(configs)
        results=pool.map(rpn_generate_kw_wrapper, configs)

        # rpn_net = ''
        # for p in processes:
        #     p.start()
        #     passed_vars = mp_queue.get()
        #     rpn_net = passed_vars['rpn_net']
        #     proposal_paths.append(passed_vars['proposal_path'])
        #
        # for p in processes:
        #     p.join()
        #
        # aggregated_proposal_path = join_pkls(proposal_paths, output_dir, rpn_net)

    else:
        mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=scenario.train_imdb,
            rpn_model_path=str(rpn_stage1_out['model_path']),
            cfg=cfg,
            rpn_test_prototxt=scenario.models['rpn_test'],
            output_dir=output_dir
        )
        p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
        p.start()
        rpn_stage1_out['proposal_path'] = mp_queue.get()['proposal_path']
        p.join()

        # processes=[]
        # proposal_paths=[]
        # base_dict=
        # for i in range(0, parts):
        #
        #
        #     part_id='%dof%d'%(i+1, parts)
        #     mp_kwargs = dict(
        #             queue=mp_queue,
        #             imdb_name= '%s_part_%s'%(scenario.train_imdb, part_id),
        #             rpn_model_path=str(rpn_stage1_out['model_path']),
        #             cfg=cfg,
        #             rpn_test_prototxt=scenario.models['rpn_test'],
        #             output_dir = output_dir,
        #             part_id=part_id
        #
        #     )
        #
        #     processes.append(mp.Process(target=rpn_generate, kwargs=mp_kwargs))


    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=scenario.train_imdb,
            init_model=scenario.weights_path,
            solver=scenario.models['stage1_fast_rcnn_solver'],
            max_iters=max_iters[1],
            cfg=cfg,
            rpn_file=aggregated_proposal_path, output_dir=output_dir)
    p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
    p.start()
    fast_rcnn_stage1_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, init from stage 1 Fast R-CNN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=scenario.train_imdb,
            init_model=str(fast_rcnn_stage1_out['model_path']),
            solver=scenario.models['stage2_rpn_solver'],
            max_iters=max_iters[2],
            cfg=cfg, output_dir=output_dir)
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=scenario.train_imdb,
            rpn_model_path=str(rpn_stage2_out['model_path']),
            cfg=cfg,
            rpn_test_prototxt=scenario.models['rpn_test'], output_dir=output_dir)


    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out['proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=scenario.train_imdb,
            init_model=str(rpn_stage2_out['model_path']),
            solver=scenario.models['stage2_fast_rcnn_solver'],
            max_iters=max_iters[3],
            cfg=cfg,
            rpn_file=rpn_stage2_out['proposal_path'], output_dir=output_dir)
    p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
    p.start()
    fast_rcnn_stage2_out = mp_queue.get()
    p.join()

    # Create final model (just a copy of the last stage)
    final_path = scenario.net_final_path
    print 'cp {} -> {}'.format(
            fast_rcnn_stage2_out['model_path'], final_path)
    shutil.copy(fast_rcnn_stage2_out['model_path'], final_path)
    print 'Final model: {}'.format(final_path)
