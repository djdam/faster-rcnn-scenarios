#!/usr/bin/env python


import network
import argparse
from network import RegionProposalNetworkConfig as RpnConfig, FastRcnnNetworkConfig as FastRcnnConfig
from solver import SolverConfig

from os import listdir, chmod
from os.path import isfile, join, dirname, abspath
from caffe_config import CaffeConfig
from scenario_bash_script import SCRIPT as bash
import _init_paths
import draw.plot as plot

this_dir = dirname(abspath(__file__))

import yaml
from to_string import to_string

def onlyFiles(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

class Scenario(CaffeConfig):
    rpn_config=None
    fast_rcnn_config=None
    def __init__(self, scenarios_dir=None, scenario=None, train_imdb=None, test_imdb=None, weights_path=None, gpu_id=None,
                 max_iters=None, rpn_config=None, fast_rcnn_config=None, solver_config=None, config=dict()):
        CaffeConfig.__init__(self, scenarios_dir, scenario)

        self.gpu_id=gpu_id
        self.rpn_config=rpn_config
        self.fast_rcnn_config=fast_rcnn_config
        self.solver_config=solver_config
        self.train_script=join(this_dir, 'train.py')

        if self.rpn_config != None:
            self.rpn_config.setScenario(scenarios_dir, scenario)
        if self.fast_rcnn_config != None:
            self.fast_rcnn_config.setScenario(scenarios_dir, scenario)

        self.train_imdb=train_imdb
        self.test_imdb=test_imdb
        self.models = {}
        self.max_iters=max_iters
        self.weights_path=weights_path
        self.config=config

    def name(self, scenario):
        if scenario != None and self.scenarios_dir != None:
            self.setScenario(self.scenarios_dir, scenario)

    def setScenario(self,scenarios_dir, scenario):
        CaffeConfig.setScenario(self, self.scenarios_dir, scenario)
        if self.rpn_config != None:
            self.rpn_config.setScenario(self.scenarios_dir, self.scenario)
        if self.fast_rcnn_config != None:
            self.fast_rcnn_config.setScenario(self.scenarios_dir, self.scenario)

    def path(self):
        return join(self.scen_dir,'scenario.yml')

    def save(self):
        self.config_path = join(self.scenarios_dir, self.scenario, 'cfg.yml')
        self.script_path=script_path=join(self.scen_dir, 'start.sh')
        self.net_final_path = join(self.scen_dir, 'output', 'final.caffemodel')

        # save scenario configuration
        settings = dict(scenarios_dir=self.scenarios_dir, scenario=self.scenario, scen_dir=self.scen_dir, train_imdb=self.train_imdb,
                 test_imdb=self.test_imdb, models=self.models, gpu_id=self.gpu_id, max_iters=self.max_iters,
                 weights_path=self.weights_path, config_path=self.config_path,train_script=self.train_script,
                        net_final_path=self.net_final_path)
        yaml.dump(settings,  open(self.path(), "wb"), default_flow_style=False)

        f = open(script_path, 'wb')

        # save configuration. config must be a dict !
        yaml.dump(self.config, open(self.config_path, "wb"), default_flow_style=False)

        # save start script
        script_settings=dict()
        script_settings.update(settings)
        script_settings['testproto']=self.models['fast_rcnn_test']
        script_settings['scenario_file']=self.path()
        script_settings['py_faster_rcnn']=_init_paths.faster_rcnn_root
        script_settings['plot_script']=abspath(plot.__file__)
        f.write(bash.format(**script_settings))
        chmod(script_path, 0755)


    def load(self, path_to_yaml):
        settings=yaml.load(open(path_to_yaml, "rb"))

        for key in settings:
            setattr(self, key, settings[key])

        self.config=yaml.load(open(self.config_path, 'rb'))

        return self

    def generate(self):
        self.empty_dir()

        rpn=self.rpn_config
        fast_rcnn=self.fast_rcnn_config
        solver_config=self.solver_config

        # stage 1
        rpn.stage = fast_rcnn.stage = 1
        rpn.train = fast_rcnn.train = True
        rpn.conv_1_to_5_learn = fast_rcnn.conv_1_to_5_learn = True

        self.models['stage1_rpn']=rpn.generate()
        self.models['stage1_fast_rcnn']=fast_rcnn.generate()

        solver_config.snapshot_prefix='rpn'
        self.models['stage1_rpn_solver']=solver_config.generate(rpn)
        solver_config.snapshot_prefix='fastrcnn'
        self.models['stage1_fast_rcnn_solver']=solver_config.generate(fast_rcnn)

        # stage 2 : freeze conv 1-5
        rpn.stage = fast_rcnn.stage = 2
        rpn.conv_1_to_5_learn = fast_rcnn.conv_1_to_5_learn = False

        self.models['stage2_rpn'] =rpn.generate()
        self.models['stage2_fast_rcnn'] =fast_rcnn.generate()

        solver_config.snapshot_prefix='rpn'
        self.models['stage2_rpn_solver'] =solver_config.generate(rpn)
        solver_config.snapshot_prefix='fast_rcnn'
        self.models['stage2_fast_rcnn_solver'] =solver_config.generate(fast_rcnn)

        rpn.train = fast_rcnn.train = False

        # generate test nets
        self.models['rpn_test']=rpn.generate()
        self.models['fast_rcnn_test']=fast_rcnn.generate()

        self.save()

    def __repr__(self):
        return to_string(self)

def parse_args():
    description = ('Split a Caffe training log into separate files per training run')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('scenarios_dir', help='Path to scenarios dir')
    parser.add_argument('scenario', help='Name of scenario')
    parser.add_argument('train_imdb', help='Name of train IMDB')
    parser.add_argument('test_imdb', help='Name of test IMDB')
    parser.add_argument('--weights', help='Name of test IMDB', default="data/imagenet_models/ZF.v2.caffemodel")
    parser.add_argument('--stage1_rpn_iters', help='#iters stage 1 RPN', default=1, type=int)
    parser.add_argument('--stage1_fast_rcnn_iters', help='#iters stage 1 Fast-RCNN', default=1, type=int)
    parser.add_argument('--stage2_rpn_iters', help='#iters stage 2 RPN', default=1, type=int)
    parser.add_argument('--stage2_fast_rcnn_iters', help='#iters stage 2 Fast-RCNN', default=1, type=int)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)

    args = parser.parse_args()
    return args

def main():
    # calling the network.py module directly will generate the default Faster-RCNN config.
    # Use generate() from another module for customization
    args=parse_args()
    print 'Generating scenario %s in folder %s'%(args.scenario, args.scenarios_dir)
    scenario=Scenario(
        args.scenarios_dir,
        args.scenario,
        args.train_imdb,
        args.test_imdb,
        args.weights,
        args.gpu_id,
        [args.stage1_rpn_iters,args.stage1_fast_rcnn_iters,args.stage2_rpn_iters, args.stage2_fast_rcnn_iters], # max iters
        RpnConfig(num_classes=2, anchor_scales=[8,16,32], anchor_feat_stride=16),
        FastRcnnConfig(num_classes=2),
        SolverConfig()
    )

    scenario.generate()

    scenario.save()

if __name__ == '__main__':
    main()