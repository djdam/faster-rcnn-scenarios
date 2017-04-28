#!/usr/bin/env python
# import tempfile
import caffe
from caffe import layers as L
import os
import argparse
import layer_tools as LT
from caffe_config import CaffeConfig
import pprint
from to_string import to_string

RPN="rpn"
FAST_RCNN="fast_rcnn"

class NetworkConfig(CaffeConfig):
    def __init__(self, network_type, scenarios_dir=None, scenario=None, stage=1, train=True, num_classes=1000, conv_1_to_5_learn=False,
                 anchor_feat_stride=16,anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1, 2]):
        CaffeConfig.__init__(self, scenarios_dir, scenario)
        self.network_type=network_type
        self.scenarios_dir=scenarios_dir
        self.scenario=scenario
        self.train=train
        self.num_classes=num_classes
        self.conv_1_to_5_learn=conv_1_to_5_learn
        self.stage=stage
        self.anchor_feat_stride = anchor_feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios

    def path(self):
        if not self._scenario_check():
            raise Exception("No scenarios dir and/or scenario set!")
        if self.train:
            return os.path.join(self.scen_dir, "stage%d_%s_train.prototxt" % (self.stage, self.network_type))

        return os.path.join(self.scen_dir, "%s_test.prototxt" % self.network_type)

    def save(self, network):
        return CaffeConfig.save(self, network.to_proto())

    def nr_of_anchors(self):
        return len(self.anchor_scales) * len(self.anchor_ratios)

    def __repr__(self):
        return to_string(self)


class RegionProposalNetworkConfig(NetworkConfig):
    def __init__(self, scenarios_dir=None, scenario=None, stage=1,  train=True, num_classes=1000, conv_1_to_5_learn=False,
                 anchor_feat_stride=16, anchor_scales=[8,16,32], anchor_ratios=[0.5, 1, 2]):
        NetworkConfig.__init__(self, RPN, scenarios_dir, scenario, stage, train, num_classes, conv_1_to_5_learn, anchor_feat_stride, anchor_scales, anchor_ratios)


    def generate(self):
        conf = self
        n = caffe.NetSpec()
        param = LT.learned_param if conf.train else LT.frozen_param

        if conf.train:
            n.data = L.Python(
                top=["im_info", 'gt_boxes'],
                python_param=dict(
                    module='roi_data_layer.layer',
                    layer='RoIDataLayer',
                    param_str="num_classes: " + str(conf.num_classes)
                ))
        else:
            n.data, n.im_info = LT.input()
        conv15_param = LT.learned_param if (conf.conv_1_to_5_learn) else LT.frozen_param
        LT.conv1_to_5(n, conv15_param)
        n.rpn_conv1, n.rpn_relu1, n.rpn_cls_score, n.rpn_bbox_pred = LT.rpn_class_and_bbox_predictors(n, self, param)
        n.rpn_cls_score_reshape = LT.reshape(n.rpn_cls_score, [0, 2, -1, 0])

        if conf.train:
            n.rpn_labels = L.Python(
                bottom=["rpn_cls_score", "gt_boxes", "im_info", "data"],
                top=['rpn_bbox_targets', "rpn_bbox_inside_weights", "rpn_bbox_outside_weights"],
                python_param=dict(
                    module='rpn.anchor_target_layer',
                    layer='AnchorTargetLayer',
                    param_str=LT.anchor_params(self.anchor_feat_stride, self.anchor_scales, self.anchor_ratios)
                ))
            n.loss_cls = LT.soft_max_with_loss(["rpn_cls_score_reshape", "rpn_labels"])
            n.loss_bbox = L.SmoothL1Loss(
                bottom=["rpn_bbox_pred", "rpn_bbox_targets", "rpn_bbox_inside_weights", "rpn_bbox_outside_weights"],
                loss_weight=1)
            # dummy RCNN layers
            n.dummy_roi_pool_conv_5 = L.DummyData(dummy_data_param=dict(
                shape=dict(dim=[1, 9216]), data_filler=LT.WEIGHT_FILLER
            ))
            n.fc6, n.relu6 = LT.fc_relu(n.dummy_roi_pool_conv_5, 4096, param=LT.frozen_param)
            n.fc7 = L.InnerProduct(n.fc6, num_output=4096, param=LT.frozen_param)
            n.silence_fc7 = LT.silence(n.fc7)
        else:
            n.rpn_cls_prob, n.rpn_cls_prob_reshape, n.rois = LT.roi_proposal(n, self)

        return self.save(n)

class FastRcnnNetworkConfig(NetworkConfig):
    def __init__(self, scenarios_dir=None, scenario=None, stage=1, train=True, num_classes=1000, conv_1_to_5_learn=False,
                 anchor_feat_stride=16, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1, 2]):
        NetworkConfig.__init__(self, FAST_RCNN, scenarios_dir, scenario, stage, train, num_classes, conv_1_to_5_learn, anchor_feat_stride, anchor_scales, anchor_ratios)

    def generate(self):
        """Returns a NetSpec specifying CaffeNet, following the original proto text
               specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
        conf = self
        n = caffe.NetSpec()
        param = LT.learned_param if conf.train else LT.frozen_param

        if self.train:

            n.data = L.Python(
                top=["rois", 'labels', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights'],
                python_param=dict(
                    module='roi_data_layer.layer',
                    layer='RoIDataLayer',
                    param_str="num_classes: " + str(conf.num_classes)
                ))
        else:
            n.data, n.im_info = LT.input()

        conv15_param = LT.learned_param if (conf.conv_1_to_5_learn) else LT.frozen_param
        LT.conv1_to_5(n, conv15_param)

        if not(self.train):
            n.rpn_conv1, n.rpn_relu1, n.rpn_cls_score, n.rpn_bbox_pred = LT.rpn_class_and_bbox_predictors(n, self, param)
            n.rpn_cls_score_reshape = LT.reshape(n.rpn_cls_score, [0, 2, -1, 0])
            n.rpn_cls_prob, n.rpn_cls_prob_reshape, n.rois = LT.roi_proposal(n, self)

        n.roi_pool = L.ROIPooling(bottom=["conv5", "rois"], pooled_w=6, pooled_h=6, spatial_scale=0.0625)

        n.fc6, n.relu6 = LT.fc_relu(n.roi_pool, 4096, param=param)

        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True, dropout_ratio=0.5, scale_train=False)
        n.fc7, n.relu7 = LT.fc_relu(fc7input, 4096, param=param)
        n.drop7 = layer7 = L.Dropout(n.relu7, in_place=True, dropout_ratio=0.5, scale_train=False)
        weight_filler = (LT.WEIGHT_FILLER if conf.train else dict())
        bias_filler = (LT.BIAS_FILLER if conf.train else dict())
        n.cls_score = L.InnerProduct(layer7, num_output=conf.num_classes,
                                     weight_filler=weight_filler,
                                     bias_filler=bias_filler, param=LT.learned_param)

        n.bbox_pred = L.InnerProduct(layer7, num_output=conf.num_classes * 4,
                                     weight_filler=weight_filler,
                                     bias_filler=bias_filler, param=LT.learned_param)

        if conf.train:
            n.loss_cls = LT.soft_max_with_loss(["cls_score", "labels"])
            n.loss_bbox = L.SmoothL1Loss(
                bottom=["bbox_pred", "bbox_targets", "bbox_inside_weights", "bbox_outside_weights"], loss_weight=1)
        else:
            n.cls_prob = L.Softmax(n.cls_score, loss_param=dict(ignore_label=-1, normalize=True))

        if self.train:
            n.rpn_conv1, n.rpn_relu1, n.rpn_cls_score, n.rpn_bbox_pred=LT.rpn_class_and_bbox_predictors(n, self, LT.frozen_param)

        n.silence_rpn_cls_score = LT.silence(n.rpn_cls_score)
        n.silence_rpn_bbox_pred =  LT.silence(n.rpn_bbox_pred)
        # write the net to a temporary file and return its filename
        return self.save(n)

def generate(rpn_config, fast_rcnn_config):
    # stage 1
    rpn_config.stage             = fast_rcnn_config.stage               = 1
    rpn_config.train             = fast_rcnn_config.train               = True
    rpn_config.conv_1_to_5_learn = fast_rcnn_config.conv_1_to_5_learn   = True

    rpn_config.generate()
    fast_rcnn_config.generate()

    # stage 2 : freeze conv 1-5
    rpn_config.stage             = fast_rcnn_config.stage               = 2
    rpn_config.conv_1_to_5_learn = fast_rcnn_config.conv_1_to_5_learn   = False

    rpn_config.generate()
    fast_rcnn_config.generate()

    rpn_config.train             = fast_rcnn_config.train               = False

    # generate test nets
    rpn_config.generate()
    fast_rcnn_config.generate()


def parse_args():
    description = ('Split a Caffe training log into separate files per training run')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('scenarios_path', help='Path to scenarios folder')
    parser.add_argument('scenario', help='Path to scenario folder')

    args = parser.parse_args()
    return args

def main():
    # calling the network.py module directly will generate the default Faster-RCNN config.
    # Use generate() from another module for customization
    args=parse_args()

    generate(
        RegionProposalNetworkConfig(scenarios_dir=args.scenarios_path, scenario=args.scenario, num_classes=2, anchor_scales=[8,16,32], anchor_feat_stride=16),
        FastRcnnNetworkConfig(scenarios_dir=args.scenarios_path, scenario=args.scenario, num_classes=2)
    )

if __name__ == '__main__':
    main()

run=0