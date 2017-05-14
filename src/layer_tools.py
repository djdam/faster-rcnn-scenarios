from caffe import layers as L
from caffe import params as P
import config
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

LRN_NORMREGION_WITHIN_CHANNEL = 1
LRN_ENGINE_CAFFE = 1

WEIGHT_FILLER=dict(type='gaussian', std=0.01)
BIAS_FILLER=dict(type='constant', value=0)


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param, weight_filler=None, bias_filler=None):
    conv_param=dict()
    if weight_filler != None:
        conv_param.update(dict(weight_filler=weight_filler))
    if bias_filler != None:
        conv_param.update(dict(bias_filler=bias_filler))
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, convolution_param=conv_param)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param):
    fc = L.InnerProduct(bottom, num_output=nout, param=param)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1, pad=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride, pad=pad)

def soft_max_with_loss(bottom):
    return L.SoftmaxWithLoss(bottom=bottom, propagate_down=[1,0], loss_weight=1, loss_param=dict(ignore_label=-1,normalize=True))

def reshape(bottom,dim):
    return L.Reshape(bottom, reshape_param=dict(shape=dict(dim=dim)))

def conv1_to_5(n, param):
    n.conv1, n.relu1 = conv_relu(n.data, ks=7, nout=96, stride=2, pad=3, param=param)
    n.norm1 = L.LRN(n.conv1, local_size=3, alpha=0.00005, beta=0.75, norm_region=LRN_NORMREGION_WITHIN_CHANNEL,
                   engine=LRN_ENGINE_CAFFE)

    n.pool1 = max_pool(n.norm1, 3, stride=2)

    n.conv2, n.relu2 = conv_relu(n.pool1, 5, 256, stride=2, pad=2, param=param)

    n.norm2 = L.LRN(n.conv2, local_size=3, alpha=0.00005, beta=0.75, norm_region=LRN_NORMREGION_WITHIN_CHANNEL,
                   engine=LRN_ENGINE_CAFFE)
    n.pool2 = max_pool(n.norm2, ks=3, stride=2, pad=1)
    n.conv3, n.relu3 = conv_relu(n.pool2, ks=3, nout=384, stride=1, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.conv3, ks=3, nout=384, stride=1, pad=1, param=param)
    n.conv5, n.relu5 = conv_relu(n.conv4, ks=3, nout=256, stride=1, pad=1, param=param)

def rpn_class_and_bbox_predictors(n, net, param):
    weight_filler = (WEIGHT_FILLER if net.train else dict())
    bias_filler = (BIAS_FILLER if net.train else dict())
    rpn_conv1, rpn_relu1 = conv_relu(n.conv5, ks=3, nout=256, stride=1, pad=1, param=param,
                                         weight_filler=weight_filler, bias_filler=bias_filler)
    rpn_cls_score = L.Convolution(rpn_conv1, kernel_size=1, stride=1,
                                    num_output=net.num_classes * net.nr_of_anchors(), pad=0,
                                    weight_filler=weight_filler,
                                    bias_filler=bias_filler, param=param)

    rpn_bbox_pred = L.Convolution(rpn_conv1, kernel_size=1, stride=1,
                                    num_output=4 * net.nr_of_anchors(), pad=0,
                                    weight_filler=weight_filler,
                                    bias_filler=bias_filler, param=param)

    return rpn_conv1, rpn_relu1, rpn_cls_score, rpn_bbox_pred

def anchor_params(feat_stride, scales, ratios):
    if config.EXTENDED_PY_FASTER_RCNN:
        return "{feat_stride: %s, scales: %s, ratios: %s}" % (str(feat_stride), str(scales),str(ratios))
    else:
        return "{feat_stride: %s, scales: %s}" % (str(feat_stride), str(scales))

def roi_proposal(n, net):
    rpn_cls_prob = L.Softmax(n.rpn_cls_score_reshape)
    rpn_cls_prob_reshape = reshape(rpn_cls_prob, [0, 2 * net.nr_of_anchors(), -1, 0]) # 2 = bg/fg
    rois = L.Python(
        bottom=["rpn_cls_prob_reshape", "rpn_bbox_pred", "im_info"],
        top=['scores'],
        python_param=dict(
            module='rpn.proposal_layer',
            layer='ProposalLayer',
            param_str=anchor_params(net.anchor_feat_stride, net.anchor_scales, net.anchor_ratios)
        ))

    return rpn_cls_prob, rpn_cls_prob_reshape, rois

def input():
    data=L.Input(input_param=dict(shape=dict(dim=[1, 3, 224, 224])))
    im_info = L.Input(input_param=dict(shape=dict(dim=[1, 3])))

    return data, im_info

def silence(bottom):
    layer=L.Silence(bottom)
    layer.fn.ntop = 0
    layer.fn.tops = ()
    return layer