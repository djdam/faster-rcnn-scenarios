#!/usr/bin/env python
import cPickle
import numpy as np
from bbox_helper import BBoxHelper
import sys
from os.path import dirname, join
from os.path import basename

this_dir = dirname(__file__)
if __name__ == '__main__':
    sys.path.insert(0, join(this_dir,'..'))
    import _init_paths
    import os
    os.chdir(_init_paths.faster_rcnn_root)

from datasets.factory import get_imdb

pkl_file='/home/dennis/workspace/faster-rcnn-scenarios/private/scenarios/feat_stride_8/output/rpn_stage1_iter_30000_proposals.pkl'

train_gt_roidb_pkl_file='/home/dennis/workspace/faster-rcnn-scenarios/src/train__gt_roidb.pkl'
cache=cPickle.load(open(train_gt_roidb_pkl_file, 'r'))

def getBasenameNoExt(filename):
   return os.path.splitext(basename(filename))[0]

entry_dict={}
for roi_entry in cache:
    entry_dict[getBasenameNoExt(roi_entry['filename'])]=roi_entry

pkl=cPickle.load(open(pkl_file, 'r'))

for im_idx in range(0,20):
    cached_img=cache[im_idx]
    boxes=pkl[im_idx]
    bbox_helper = BBoxHelper(cached_img)
    # bboxes=[ [x1,y1,x2,y2] for y1,x1, y2, x2 in bboxes]
    bbox_helper.saveBoundingBoxesToImage(boxes,'/home/dennis/workspace/faster-rcnn-scenarios/src/analysis/output')

