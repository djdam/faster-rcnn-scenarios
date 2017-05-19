#!/usr/bin/env python
import cPickle
import numpy as np
from bbox_helper import BBoxHelper
import sys
from os.path import dirname, join
this_dir = dirname(__file__)
if __name__ == '__main__':
    sys.path.insert(0, join(this_dir,'..'))
    import _init_paths
    import os
    os.chdir(_init_paths.faster_rcnn_root)

from datasets.factory import get_imdb

pkl_file='/home/dennis/workspace/data/broad_range_of_scales/output/_stage2_iter_10000_proposals.pkl'

imdb=get_imdb("technicaldrawings_numbers_train")
roi_db=imdb.gt_roidb()
for im_idx in range(0,20):
    bboxes=cPickle.load(file(pkl_file, 'r'))[im_idx]



    first_img=roi_db[im_idx]

    bbox_helper = BBoxHelper(first_img)
    print len(bboxes)
    # bboxes=[ [x1,y1,x2,y2] for y1,x1, y2, x2 in bboxes]
    bbox_helper.saveBoundingBoxesToImage(bboxes,'/home/dennis/workspace/faster-rcnn-scenarios/src/analysis/output')

