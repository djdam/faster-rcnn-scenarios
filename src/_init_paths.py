# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Dennis Dam
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""
import os
from os.path import join, dirname
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = dirname(__file__)
caffe_root = os.environ['CAFFE_ROOT']
faster_rcnn_root = os.environ['FASTER_RCNN_ROOT']
lib_path = join(faster_rcnn_root, 'lib')

# Add caffe & faster-rcnn to PYTHONPATH
add_path(caffe_root)
# Add lib to PYTHONPATH
add_path(lib_path)
