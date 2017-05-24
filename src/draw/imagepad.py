#!/usr/bin/env python

import cv2
import os
from os import listdir
from os.path import isfile, join, basename
import numpy as np

input_dir='/home/dennis/workspace/py-faster-rcnn/data/technicaldrawings/numbers/images'
output_dir='/home/dennis/workspace/py-faster-rcnn/data/technicaldrawings/numbers/images_padded'

def isImage(path):
    if not isfile(join(path)):
        return False
    base, ext = os.path.splitext(basename(path))[0]
    return ext in ['.png', '.jpg', '.jpeg', '.JPEG']

def onlyImages(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def pad(im):
    print im.shape
    height, width, channels=im.shape
    max_dim=max(height, width)+1
    print max_dim
    new_im = np.empty((max_dim,max_dim, channels))
    print new_im.shape
    new_im.fill(255)

    new_im[0:im.shape[0],0:im.shape[1],:]=im
    return new_im

for im_path in onlyImages(input_dir):
    im_path=join(input_dir, im_path)
    im=cv2.imread(im_path)
    cv2.imwrite(join(output_dir, basename(im_path)),pad(im))


