#!/usr/bin/env python
import os
import Image
import argparse
import cv2
from os.path import join, basename, isfile
from os import listdir
import numpy as np

def onlyFiles(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

img_count=0
sum=np.zeros((3))

dir = '/home/dennis/workspace/datasets/technicaldrawings/numbers/images'
img_name='steenwijkerland_1316.pdf-0.png'

for f in onlyFiles(dir):

    try:
        img=np.transpose(np.array(cv2.imread(join(dir,f))), (2,0,1))

        # print img
        sum+=np.average(np.average(img, axis=2), axis=1)
        img_count+=1
    except:
        print 'warning'

print sum/img_count
