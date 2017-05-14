#!/usr/bin/env python

import os
from os import listdir
from os.path import isfile, join, basename
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import re

def onlyFiles(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def _load_technicaldrawings_annotation(filename):
    """
    Load image and bounding boxes info from XML file in the technicaldrawings
    format.
    """
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        boxes[ix, :] = [x1, y1, x2, y2]

    return boxes

def merge(filename, bboxes, outpath):
    tree = ET.parse(filename)
    objs = tree.findall('object')
    root=tree.getroot()

    new_boxes=[]
    idx=0
    for ix, obj in enumerate(objs):
        if idx >= len(bboxes):
            break

        master_box = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(master_box.find('xmin').text)
        y1 = float(master_box.find('ymin').text)
        x2 = float(master_box.find('xmax').text)
        y2 = float(master_box.find('ymax').text)

        for sub_box in bboxes[idx]:
            print 'subbox:'
            print sub_box
            sub_box[0]+=x1
            sub_box[1]+=y1
            sub_box[2]+=x1
            sub_box[3]+=y1
            print sub_box
            new_boxes.append(sub_box)


        idx+=1

    for ix, obj in enumerate(objs):
        root.remove(obj)
    def str_elem(parent, childElemName, value):
        e=ET.SubElement(parent,childElemName)
        e.text=value
        return e

    for b in new_boxes:
        obj=ET.SubElement(root,'object')

        str_elem(obj, 'name', 'number')
        str_elem(obj, 'pose', 'Unspecified')
        str_elem(obj, 'difficult', '0')
        bndbox=ET.SubElement(obj,'bndbox')
        str_elem(bndbox, 'xmin', str(int(b[0])))
        str_elem(bndbox, 'ymin', str(int(b[1])))
        str_elem(bndbox, 'xmax', str(int(b[2])))
        str_elem(bndbox, 'ymax', str(int(b[3])))

    tree.write(join(outpath, basename(filename)))

techdraw_path='/home/dennis/workspace/datasets/technicaldrawings'
numbers_measurements_path=os.path.join(techdraw_path, 'numbers-and-measurements/annotations')
single_numbers_path=os.path.join(techdraw_path, 'single-numbers/annotations')
outpath=os.path.join(techdraw_path, 'numbers-and-measurements/annotations-new')

regex_numbers_measurements_filename = re.compile('([a-zA-Z\.]+-\d+)\.xml')
regex_single_numbers_filename = re.compile('([a-zA-Z\.]+-\d+)-(measurement|housenumbers)-(\d+)\.xml')

nm_and_sn={}

for sn in onlyFiles(numbers_measurements_path):
    match=regex_numbers_measurements_filename.match(sn)
    if match:
        nm_and_sn[match.group(1)]=[os.path.join(numbers_measurements_path,sn)]

for sn in onlyFiles(single_numbers_path):
    match=regex_single_numbers_filename.match(sn)
    if match:
        existing=nm_and_sn.get(match.group(1), None)
        if existing != None:
            existing.append(_load_technicaldrawings_annotation(os.path.join(single_numbers_path,sn)))

for k in nm_and_sn:
    values=nm_and_sn[k]
    if len(values)>1:
        print k
        print values
        merge(values[0], values[1:], outpath)


