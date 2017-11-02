#!/usr/bin/env python

import sys
from os.path import dirname, join
import numpy as np
this_dir = dirname(__file__)
if __name__ == '__main__':
    sys.path.insert(0, join(this_dir,'..'))
    import _init_paths
    import os
    os.chdir(_init_paths.faster_rcnn_root)


from datasets.factory import get_imdb

WARNING_AMOUNT_OF_BOXES=50

imdb=get_imdb("technicaldrawings_numbers_train")

resize_dim=600

class Image:

    def __init__(self):
        pass

def get_statistics(imdb):
    images=[]
    for boxes, width, height, filename in [(entry['boxes'], entry['width'], entry['height'], entry['filename']) for entry in imdb.gt_roidb()]:
        image=Image()
        image.filename=filename
        image.width=int(width)
        image.height=int(height)
        if width > height:
            image.dest_height = resize_dim
            image.scale_factor = resize_dim / float(image.height)
            image.dest_width = image.width * image.scale_factor
        else:
            image.dest_width = resize_dim
            image.scale_factor = resize_dim / float(image.width)
            image.dest_height = image.height * image.scale_factor

        image.boxes=[]
        image.resized_boxes=[]
        image.aspect_ratios=[]
        image.bbox_dims=[]
        for x1, y1, x2, y2 in boxes:
            image.boxes.append([x1,y1,x2,y2])

            x1_dest = int(x1 * image.scale_factor)
            y1_dest = int(y1 * image.scale_factor)
            x2_dest = int(x2 * image.scale_factor)
            y2_dest = int(y2 * image.scale_factor)
            image.resized_boxes.append([x1_dest,
                                        y1_dest,
                                        x2_dest,
                                        y2_dest
                                        ])
            image.bbox_dims.append((x2_dest-x1_dest, y2_dest-y1_dest))
            image.aspect_ratios.append(float((x2 - x1) / float(y2 - y1)))

        images.append(image)
    return images


stats = get_statistics(imdb)
warnings=[]
for im in stats:
    print 'image size: %s x %s  -> %s x %s'%(im.width,im.height, int(im.dest_width), int(im.dest_height))
    if len(im.boxes) > WARNING_AMOUNT_OF_BOXES :
        warnings.append("Warning: more than %d boxes: %s"%(WARNING_AMOUNT_OF_BOXES, im.filename))
    elif len(im.boxes)==0:
        warnings.append("Warning: zero boxes: %s" % (im.filename))
    for box_idx in range(0,len(im.boxes)):
        x1, y1, x2, y2 = im.boxes[box_idx]
        dest_x1, dest_y1, dest_x2, dest_y2 = im.resized_boxes[box_idx]
        aspect_ratio=im.aspect_ratios[box_idx]
        width = (dest_x2-dest_x1)
        height = (dest_y2-dest_y1)

        if (width <= 0 or height <= 0):
            warnings.append("ERROR: height or width is 0 for "+im.filename)
        if (width*height < 10):
            warnings.append("Warning: contains small box: "+im.filename)
        print "bbox: "
        print "    resize       : [%s,%s,%s,%s] -> [%s,%s,%s,%s]"%(x1,y1,x2,y2,dest_x1,dest_y1,dest_x2,dest_y2)
        print "    width       : %d"%(dest_x2-dest_x1)
        print "    height       : %d" % (dest_y2 - dest_y1)
        print "    aspect ratio : %.2f"%aspect_ratio

for msg in warnings:
    print 'warning:',msg

flatten = lambda l: [item for sublist in l for item in sublist]

dims= flatten([im.bbox_dims for im in stats])

aspect_ratios = flatten([im.aspect_ratios for im in stats])
bbox_widths = [w for w,h in dims ]
bbox_heights = [h for w,h in dims]
scale_factors = [im.scale_factor for im in stats]
widths=[im.width for im in stats]
heights=[im.height for im in stats]

print 'min im width: %.2f '%(min(widths))
print 'mean im width: %.2f '%(np.mean(widths))
print 'max im width: %.2f '%(max(widths))

print 'min im height: %.2f '%(min(heights))
print 'mean im height: %.2f '%(np.mean(heights))
print 'max im height: %.2f '%(max(heights))


print 'min scale_factors: %.2f '%(min(scale_factors))
print 'mean scale_factors: %.2f '%(np.mean(scale_factors))
print 'max scale_factors: %.2f '%(max(scale_factors))

print 'min bbox width: %.2f '%(min(bbox_widths))
print 'mean bbox width: %.2f '%(np.mean(bbox_widths))
print 'max bbox width: %.2f '%(max(bbox_widths))

print 'min bbox height: %.2f '%(min(bbox_heights))
print 'mean bbox height: %.2f '%(np.mean(bbox_heights))
print 'max bbox height: %.2f '%(max(bbox_heights))

print 'min aspect ratio: %.2f '%(min(aspect_ratios))
print 'mean aspect ratio: %.2f '%(np.mean(aspect_ratios))
print 'max aspect ratio: %.2f '%(max(aspect_ratios))
