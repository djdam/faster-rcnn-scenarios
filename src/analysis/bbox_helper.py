#!/usr/bin/env python
import os
import Image
import argparse
import cv2
from os.path import join, basename

# Bounding Box Helper
class BBoxHelper:
    def __init__(self, imdb_entry, filename=None, scaled_img=None):
        self.rects=imdb_entry['boxes']
        self.scaled_img=scaled_img
        if filename == None:
            self.annotation_filename = imdb_entry['filename']
        else:
            self.annotation_filename=filename

        dir, filename = os.path.split(self.annotation_filename)
        base=os.path.splitext(basename(filename))[0]
        img_no_ext=join(dir,'..','images', base)
        for ext in ['.png', '.jpg', '.jpeg', '.JPEG']:
            if os.path.isfile(img_no_ext+ext):
                self.imgPath=img_no_ext+ext
                break

    def saveBoundingBoxesToImage(self, boxes, outputFolder):
        print self.imgPath

        im=cv2.imread(self.imgPath)
        if self.scaled_img != None:
            width, height=self.scaled_img[:2]
            im=cv2.resize(im, (int(height),int(width)), interpolation=cv2.INTER_CUBIC)

        if (len(boxes) == 0):
            print 'Warning'
        for rectIdx in range(0, len(boxes)):

            x1,y1,x2,y2=boxes[rectIdx]
            color=(255,0,0)
            im=cv2.rectangle(im, (int(x1),int(y1)), (int(x2), int(y2)), color=color, thickness=3)

        cv2.imwrite(join(outputFolder, basename(self.imgPath)), im)

    def saveBoundBoxImage(self, imgPath=None, outputFolder=None):
            if imgPath is not None:
                self.imgPath = imgPath

            if imgPath is None and self.imgPath is None:
                self.imgPath = self.findImagePath()

            if outputFolder == None:
                outputFolder = join(self.wnid, 'bounding_box_imgs')

            # annotation_file_dir = os.path.dirname(os.path.realpath(self.annotation_file))
            # outputFolder = join(annotation_file_dir, savedTargetDir)
            if not os.path.exists(outputFolder):
                os.mkdir(outputFolder)

            # Get crop images
            bbs = []
            im = Image.open(self.imgPath)
            for box in self.rects:
                 bbs.append(im.crop(box))
       # Save them to target dir
            count = 0
            for box in bbs:
                    count = count + 1
                    outPath = str(join(outputFolder, self.annotation_filename + '_box' + str(count) + '.JPEG'))
                    box.save(outPath)
                    print 'save to ' + outPath

    def get_BoudingBoxs(self):
        return self.rects

        def getWnid(self):
            return self.wnid

    def findImagePath(self, search_folder='.'):
        filename = self.annotation_filename + str('.jpg')
        for root, dirs, files in os.walk(search_folder):
            for file in files:
                if filename == file:
                    return join(root, file)
        print filename + ' not found'
        return None


def saveAsBoudingBoxImg(xmlfile):
    bbhelper = BBoxHelper(xmlfile)
    print bbhelper.findImagePath('/home/dennis/workspace/datasets/technicaldrawings/JPEGImages')
    # Search image path according to bounding box xml, and crop it
    if shouldSaveBoundingBoxImg:
        print bbhelper.get_BoudingBoxs()
        bbhelper.saveBoundBoxImage(outputFolder='/home/dennis/workspace/datasets/technicaldrawings/bboxes')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Help the user to download, crop, and handle images from ImageNet')
    p.add_argument('--bxmlpath', help='Boudingbox xml path')
    p.add_argument('--bxmldir', help='Boudingbox dir path')
    p.add_argument('--save_boundingbox', help='Search images and crop the bounding box by image paths', action='store_true', default=False)
    args = p.parse_args()
    # Give bounding_box XML and show its JPEG path and bounding rects
    boundingbox_xml_file = args.bxmlpath
    boudingbox_xml_dir = args.bxmldir
    shouldSaveBoundingBoxImg = args.save_boundingbox

    if not boundingbox_xml_file is None:
        saveAsBoudingBoxImg(boundingbox_xml_file)

    if not boudingbox_xml_dir is None:
        allAnnotationFiles = scanAnnotationFolder(boudingbox_xml_dir)
        for xmlfile in allAnnotationFiles:
            saveAsBoudingBoxImg(xmlfile)
