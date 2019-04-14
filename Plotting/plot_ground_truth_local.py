import os
import re
import sys
from glob import glob

import cv2
import numpy as np

from PIL import Image, ImageDraw

def list_crops_to_annotated_image(original_image, annotations, outfile):
    """
        Inputs:
            - original_image: The image which will be copied and have annotations drawn on it
            - annotations: The annotation filepaths that need to be translated and drawn on the image copy
            - outfile: Where the drawn on image should be saved
    """
    print("Stitching Image: ", original_image)
    image = Image.open(original_image)
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        print("Considering annotations from:", annotation)
        anchor_x0 = anchor_y0 = 0
        for line in open(annotation).readlines():
            gt = line.split(',')
            oriented_box = [int(gt[i]) for i in range(8)]
            print("Drawing Box: ", oriented_box)
            draw.polygon(oriented_box, outline="blue", fill=None)
            #draw.rectangle([oriented_box[6] + anchor_x0, oriented_box[7] + anchor_y0, oriented_box[2] + anchor_x0, oriented_box[3] + anchor_y0], outline='red')
    del draw
    image.save(outfile)
    print("Image Saved: ", outfile)

def convert_npy_to_text_file(npy_filepath):
    """
        Inputs:
            - npy_filepath: The ground truth annotations
        Outputs:
            - text_filepath: The filepath of the ground truth annotations in one line format
    """
    outfile = "/Users/shishir/Desktop/MapRecoFiles/TranslationTest/GroundTruth/annotated_ground_truth.txt"

    ## need to reshape the numpy array
    gt = np.load(npy_filepath)
    gt = gt.reshape(-1)
    gt = gt[0]
    ## write all the bounding boxes into a text file with the 8 point, one line format
    with open(outfile, 'w+') as f:
        for i in range(len(gt)):
            box = gt[i]
            box_points = []
            for p_1, p_2 in box['vertices']:
                box_points.append(p_1)
                box_points.append(p_2)

            box_line = ", ".join(str(int(x)) for x in box_points)
            f.write(box_line + "\n")
    return outfile

convert_npy_to_text_file("/Users/shishir/Desktop/MapRecoFiles/TranslationTest/GroundTruth/D0042-1070005.npy")
list_crops_to_annotated_image("/Users/shishir/Desktop/MapRecoFiles/TranslationTest/maps/D0042-1070005.tiff", ["/Users/shishir/Desktop/MapRecoFiles/TranslationTest/GroundTruth/annotated_ground_truth.txt"], "/Users/shishir/Desktop/MapRecoFiles/TranslationTest/maps/changed.jpeg")