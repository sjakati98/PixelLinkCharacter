import glob
import math
import os
import sys
from optparse import OptionParser

import numpy as np
from scipy.optimize import linear_sum_assignment

from util import (getAnnotationsFromFile, ground_truth_to_image_anchor,
                  res_to_image_anchor)


def createGroundTruthDictionary(original_images_dir, ground_truth_annotations_dir):
    """
    Takes all cropped annotations by specified letter, and compiles them into a dictionary. (Handles crop offsets)
    Inputs:
        - original_images_dir:  Directory containing cropped images
        - ground_truth_annotations_dir: Directory containing ground truth annotations
    Outputs:
        - image_dict: Dictionary of format {
                (key) image_filename: (value) [annotation filename]
            }
    """
    ## get all the original filenames
    image_filenames = glob.glob(os.path.join(original_images_dir, "*.tiff"))
    ## get all the ground truth annotation filenames
    ground_truth_filenames = glob.glob(os.path.join(ground_truth_annotations_dir, "*.txt"))
    print("Number of ground truth files: ", len(ground_truth_filenames))

    ## create empty dictionary
    image_dict = {filename[:-5].split(os.sep)[-1]: [] for filename in image_filenames}

    ## iterate through ground truth annotation filenames
    for filename in ground_truth_filenames:
        ## marshal filename into components
        image_name, anchorX, anchorY = ground_truth_to_image_anchor(filename)
        ## if the base filename is in the dictionary, add the annotations into the dictionary
        if image_name in image_dict:
            curr_list = image_dict[image_name]
            ## extend the image ground truth annotation list with new annotations
            curr_list.extend(getAnnotationsFromFile(os.path.join(ground_truth_annotations_dir, filename), anchorX, anchorY))
            image_dict[image_name] = curr_list
        else:
            image_dict[image_name] = [getAnnotationsFromFile(os.path.join(ground_truth_annotations_dir, filename), anchorX, anchorY)]
    return image_dict

def createPredictedDictionary(original_images_dir, predicted_annotations_dir):
    """
    Takes all cropped annotations by specified letter, and compiles them into a dictionary. (Handles crop offsets)
    Inputs:
        - original_images_dir:  Directory containing cropped images
        - predicted_annotations_dir: Directory containing ground truth annotations
    Outputs:
        - image_dict: Dictionary of format {
                (key) image_filename: (value) [annotations]
            }
    """
    ## get all the original filenames
    image_filenames = glob.glob(os.path.join(original_images_dir, "*.tiff"))
    ## get all the ground truth annotation filenames
    predicted_filenames = glob.glob(os.path.join(predicted_annotations_dir, "*.txt"))
    print("Number of predicted annotation files: ", len(predicted_filenames))

    ## create empty dictionary
    image_dict = {filename[:-5].split(os.sep)[-1]: [] for filename in image_filenames}

    ## iterate through ground truth annotation filenames
    for filename in predicted_filenames:
        ## marshal filename into components
        image_name, anchorX, anchorY, angle = res_to_image_anchor(filename, True)
        ## if the base filename is in the dictionary, add the annotations into the dictionary
        if image_name in image_dict:
            curr_list = image_dict[image_name]
            ## extend the image ground truth annotation list with new annotations
            curr_list.extend(getAnnotationsFromFile(os.path.join(predicted_annotations_dir, filename), anchorX, anchorY, angle))
            image_dict[image_name] = curr_list
        else:
            image_dict[image_name] = [getAnnotationsFromFile(os.path.join(predicted_annotations_dir, filename), anchorX, anchorY, angle)]
    return image_dict


def generateIoUReport(calculated_iou_dictionary, outfile):
    """
    Creates a text file with the keys and values of the IoU values

    Inputs:
        - calculated_iou_dictionary: Dictionary of files with corresponding average IoU values
        - outfile: Filepath to where hte report should be written
    """

    with open(outfile, "w+") as f:
        for key in calculated_iou_dictionary:
                f.write("%s: %.5f" % (key, calculated_iou_dictionary[key]))

