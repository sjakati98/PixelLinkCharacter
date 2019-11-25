import glob
import math
import os
import sys
from optparse import OptionParser

import numpy as np
from scipy.optimize import linear_sum_assignment

from curve_utils import marshal_thresholded_dictionary, subplot_image, subplot_curve
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

def createPredictedDictionary(original_images_dir, predicted_annotations_dir, east=False):
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
        # image_name, anchorX, anchorY, angle = res_to_image_anchor(filename, True)
        image_name, anchorX, anchorY = res_to_image_anchor(filename, False, east)

        ## set angle to 0 to modify original functionality
        angle = 0

        ## if the base filename is in the dictionary, add the annotations into the dictionary
        if image_name in image_dict:
            curr_list = image_dict[image_name]
            ## extend the image ground truth annotation list with new annotations
            curr_list.extend(getAnnotationsFromFile(os.path.join(predicted_annotations_dir, filename), anchorX, anchorY, angle))
            image_dict[image_name] = curr_list
        else:
            image_dict[image_name] = [getAnnotationsFromFile(os.path.join(predicted_annotations_dir, filename), anchorX, anchorY, angle)]
    return image_dict


def generateIoUReportThresholded(thresholds, calculated_iou_dictionary, outfile):
    """
    DEPRECATED: flawed logic

    Creates a text file with the keys and values of the IoU values

    Inputs:
        - thresholds: List of thresholds being considered
        - calculated_iou_dictionary: Dictionary of files with corresponding average IoU values
        - outfile: Filepath to where the report should be written
    """
    with open(outfile, "w+") as f:
        for threshold in thresholds:
            f.write("Considering Threshold: %.1f\n" % threshold)
            for key in calculated_iou_dictionary[threshold]:
                f.write("%s (%.1f): Precision=%.5f Recall=%.5f\n" % (key, threshold, calculated_iou_dictionary[threshold][key][0], calculated_iou_dictionary[threshold][key][1]))

def generateIoUReport(calculated_iou_dictionary, outfile):
    """
    DEPRECATED: flawed logic

    Creates a text file with the keys and values of the IoU values

    Inputs:
        - calculated_iou_dictionary: Dictionary of files with corresponding average IoU values
        - outfile: Filepath to where the report should be written
    """

    with open(outfile, "w+") as f:
        for key in calculated_iou_dictionary:
                f.write("%s: Precision=%.5f Recall=%.5f\n" % (key, calculated_iou_dictionary[key][0], calculated_iou_dictionary[key][1]))


def generatePRCurves(thresholded_dictionary, detector, figures_dir):
    """
    DEPRECATED: flawed logic

    Creates the PR Curves for thresholded precision and recall results

    Inputs
        - thresholded_dictionary: Dictionary of precision recall for thresholds
        - figures_dir: Filepath to save curves
        - detector: Character detector letter
    """

    ## invert the dictionaries
    inverted_dict = marshal_thresholded_dictionary(thresholded_dictionary)
    ## plot the curves
    subplot_image(inverted_dict, detector, figures_dir)

def generatePrecisionRecallReport(calculated_iou_dictionary, detector, report_dir, report_path):
    """
    Creates a text file with the keys and values of the mean precision recall values
    and plots the precision recall curves for detectors.

    Inputs:
        - calculated_iou_dictionary: Dictionary of files with corresponding performance values
        - detector: Character detector letter
        - report_path: Filepath to where the report should be written
    """

    ## capture the detector's mean average precision over the different images
    mAP = calculated_iou_dictionary['detector_mAP']
    
    with open(report_path, "w+") as f:

        ## write the mean average precision value
        f.write("Mean Average Precision=%s\n" % (str(mAP)))

        ## write the report for each image
        for key in calculated_iou_dictionary:
            ## each image has (prec, mprec, rec, mrec)
            ## where prec is a list of precision values
            ## mprec is the list of monotonically decreasing precision values
            ## rec is a list of recall values
            ## mrec is the monotonically increasing recall values
            ## ap is the average precision

            if key == 'detector_mAP':
                continue

            ## unpack the values
            prec, mprec, rec, mrec, ap = calculated_iou_dictionary[key]
            ## write the average precision over the image
            f.write("%s: Average Precision=%s\n" % (key, str(ap)))
            ## plot the precision recall curve; use mprec and mrec for interpolated values
            subplot_curve(prec, rec, ap, detector, key, report_dir)
            
