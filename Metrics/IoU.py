"""
    Author: Shishir Jakati.
    
    The purpose of this file is to calculate the raw IoU of predicted annotations vs. ground truth
    annotations.
"""

import glob
import math
import os
import sys
from optparse import OptionParser

import numpy as np
from scipy.optimize import linear_sum_assignment

from file_dictionary_util import (createGroundTruthDictionary,
                                  createPredictedDictionary, generateIoUReport)
from iou_util import performIoUCalculation

## instantiate an options parser to load ground truth annotations and predictions
parser = OptionParser()
parser.add_option("-d", "--detector", help="specify which character detector is being evaluated")
parser.add_option("-o", "--original", help="directory containing the original images in tiff format")
parser.add_option("-g", "--groundtruth", help="directory containing ground truth annotation files")
parser.add_option("-p", "--predictions", help="directory containing predicted annotation files")

## get options
(options, args) = parser.parse_args()
detector = options.detector
original_images_dir = options.original
ground_truth_directory = options.groundtruth
ground_truth_files = glob.glob(ground_truth_directory)
predictions_directory = options.predictions
predictions_files = glob.glob(predictions_directory)

def driver(detector, original_images_dir, ground_truth_directory, predictions_directory):
    """
    Run IoU metric script for specified character detector
    
    Inputs:
        - detector: Letter of the detector to be evaluated
        - ground_truth_directory: Directory containing the ground truth annotations for the specified letter
        - predictions_directory: Directory containing the predictied annotations for the specified letter
    """
    ## concatenate all ground truth annotations for an image into a single text file
    ## dictionary keys are image filenames and value is list of lists of all annotations in float 8 point form
    ground_truth_annotation_dictionary = createGroundTruthDictionary(original_images_dir, ground_truth_directory)
    
    ## concatenate all predicted annotations for an image into a single text file
    ## dictionary keys are image filenames and value is list of lists of all annotations in float 8 point form
    predicted_annotation_dictionary = createPredictedDictionary(original_images_dir, predictions_directory)

    # dictionary keys are the image filenames and value is the average IoU score
    calculated_iou_dictionary = performIoUCalculation(ground_truth_annotation_dictionary, predicted_annotation_dictionary)
    
    ## output text file with all average IoU values
    report_filepath = os.path.join(os.curdir, "iou_report.txt")
    generateIoUReport(calculated_iou_dictionary, report_filepath)

    ## signal completion
    print("IoU Calculation Complete!")

## run
driver(detector, original_images_dir, ground_truth_directory, predictions_directory)
