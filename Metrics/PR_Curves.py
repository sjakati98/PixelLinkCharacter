"""
    Author: Shishir Jakati.
    
    The purpose of this file is to calculate the precision recall curves of detector performance.
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
from iou_util import performPolygonIoUCalculationThresholded

## instantiate an options parser to load ground truth annotations and predictions
parser = OptionParser()
parser.add_option("-d", "--detector", help="specify which character detector is being evaluated")
parser.add_option("-o", "--original", help="directory containing the original images in tiff format")
parser.add_option("-g", "--groundtruth", help="directory containing ground truth annotation files")
parser.add_option("-p", "--predictions", help="directory containing predicted annotation files")
parser.add_option("-r", "--reports", help="outfile directory to output the reports")

## get options
(options, args) = parser.parse_args()
detector = options.detector
original_images_dir = options.original
ground_truth_directory = options.groundtruth
ground_truth_files = glob.glob(ground_truth_directory)
predictions_directory = options.predictions
predictions_files = glob.glob(predictions_directory)
reports_directory = options.reports

def driver(detector, original_images_dir, ground_truth_directory, predictions_directory, reports_dir):
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

    ## set number of curve threshold values
    thresholds = np.linspace(0, 1, 11)
    thresholded_dictionary = {}

    # dictionary keys are the image filenames and value is the average IoU score
    for threshold in thresholds:
        thresholded_dictionary[threshold] = performPolygonIoUCalculationThresholded(ground_truth_annotation_dictionary, predicted_annotation_dictionary, threshold)
    
    ## output text file with all average IoU values
    if not reports_dir:
        reports_dir = os.path.join(os.curdir, "reports")
        
    if not os.path.isdir(reports_dir):
        os.mkdir(reports_dir)
    
    report_filepath = os.path.join(reports_dir, "%s_iou_report_thresholded.txt" % detector)
    generateIoUReportThresholded(thresholded_dictionary, report_filepath)

    ## signal completion
    print("IoU Calculation Complete!")

## run
driver(detector, original_images_dir, ground_truth_directory, predictions_directory, reports_directory)
