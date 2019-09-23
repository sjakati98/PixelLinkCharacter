from __future__ import division
import numpy as np
from shapely.geometry import Polygon


def performIoUCalculation(ground_truth_annotation_dictionary, predicted_annotation_dictionary):
    """
    Takes ground truth annotations and predicted annotations to create a dictionary of IoU scores
    Inputs:
        - ground_truth_annotation_dictionary: Dictionary of images and associated ground truth annotations
        - predicted_annotation_dictionary: Dictionary of images and predicted annotations
    Outputs:
        - Image_IoU: Dictionary of images and calculated mean IoU values
    """
    Image_IoU = {}
    for ground_truth_key in ground_truth_annotation_dictionary:
        
        print("Reading Image:", ground_truth_key)

        ## safety check to make sure that key is in the dictionary
        if ground_truth_key in predicted_annotation_dictionary:
            ## load gt annotations for image
            gt = np.array(ground_truth_annotation_dictionary[ground_truth_key])
            ## load predicted annotations for image
            predicted = np.array(predicted_annotation_dictionary[ground_truth_key])



        print("Total Ground Truth Annotations", len(gt))
        print("Total Predicted Annotations", len(predicted))

        if len(gt) == 0 or len(predicted) == 0:
            continue

        ## vectorized IoU calculation
        x11, y11, x12, y12 = np.split(predicted[:, [2,3,6,7]], 4, axis=1)
        x21, y21, x22, y22 = np.split(gt[:, [6,7,2,3]], 4, axis=1)
        
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 1e-5) # --> (len(predicted), len(gt))

        ## do calculations for ground truth
        iou = iou.T ## --> (len(gt), len(predicted))

        ## get all scores above the threshold
        above_threshold = (iou >= 0.1).astype(int)
        
        print("Above Threshold", np.count_nonzero(above_threshold))

        ## get all false negatives
        gt_matches = np.sum(above_threshold, axis=1)
        num_false_negatives = len(gt_matches) - np.count_nonzero(gt_matches)
        
        ## get all the true positives and false positives 
        predicted_matches = np.sum(above_threshold, axis=0)
        num_true_positives = np.count_nonzero(predicted_matches)
        num_false_positives = len(predicted_matches) - np.count_nonzero(predicted_matches)

        print("Num True Positives", num_true_positives)
        print("Num False Positives", num_false_positives)
        print("Num False Negatives", num_false_negatives)

        ## get precision and recall numbers
        precision = num_true_positives / (num_true_positives + num_false_positives)
        recall = num_true_positives / (num_true_positives + num_false_negatives)

        ## set dictionary value
        Image_IoU[ground_truth_key] = (precision, recall)


    return Image_IoU


def performPolygonIoUCalculation(ground_truth_annotation_dictionary, predicted_annotation_dictionary):
    """
    Takes ground truth annotations and predicted annotations to create a dictionary of IoU scores
    Inputs:
        - ground_truth_annotation_dictionary: Dictionary of images and associated ground truth annotations
        - predicted_annotation_dictionary: Dictionary of images and predicted annotations
    Outputs:
        - Image_IoU: Dictionary of images and calculated mean IoU values
    """
    Image_IoU = {}
    for ground_truth_key in ground_truth_annotation_dictionary:
        
        print("Reading Image:", ground_truth_key)

        ## safety check to make sure that key is in the dictionary
        if ground_truth_key in predicted_annotation_dictionary:
            ## load gt annotations for image
            gt = ground_truth_annotation_dictionary[ground_truth_key]
            ## load predicted annotations for image
            predicted = predicted_annotation_dictionary[ground_truth_key]

        print("Total Ground Truth Annotations", len(gt))
        print("Total Predicted Annotations", len(predicted))

        if len(gt) == 0 or len(predicted) == 0:
            continue
        
        iou = np.zeros((len(gt), len(predicted)))

        for i, groundTruthAnnotation in enumerate(gt):
            for j, predictedAnnotation in enumerate(predicted):
                
                groundTruthPolygon = Polygon(groundTruthInidicies(groundTruthAnnotation))
                predictedPolygon = Polygon(predictedIndicies(predictedAnnotation))

                iou[i][j] = polygonIOU(groundTruthPolygon, predictedPolygon)

        ## get all scores above the threshold
        above_threshold = (iou >= 0.1).astype(int)
        
        print("Above Threshold", np.count_nonzero(above_threshold))

        ## get all false negatives
        gt_matches = np.sum(above_threshold, axis=1)
        num_false_negatives = len(gt_matches) - np.count_nonzero(gt_matches)
        
        ## get all the true positives and false positives 
        predicted_matches = np.sum(above_threshold, axis=0)
        num_true_positives = np.count_nonzero(predicted_matches)
        num_false_positives = len(predicted_matches) - np.count_nonzero(predicted_matches)

        print("Num True Positives", num_true_positives)
        print("Num False Positives", num_false_positives)
        print("Num False Negatives", num_false_negatives)

        ## get precision and recall numbers
        precision = num_true_positives / (num_true_positives + num_false_positives)
        recall = num_true_positives / (num_true_positives + num_false_negatives)

        ## set dictionary value
        Image_IoU[ground_truth_key] = (precision, recall)

    return Image_IoU
    

def pointsPairs(points):
    """
    Creates list of (x,y) points
    Inputs:
        - points: Pure list format
    Outputs:
        - verticies: Correct tuple format
    """
    return list(zip(points[0::2], points[1::2]))

def predictedIndicies(predicted):
    """
    Correctly marshals the predicted points into the correct format
    Inputs:
        - predicted: The inputted prediction
    Outputs:
        - marshalled: The outputted prediction in current format
    """
    points = [predicted[0], predicted[1], predicted[6], predicted[7], predicted[4], predicted[5], predicted[2], predicted[3]]
    marshalled = pointsPairs(points)
    return marshalled

def groundTruthInidicies(gt):
    """
    Correctly marshals the predicted points into the correct format
    Inputs:
        - predicted: The inputted prediction
    Outputs:
        - marshalled: The outputted prediction in current format
    """
    marshalled = pointsPairs(gt)
    return marshalled

def polygonIOU(poly1, poly2):
    """
    IoU calculation for two polygons
    Inputs:
        - poly1: Shapely Polygon Object
        - poly2: Shapely Polygon Object
    Outputs:
        - iou: Calculated intersection over union of two inputted polygons
    """
    return poly1.intersection(poly2).area / poly1.union(poly2).area