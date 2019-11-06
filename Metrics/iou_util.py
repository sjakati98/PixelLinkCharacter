from __future__ import division
import numpy as np
from shapely.geometry import Polygon
import statistics


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


def performPolygonIoUCalculationThresholded(ground_truth_annotation_dictionary, predicted_annotation_dictionary, threshold):
    """
    Wrapper function for performPolygonIoUCalculation with a specified threshold
    """
    return performPolygonIoUCalculation(ground_truth_annotation_dictionary, predicted_annotation_dictionary, threshold=threshold)

def performPolygonIoUCalculation(ground_truth_annotation_dictionary, predicted_annotation_dictionary, threshold=0.5):
    """
    Takes ground truth annotations and predicted annotations to create a dictionary of IoU scores
    Inputs:
        - ground_truth_annotation_dictionary: Dictionary of images and associated ground truth annotations
        - predicted_annotation_dictionary: Dictionary of images and predicted annotations
    Outputs:
        - Image_IoU: Dictionary of images and calculated mean IoU values; additional field 'detector_mAP'
    """
    Image_IoU = {}

    sum_AP = 0.0

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
        
        # iou = np.zeros((len(gt), len(predicted)))
        
        ## instantiate arrays for record keeping
        true_positives = [0] * len(predicted)
        false_positives = [0] * len(predicted)

        used = set()

        for i, predictedAnnotation in enumerate(predicted):
            
            ## keep track of largest intersection; so as to not double count
            iou_max = -1
            gt_match = -1

            for j, groundTruthAnnotation in enumerate(gt):
                
                groundTruthPolygon = Polygon(groundTruthInidicies(groundTruthAnnotation))
                predictedPolygon = Polygon(predictedIndicies(predictedAnnotation))

                curr_iou = polygonIOU(groundTruthPolygon, predictedPolygon)

                if curr_iou >= iou_max:
                    ## update largest intersection over union value
                    iou_max = curr_iou
                    ## keep track of the max
                    gt_match = groundTruthAnnotation

            if iou_max >= threshold:
                if str(gt_match) not in used:
                    ## this is a good detection
                    true_positives[i] = 1
                    used.add(str(gt_match))
                else:
                    ## this match has already been used
                    false_positives[i] = 1
            else:
                ## insufficient over lap of match
                false_positives[i] = 1

        ## compute precision and recall
        cumsum = 0
        for idx, val in enumerate(false_positives):
            false_positives[idx] += cumsum
            cumsum += val

        cumsum = 0
        for idx, val in enumerate(true_positives):
            true_positives[idx] += cumsum
            cumsum += val
        
        print("Number True Positives: ", true_positives[-1])
        print("Number False Postiives: ", false_positives[-1])
        print("Number False Negatives: ", len(gt) - len(used))
        
        
        rec = true_positives[:]
        for idx, val in enumerate(true_positives):
            rec[idx] = float(true_positives[idx]) / len(gt)
        #print(rec)
        prec = true_positives[:]
        for idx, val in enumerate(true_positives):
            prec[idx] = float(true_positives[idx]) / (false_positives[idx] + true_positives[idx])
        
        ap, mrec, mprec = voc_ap(rec[:], prec[:])

        ap = statistics.mean(prec)

        sum_AP += ap

        ## set dictionary value
        Image_IoU[ground_truth_key] = (prec, mprec, rec, mrec, ap)

    ## calculate mean average precision over images
    Image_IoU['detector_mAP'] = sum_AP / len(ground_truth_annotation_dictionary)

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


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre