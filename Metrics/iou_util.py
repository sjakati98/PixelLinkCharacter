import numpy as np


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
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 1e-5)

        avg_iou = np.mean(np.max(iou, axis=1))
        Image_IoU[ground_truth_key] = avg_iou
        # iou_predicted = np.sum(iou, axis=1)
        
        # #Give the option of no assignment, which is a bit better than the worst possible assignment
        # maxc = np.max(iou, axis=1)
        # for i in range(iou.shape[0]):
        #     iou[i,len(gt):] = maxc[i] - 1.0

        # rows, columns = linear_sum_assignment(iou)


        # avg_iou = 0.0
        # for idx in range(len(rows)):
        #     row = rows[idx]
        #     col = columns[idx]
        #     if col < len(annotations):
        #         intersection = compute_intersection(predictions[row], annotations[col])
        #         union = compute_union(predictions[row], annotations[col])

        #         iou = intersection / union
        #         avg_iou += iou
        #         IoU.append( (row, col, iou))

        # Image_IoU[ground_truth_key] = avg_iou
        

    return Image_IoU