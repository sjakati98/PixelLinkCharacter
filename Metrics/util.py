import os
import re
import numpy as np
import cv2

def res_to_image_anchor(filename, rotated=False, east=False):
    """
        Inputs:
            - filename: The filename of the predicted output annotation
            - rotated: Indicates whether the crop is rotated
            - east: For east outputs
        Outputs:
            - image_name:  The name of the corresponding large image, with no extension
            - anchorX: The x-coordinate of the top left of the associated crop region
            - anchorY: The y-coordinate of the top left of the associated crop region
            - angle: The angle which the crop region is rotated
    """
    
    filename = filename.split(os.sep)[-1]    
    pattern_horizontal = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\.txt"
    pattern_angle = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\_(-?\d*)\.txt"

    if east:
        pattern = "cropped\_image\_(.*)\_(\d*)\_(\d*)\.txt"
        image_name, anchorX, anchorY = re.match(pattern, filename).groups()
        return (image_name, int(anchorX), int(anchorY))

    if rotated:
        pattern = pattern_angle
        image_name, anchorX, anchorY, angle = re.match(pattern, filename).groups()
        return (image_name, int(anchorX), int(anchorY), int(angle))
    else:
        pattern = pattern_horizontal
        image_name, anchorX, anchorY = re.match(pattern, filename).groups()
        return (image_name, int(anchorX), int(anchorY))

def ground_truth_to_image_anchor(filename):
    """
        Inputs:
            - filename: The filename of the ground truth annotation
        Outputs:
            - image_name:  The name of the corresponding large image, with no extension
            - anchorX: The x-coordinate of the top left of the associated crop region
            - anchorY: The y-coordinate of the top left of the associated crop region
    """
    
    filename = filename.split(os.sep)[-1]    
    pattern_horizontal = "annotation\_(.*)\_(\d*)\_(\d*)\.txt"
    image_name, anchorX, anchorY = re.match(pattern_horizontal, filename).groups()
    return (image_name, int(anchorX), int(anchorY))

def getAnnotationsFromFile(annotation_filename, anchorX, anchorY, angle=0):
    """
    Returns list of 8 point annotation
    Inputs:
        - annotation_filename: Full filename of annotation file
        - anchorX: X-axis offset of crop
        - anchorY: Y-axis offset of crop
        - angle: Angle of the crop if 0, no additional processing is done
    Outputs:
        - annotations: List of annotations
    """
    
    ## read file contents
    with open(annotation_filename) as f:
        annotation_text = f.read()
    ## get annotations in string form
    lines = annotation_text.split("\n")[:-1]
    ## convert to float
    float_points = lambda line: [float(x.strip()) for x in line.split(",")]
    if angle == 0:
        add_anchors = lambda points: [int(x) for x in [points[0] + anchorX, points[1] + anchorY, points[2] + anchorX, points[3] + anchorY, points[4] + anchorX, points[5] + anchorY, points[6] + anchorX, points[7] + anchorY]]
        annotations = [add_anchors(float_points(line)) for line in lines]
    else:
        annotations = [rotateAnnotations(float_points(line), anchorX, anchorY, angle) for line in lines]

    return annotations



def rotateAnnotations(points, anchorX, anchorY, angle):
    """
    Rotates an annotated box back to the appropriate angle
    Inputs:
        - points: The eight point points of the annotation
        - anchorX: The x-axis anchor of the crop
        - anchorY: The y-axis anchor of the crop
        - angle: The angle the annotation needs to be translated back
    Outputs:
        - new_box: The original annotations, rotated back
    """
    image_default_height = image_default_width = 512


    image_center = (512 // 2, 512 // 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)
    cos = np.abs(rotation_mat[0, 0])
    sin = np.abs(rotation_mat[0, 1])
    
    adjustedWidth = int((image_default_height * sin) + (image_default_width * cos))
    adjustedHeight = int((image_default_height * cos) + (image_default_width * sin))

    cX, cY = (adjustedWidth // 2, adjustedHeight // 2)
    ## rotate the box around the center of the original rotated image dimensions
    box_matrix  = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    points = np.array(points)
    corners = points.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    ## calculate the new box
    new_box = np.dot(box_matrix, corners.T).T.reshape(1, 8)[0]
    
    new_box = [int(x) for x in [new_box[0] + anchorX, new_box[1] + anchorY, new_box[2] + anchorX, new_box[3] + anchorY, new_box[4] + anchorX, new_box[5] + anchorY, new_box[6] + anchorX, new_box[7] + anchorY]]
    new_box = np.array(new_box)
    new_box[0::2] += int((image_default_width / 2) - cX)
    new_box[1::2] += int((image_default_height / 2) - cY)
    new_box = list(new_box)

    return new_box
            
            