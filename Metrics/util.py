import os
import re

def res_to_image_anchor(filename, rotated=False):
    """
        Inputs:
            - filename: The filename of the predicted output annotation
            - rotated: Indicates whether the crop is rotated
        Outputs:
            - image_name:  The name of the corresponding large image, with no extension
            - anchorX: The x-coordinate of the top left of the associated crop region
            - anchorY: The y-coordinate of the top left of the associated crop region
            - angle: The angle which the crop region is rotated
    """
    
    filename = filename.split(os.sep)[-1]    
    # pattern_horizontal = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\.txt"
    pattern = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\_(-?\d*)\.txt"

    # if rotated:
    #     pattern = pattern_angle
    image_name, anchorX, anchorY, angle = re.match(pattern, filename).groups()
    return (image_name, int(anchorX), int(anchorY), int(angle))
    # else:
    #     pattern = pattern_horizontal
    #     image_name, anchorX, anchorY= re.match(pattern, filename).groups()
    #     return (image_name, int(anchorX), int(anchorY))

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

def getAnnotationsFromFile(annotation_filename, anchorX, anchorY):
    """
    Returns list of 8 point annotation
    Inputs:
        - annotation_filename: Full filename of annotation file
        - anchorX: X-axis offset of crop
        - anchorY: Y-axis offset of crop
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
    add_anchors = lambda points: [int(x) for x in [points[0] + anchorX, points[1] + anchorY, points[2] + anchorX, points[3] + anchorY, points[4] + anchorX, points[5] + anchorY, points[6] + anchorX, points[7] + anchorY]]
    annotations = [add_anchors(float_points(line)) for line in lines]
    return annotations