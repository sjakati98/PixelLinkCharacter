"""
    Author: Shishir Jakati
    This file takes in predicted annotations of crops of one particular image
    and then draws on the original large image
"""
import re
import numpy as np
from PIL import Image

## function to partition the whole predicted files into large files
## res_cropped_image_D5005-5028149_800_5000.txt is the format

## need to create a regex that gets out the image name and the anchor point

def res_to_image_anchor(filename):
    """
        Inputs:
            filename - The filename of the predicted output annotation
        Outputs:
            image_name - The name of the corresponding large image, with no extension
            anchor_x0 - The x-coordinate of the top left of the associated crop region
            anchor_y0 - The y-coordinate of the top left of the associated crop region
    """    
    pattern = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\.txt"
    image_name, anchor_x0, anchor_y0 = re.match(pattern, filename).groups()
    return (image_name, int(anchor_x0), int(anchor_y0))
