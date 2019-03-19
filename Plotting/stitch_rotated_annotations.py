"""
    Author: Shishir Jakati
    This file takes in predicted annotations of crops of one particular image, which is rotated,
    and then draws on the original large image
"""
import re
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from glob import glob


## function to partition the whole predicted files into large files
## res_cropped_image_D5005-5028149_800_5000.txt is the format

## need to create a regex that gets out the image name and the anchor point

def res_to_image_anchor(filename):
    """
        Inputs:
            - filename: The filename of the predicted output annotation
        Outputs:
            - image_name:  The name of the corresponding large image, with no extension
            - anchor_x0: The x-coordinate of the top left of the associated crop region
            - anchor_y0: The y-coordinate of the top left of the associated crop region
    """
    filename = filename.split(os.sep)[-1]    
    pattern = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\_(-?\d*)\.txt"
    image_name, anchor_x0, anchor_y0, angle = re.match(pattern, filename).groups()
    return (image_name, int(anchor_x0), int(anchor_y0), int(angle))


def create_file_dictionary(original_images_dir, predicted_annotations_dir):
    """
        Inputs:
            - original_images_dir: Directory where the original images which were cropped live
            - predicted_annotations_dir: Directory where the text box annotation files live; this is for crops
        Outputs:
            - image_dict: Dictionary of format {
                (key) image_filename: (value) [image_crop_filename]
            }
    """
    image_filenames = glob(os.path.join(original_images_dir, "*.tiff"))
    predicted_filenames = glob(os.path.join(predicted_annotations_dir, "*.txt"))
    print("Number of annotations files: ", len(predicted_filenames))
    image_dict = {filename[:-5].split(os.sep)[-1]: [] for filename in image_filenames}
    for filename in predicted_filenames:
        image_name, _, _, _ = res_to_image_anchor(filename)
        if image_name in image_dict:
            curr_list = image_dict[image_name]
            curr_list.append(os.path.join(predicted_annotations_dir, filename))
            image_dict[image_name] = curr_list
        else:
            image_dict[image_name] = [os.path.join(predicted_annotations_dir, filename)]
    return image_dict

def list_crops_to_annotated_image(original_image, annotations, outfile, image_default_width=512, image_default_height=512):
    """
        Inputs:
            - original_image: The image which will be copied and have annotations drawn on it
            - annotations: The annotation filepaths that need to be translated and drawn on the image copy
            - outfile: Where the drawn on image should be saved
    """
    print("Stitching Image: ", original_image)
    image = Image.open(original_image)
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        print("Considering annotations from:", annotation)
        _, anchor_x0, anchor_y0, angle = res_to_image_anchor(annotation)
        for line in open(annotation).readlines():
            gt = line.split(',')
            oriented_box = np.array([int(gt[i]) for i in range(8)])

            ## need to warp oriented box using the negative angle
            height, width = image.size
            rotation_angle = -angle
            image_center = (width // 2, height // 2)
            rotation_mat = cv2.getRotationMatrix2D(image_center, rotation_angle, scale=1.0)
            adjustedWidth = image_default_width
            adjustedHeight = image_default_height
            bounds = (adjustedWidth, adjustedHeight)
            rotation_mat[0, 2] += (adjustedWidth / 2) - image_center[0]
            rotation_mat[1, 2] += (adjustedHeight / 2) - image_center[1]
            oriented_box = cv2.warpAffine(oriented_box, rotation_mat, bounds)

            print("Drawing Box: ", oriented_box)
            draw.rectangle([oriented_box[6] + anchor_x0, oriented_box[7] + anchor_y0, oriented_box[2] + anchor_x0, oriented_box[3] + anchor_y0], outline='red')
    del draw
    image.save(outfile)
    print("Image Saved: ", outfile)


def driver(original_images_dir, predicted_annotations_dir, output_dir):
    """
        Inputs:
            - original_images_dir: Where the uncropped images live
            - predicted_annotations_dir: Where the test time predicted annotations live
            - output_dir: Where the drawn on images live
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    image_dict = create_file_dictionary(original_images_dir, predicted_annotations_dir)
    print("Dictionary Created")
    ##print(image_dict)
    for i, image in enumerate(image_dict.keys()):
        print("Key #:", i)
        ## pass the function the image filename and the list of annotation files
        original_image_filename = os.path.join(original_images_dir, image + ".tiff")
        outfile = os.path.join(output_dir, image + ".jpg")
        list_crops_to_annotated_image(original_image_filename, image_dict[image], outfile)


original_images_dir = sys.argv[1]
predicted_annotations_dir = sys.argv[2]
output_dir = sys.argv[3]

print("Ingesting images from:", original_images_dir)
print("Ingesting annotations from: ", predicted_annotations_dir)
print("Saving annotated images to:", output_dir)

driver(original_images_dir, predicted_annotations_dir, output_dir)

print("Done Annotating")