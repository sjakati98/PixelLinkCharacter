"""
    Author: Shishir Jakati
    This file takes in predicted annotations of crops of one particular image
    and then draws on the original large image
"""
import re
import os
import sys
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
    pattern = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\.txt"
    image_name, anchor_x0, anchor_y0 = re.match(pattern, filename).groups()
    return (image_name, int(anchor_x0), int(anchor_y0))


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
    predicted_filenames = glob(os.path.join(predicted_annotations_dir, ".txt"))
    image_dict = {filename[:-5]: [] for filename in image_filenames}
    for filename in predicted_filenames:
        image_name, _, _ = res_to_image_anchor(filename)
        image_dict[image_name] = image_dict.get(image_name, []).append(os.path.join(predicted_annotations_dir, filename)) ## need to append the absolute file path
    return image_dict

def list_crops_to_annotated_image(original_image, annotations, outfile):
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
        _, anchor_x0, anchor_y0 = res_to_image_anchor(annotation)
        for line in open(annotation).readlines():
            gt = line.split(',')
            oriented_box = [int(gt[i]) for i in range(8)]
            print("Drawing Box: ", oriented_box)
            draw.rectangle([oriented_box[0] + anchor_x0, oriented_box[1] + anchor_y0, oriented_box[-2] + anchor_x0, oriented_box[-1] + anchor_y0], outline='red')
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