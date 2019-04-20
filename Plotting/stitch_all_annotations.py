import os
import re
import sys
from glob import glob

import cv2
import numpy as np

from PIL import Image, ImageDraw

def res_to_image_anchor(filename, rotated=False):
    """
        Inputs:
            - filename: The filename of the predicted output annotation
            - rotated: Indicates whether the crop is rotated
        Outputs:
            - image_name:  The name of the corresponding large image, with no extension
            - anchor_x0: The x-coordinate of the top left of the associated crop region
            - anchor_y0: The y-coordinate of the top left of the associated crop region
            - angle: The angle which the crop region is rotated
    """
    
    filename = filename.split(os.sep)[-1]    
    pattern_horizontal = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\.txt"
    pattern_angle = "res\_cropped\_image\_(.*)\_(\d*)\_(\d*)\_(-?\d*)\.txt"

    if rotated:
        pattern = pattern_angle
        image_name, anchor_x0, anchor_y0, angle = re.match(pattern, filename).groups()
        return (image_name, int(anchor_x0), int(anchor_y0), int(angle))
    else:
        pattern = pattern_horizontal
        image_name, anchor_x0, anchor_y0= re.match(pattern, filename).groups()
        return (image_name, int(anchor_x0), int(anchor_y0))

def create_file_dictionary(original_images_dir, horizontal_annotations_dir, rotated_annotations_dir):
    """
        Inputs:
            - original_images_dir: Directory where the original images which were cropped live
            - horizontal_annotations_dir: Directory where the horizontal text box annotation files live; this is for crops
            - rotated_annotations_dir: DIrectory where the rotated text box annotation files live; this is for crops
        Outputs:
            - image_dict_horizontal: Dictionary of format {
                (key) image_filename: (value) [image_crop_filename]
            }
            - image_dict_rotated: Dictionary of format {
                (key) image_filename: (value) [image_crop_filename]
            }
    """
    image_filenames = glob(os.path.join(original_images_dir, "*.tiff"))
    horizontal_predicted_filenames = glob(os.path.join(horizontal_annotations_dir, "*.txt"))
    rotated_predicted_filenames = glob(os.path.join(rotated_annotations_dir, "*.txt"))
    print("Number of annotations files: ", len(horizontal_annotations_dir) + len(rotated_predicted_filenames))
    image_dict_horizontal = {filename[:-5].split(os.sep)[-1]: [] for filename in image_filenames}
    image_dict_rotated = {filename[:-5].split(os.sep)[-1]: [] for filename in image_filenames}
    
    for filename in horizontal_predicted_filenames:
        image_name, _, _ = res_to_image_anchor(filename)
        if image_name in image_dict_horizontal:
            curr_list = image_dict_horizontal[image_name]
            curr_list.append(os.path.join(horizontal_annotations_dir, filename))
            image_dict_horizontal[image_name] = curr_list
        else:
            image_dict_horizontal[image_name] = [os.path.join(horizontal_annotations_dir, filename)]
    
    for filename in rotated_predicted_filenames:
        image_name, _, _, _= res_to_image_anchor(filename, True)
        if image_name in image_dict_rotated:
            curr_list = image_dict_rotated[image_name]
            curr_list.append(os.path.join(rotated_annotations_dir, filename))
            image_dict_rotated[image_name] = curr_list
        else:
            image_dict_rotated[image_name] = [os.path.join(rotated_annotations_dir, filename)]
    
    return image_dict_horizontal, image_dict_rotated


def list_crops_to_annotated_image(original_image, horizontal_annotations, rotated_annotations, outfile):
    """
        Inputs:
            - original_image: The image which will be copied and have annotations drawn on it
            - annotations: The annotation filepaths that need to be translated and drawn on the image copy
            - outfile: Where the drawn on image should be saved
    """
    print("Stitching Image: ", original_image)
    image = Image.open(original_image)
    draw = ImageDraw.Draw(image)

    for annotation in horizontal_annotations:
        print("Considering annotations from:", annotation)
        _, anchor_x0, anchor_y0 = res_to_image_anchor(annotation)
        for line in open(annotation).readlines():
            gt = line.split(',')
            oriented_box = [int(gt[i]) for i in range(8)]
            print("Drawing Box: ", oriented_box)
            draw.rectangle([oriented_box[6] + anchor_x0, oriented_box[7] + anchor_y0, oriented_box[2] + anchor_x0, oriented_box[3] + anchor_y0], outline='red')
        
    image_default_height = image_default_width = 512
    
    for annotation in rotated_annotations:
        print("Considering annotations from:", annotation)
        _, anchor_x0, anchor_y0, angle = res_to_image_anchor(annotation)
        for line in open(annotation).readlines():
            gt = line.split(',')
            oriented_box = np.array([int(gt[i]) for i in range(8)])
            
            ## get the dimensions of the original rotated image
            image_center = (image_default_width // 2, image_default_height // 2)
            rotation_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)
            cos = np.abs(rotation_mat[0, 0])
            sin = np.abs(rotation_mat[0, 1])
            
            adjustedWidth = int((image_default_height * sin) + (image_default_width * cos))
            adjustedHeight = int((image_default_height * cos) + (image_default_width * sin))

            cX, cY = (adjustedWidth // 2, adjustedHeight // 2)
            ## rotate the box around the center of the original rotated image dimensions
            box_matrix  = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            
            corners = oriented_box.reshape(-1,2)
            corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
            ## calculate the new box
            new_box = np.dot(box_matrix, corners.T).T.reshape(1, 8)[0]
            
            new_box = [int(x) for x in [new_box[0] + anchor_x0, new_box[1] + anchor_y0, new_box[2] + anchor_x0, new_box[3] + anchor_y0, new_box[4] + anchor_x0, new_box[5] + anchor_y0, new_box[6] + anchor_x0, new_box[7] + anchor_y0]]
            new_box = np.array(new_box)
            new_box[0::2] += int((image_default_width / 2) - cX)
            new_box[1::2] += int((image_default_height / 2) - cY)
            new_box = list(new_box)
            
            print("Drawing Box: ", new_box)
            draw.polygon(new_box, outline="blue", fill=None)

    del draw
    image.save(outfile)
    print("Image Saved: ", outfile)


def driver(original_images_dir, horizontal_predicted_annotations_dir, rotated_predicted_annotations_dir, output_dir):
    """
        Inputs:
            - original_images_dir: Where the uncropped images live
            - horizontal_predicted_annotations_dir: Where the test time predicted annotations live
            - rotated_predicted_annotations_dir: Where the test time predicted annotations live
            - output_dir: Where the drawn on images live
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    image_dict_horizontal, image_dict_rotated = create_file_dictionary(original_images_dir, horizontal_predicted_annotations_dir, rotated_predicted_annotations_dir)
    print("Dictionary Created")
    ##print(image_dict)
    for i, image in enumerate(image_dict_horizontal.keys()):
        print("Key #:", i)
        ## pass the function the image filename and the list of annotation files
        original_image_filename = os.path.join(original_images_dir, image + ".tiff")
        outfile = os.path.join(output_dir, image + ".jpg")
        list_crops_to_annotated_image(original_image_filename, image_dict_horizontal[image], image_dict_rotated[image],outfile)