import re
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from glob import glob
# from DataWrangling.Cropping.crop_and_convert_tiff import save_image_as_jpg

## get all the images in the directory
## create a collection of images that are rotated from -30 to 30 degrees, in 5 degree intervals
## save them into a cropped/rotated directory

def save_image_as_jpg(image_array, outfile, crop_directory):
    
    ## Arguments
        ## image_array: This is the array container of the image pixels
        ## outfile: Path to new location of converted image; DO NOT ADD EXTENSION
        ## crop_directory: Parent directory of the newly cropped image
    ## Outputs
        ## None; will log output
    try:
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        outfile = os.path.join(crop_directory, "images", outfile)
        # result = image.point(lambda i: i*(1./256)).convert('L')
        image.save(outfile + ".jpg")
        print("Output Saved: ", outfile)
    except Exception as e:
        print(e)


def find_crops(images_parent_directory):
    """
        Inputs:
            - cropped_annotations: The directory that contains all the test time cropped images
        Outputs:
            - image_filenames: The list of filenames
    """
    ## get all the images which have a jpg file extension
    image_filenames = glob(os.path.join(images_parent_directory, "*.jpg"))
    return image_filenames


def rotate_image_and_save(image, rotation_angle, image_shape, image_outfile, outfile_parent_directory):
    """
        Inputs:
            - image_filename: The image array of the image to be rotated/translated
            - rotation_angle: The angle for the image to be rotated
            - image_shape: The shape tuple of the image
            - image_outfile: The filepath where the rotated/translated image will be saved
        Outputs:
            - rotation_mat: Rotation matrix, accounting for translation, of image
    """

    ## rotate and translate the image
    height, width = image_shape[0], image_shape[1]
    image_center = (width // 2, height // 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, rotation_angle, scale=1.0)
    cos = np.abs(rotation_mat[0, 0])
    sin = np.abs(rotation_mat[0, 1])
    
    adjustedWidth = int((height * sin) + (width * cos))
    adjustedHeight = int((height * cos) + (width * sin))
 
    bounds = (adjustedWidth, adjustedHeight)
    
    rotation_mat[0, 2] += (adjustedWidth / 2) - image_center[0]
    rotation_mat[1, 2] += (adjustedHeight / 2) - image_center[1]

    rotated_mat = cv2.warpAffine(image, rotation_mat, bounds)

    ## save the image
    save_image_as_jpg(rotated_mat, image_outfile, outfile_parent_directory)
    print("Rotated: ", image_outfile)
    
    return rotation_mat

def save_rotation_matrix(rotation_matrix, outfile):
    """
        Inputs:
            - rotation_matrix: The rotation matrix of the warped image
            - outfile: The outfile of the rotation matrix save file
    """
    np.save(outfile + ".npy", rotation_matrix)
    print("Saved Rotation Matrix: ", outfile)


def driver(images_parent_directory, image_rotation_angle_max):
    """
        Inputs:
            - images_parent_directory: Where the cropped test-time images live
            - image_rotation_angle_max: The maximal angle of rotation
    """
    ## for all images in test time crop directory
        ## for all angles between -30 and 30 degress at 5 degree increments
            ## save the rotated image
            ## save the corresponding rotation matrix
    for image_filename in find_crops(images_parent_directory):
        
        image = Image.open(image_filename)
        width, height = image.size
        
        for angle in range(-image_rotation_angle_max, image_rotation_angle_max + 5, 5):
            ## function to produce the filename of the image, with no extension
            file_name_no_extension = image_filename.split(os.sep)[-1].split('.')[0]
            build_string = lambda filename, rotation: "%s_%d" % (filename, rotation)
            rotation_mat = rotate_image_and_save(image, angle, (width, height), build_string(file_name_no_extension, angle), os.path.join(images_parent_directory, "rotated"))
            save_rotation_matrix(rotation_mat, build_string(file_name_no_extension, angle))


if __name__ == "__main__":
    
    crops_parent_directroy = sys.argv[1]
    rotation_max_angle = sys.argv[2]

    if not os.path.exists(os.path.join(crops_parent_directroy, "rotated")):
        os.mkdir(os.path.join(crops_parent_directroy, "rotated"))

    driver(crops_parent_directroy, rotation_max_angle)