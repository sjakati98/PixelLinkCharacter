"""
    Author: Shishir Jakati

    Create 512 x 512 crops of images, as well as their corresponding annotations
    for the purpose of training PixelLink. Utilize OpenCV to crop images, and Pillow
    to convert the crops to TIFF.
"""
import os
import numpy as np

from glob import glob
import logging
from PIL import Image


def crop_images_to_512_512(image_array, height, width):
    ## Generator function
    ## Arguments
        ## image_array: This is the array container of the image pixels
        ## height: Height of the image
        ## width: Width of the image
    ## Outputs
        ## (cropped: Cropped region of image, 
        #   x_0: Top-Left anchor x-coordinate,
        #   y_0: Top-Left anchor y-coordinate,
        #   image_crop_x: Width of cropped region,
        #   image_crop_y: Height of cropped region)
    
    image_x_max = height 
    image_y_max = width

    image_crop_x = image_crop_y = 512
    x_0 = y_0 = 0
    image_crop_step = 200
    crops = []

    for y_0 in range(0, image_y_max - image_crop_y, image_crop_step):
        for x_0 in range(0, image_x_max - image_crop_x, image_crop_step):
            cropped = image_array[y_0:y_0+image_crop_y, x_0:x_0+image_crop_x]
            yield (cropped, x_0, y_0, image_crop_x, image_crop_y)

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


def save_cropped_image(image, width, height, file_stripped_name, crop_directory):

    ## Arguments
        ## image: Image array of image to be cropped
        ## width: Width of the image; for naming usage
        ## height: Height of the image; from naming usage
        ## file_stripped_name: Raw name identifier of the image
        ## crop_directory: Parent directory of the newly cropped images

    for cropped, x_0, y_0, _, _ in crop_images_to_512_512(image, width, height):
        build_string = lambda x_0, y_0, file_stripped_name: "cropped_image_%s_%d_%d" % (file_stripped_name, x_0 , y_0)
        save_image_as_jpg(cropped, build_string(x_0, y_0, file_stripped_name), crop_directory)
