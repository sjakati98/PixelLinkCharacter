"""
    Author: Shishir Jakati

    Process to handle both image and annotation cropping. 
"""
from glob import glob
import os
import numpy as np
from PIL import Image
import sys
from crop_and_convert_tiff import save_cropped_image
from crop_character_annotations import save_cropped_annotations

def harvest(images_parent_directory, annotations_parent_directory, crop_directory):
    ## Find the filenames of TIFF images, and then finds the corresponding annotation file
    ## Arguments
        ## images_parent_directory: Parent directory where the images live
        ## annotations_parent_directory: Directory where the character annotations home folder is (contains each character folder)
        ## crop_directory: Where the cropped images and annotations will go
    
    image_filenames = glob(os.path.join(images_parent_directory, "*.tiff"))
    for filename in image_filenames:
        try:
            ## get the stripped filename
            file_stripped_name = filename.split(os.sep)[-1].split('.')[0]
            
            image = Image.open(filename)
            width, height = image.size

            ## start image cropping
            save_cropped_image(np.array(image), width, height, file_stripped_name, crop_directory)

            for annotation_folder in list(filter(lambda folder: os.path.isdir(os.path.join(annotations_parent_directory, folder), os.listdir(annotations_parent_directory))):
                ## get the corresponding annotation filename
                annotation_filename = os.path.join(annotations_parent_directory, annotation_folder, "current", file_stripped_name + ".npy")
                ## start annotation cropping
                save_cropped_annotations(annotation_filename, width, height, os.path.join(crop_directory, annotation_folder))

            print("Cropped: ", file_stripped_name)
        except Exception as e:
            print(e)

    
if __name__ == "__main__":
    images_parent_directory = sys.argv[1]
    annotations_parent_directory = sys.argv[2]
    crop_directory = sys.argv[3]
    
    if not os.path.exists(crop_directory):
        os.mkdir(crop_directory)
        os.mkdir(os.path.join(crop_directory, "images"))
        os.mkdir(os.path.join(crop_directory, "annotations"))

    harvest(images_parent_directory, annotations_parent_directory, crop_directory)