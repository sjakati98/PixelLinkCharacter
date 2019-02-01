import numpy as np
from glob import glob
import os
import cv2
import sys

def check_box_in_crop(box_coordinates, crop_x, crop_y, crop_width=512, crop_height=512):
	"""
		Inputs:
		- box_coordinates: List[(Int, Int)]; the 4 coordinate tuples
		- crop_x: Int; top-left x coordinate of the crop
		- crop_y: Int; top-left y coordinate of the crop 
		- crop_width: Int; width of the crop window
		- crop_height: Int; height of the crop window
		Outputs:
		- relative_coordinates: List[(Int, Int)]; returns 4 coordinate tuples of the relative_coordinates
			- **returns None if the original box_coordinates are not completely within the crop**
	"""
	crop_x_max = crop_x + crop_width
	crop_y_max = crop_y + crop_height

	relative_coordinates = []

	for i, coordinate_pair in enumerate(box_coordinates):
		coordinate_x = coordinate_pair[0]
		coordinate_y = coordinate_pair[1]
		if crop_x <= coordinate_x <= crop_x_max and crop_y <= coordinate_y <= crop_y_max:
			relative_x = int(coordinate_x - crop_x)
			relative_y = int(coordinate_y - crop_y)
			relative_coordinates.append((relative_x, relative_y))
		else:
			return None

	return relative_coordinates

def find_and_parse_npy_files(parent_directory):
    ## Generator function
    ## Arguments
        ## parent_directory: Directory where the NPY annotations live
    ## Outputs
        ## _dict: Verticies format of annotations

    annotation_filenames = glob(os.path.join(parent_directory, "*.npy"))
    for annotation_file in annotation_filenames:
        anot = np.load(annotation_file)
        _dict = np.reshape(anot, -1)[0]
        yield (_dict, annotation_file)

def get_image_dimensions(file_name_no_extension, images_dir_path):
	current_image_path = os.path.join(images_dir_path, file_name_no_extension + ".tiff")
	current_image_path = os.path.join(images_dir_path, file_name_no_extension + ".tiff")
	current_image = cv2.imread(current_image_path)
	(width, height, _) = current_image.shape
	return width, height

def save_cropped_annotations(annotations_directory, images_dir_path):
	parent_directory = annotations_directory
	for _dict, annotation_file in find_and_parse_npy_files(parent_directory):
		
		file_name_no_extension = annotation_file.split('.')[0]

		width, height = get_image_dimensions(file_name_no_extension, images_dir_path)
		image_x_max = width 
		image_y_max = height

		image_crop_x = image_crop_y = 512
		image_crop_step = 200

		for y_0 in range(0, image_y_max - image_crop_y, image_crop_step):
			for x_0 in range(0, image_x_max - image_crop_x, image_crop_step):
				with open(os.path.join(annotations_directory, "cropped", ("annotation_%s_%d_%d.txt" % (file_name_no_extension, x_0 , y_0))), "w+") as f:
					for key in _dict:
						verticies = _dict[key]['vertices']
						if len(verticies) == 4:
							relative_coordinates = check_box_in_crop(verticies, x_0, y_0, image_crop_x, image_crop_y)
							if relative_coordinates is not None: 
								box_string = ",".join([str(x) for t in relative_coordinates for x in t])
								f.write("%s\n" % box_string)