import os
import sys
from sklearn.model_selection import train_test_split
from shutil import copyfile
from glob import glob

cropped_path = sys.argv[1] ## parent cropped directory

images_dir_path = os.path.join(cropped_path, "images")
annotations_dir_path = os.path.join(cropped_path, "annotations")

train_split_path = os.path.join(cropped_path, "train_split")

test_split_path = os.path.join(cropped_path, "test_split")

if not os.path.exists(train_split_path):
	os.mkdir(train_split_path)
	os.mkdir(os.path.join(train_split_path, "images"))
	os.mkdir(os.path.join(train_split_path, "annotations"))
if not os.path.exists(test_split_path):
	os.mkdir(test_split_path)
	os.mkdir(os.path.join(test_split_path, "images"))
	os.mkdir(os.path.join(test_split_path, "annotations"))

train_split_images_path = os.path.join(train_split_path, "images")
train_split_annotations_path = os.path.join(train_split_path, "annotations")

test_split_images_path = os.path.join(test_split_path, "images")
test_split_annotations_path = os.path.join(test_split_path, "annotations")


image_filenames = glob(os.path.join(images_dir_path, "*.jpg"))
train_idx, test_idx = train_test_split(list(range(len(image_filenames))), test_size=0.2)

get_image_name = lambda raw_file_name: raw_file_name.split(os.sep)[-1].split('.')[0][14:]

with open(os.path.join(train_split_path, "train_split.txt"), 'w+') as train_file:
    for idx in train_idx:
        image_filename = image_filenames[idx]
        train_file.write("%s\n" % image_filename)
        copyfile(image_filename, os.path.join(train_split_images_path, image_filename.split(os.sep)[-1]))
        image_name = get_image_name(image_filename)
        for annotation_folder in list(filter(lambda folder: os.path.isdir(os.path.join(annotations_dir_path, folder)), os.listdir(annotations_dir_path))):
            current_annotation_folder = os.path.join(train_split_annotations_path, annotation_folder)
            if not os.path.exists(current_annotation_folder):
                os.mkdir(current_annotation_folder)
            
            annotation_file_name = os.path.join(annotations_dir_path, annotation_folder, "annotation_" + image_name + ".txt")
            output_annotation_file_name = os.path.join(train_split_annotations_path,, annotation_folder "annotation_" + image_name + ".txt")
            copyfile(annotation_file_name, output_annotation_file_name)

with open(os.path.join(test_split_path, "test_split.txt"), 'w+') as test_file:
    for idx in test_idx:
        image_filename = image_filenames[idx]
        test_file.write("%s\n" % image_filename)
        copyfile(image_filename, os.path.join(test_split_images_path, image_filename.split(os.sep)[-1]))
        image_name = get_image_name(image_filename)
        for annotation_folder in list(filter(lambda folder: os.path.isdir(os.path.join(annotations_dir_path, folder)), os.listdir(annotations_dir_path))):
            current_annotation_folder = os.path.join(test_split_annotations_path, annotation_folder)
            if not os.path.exists(current_annotation_folder):
                os.mkdir(current_annotation_folder)
            
            annotation_file_name = os.path.join(annotations_dir_path, annotation_folder, "annotation_" + image_name + ".txt")
            output_annotation_file_name = os.path.join(test_split_annotations_path,, annotation_folder "annotation_" + image_name + ".txt")
            copyfile(annotation_file_name, output_annotation_file_name)

