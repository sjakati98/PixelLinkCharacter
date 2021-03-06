#encoding=utf-8
import numpy as np;
import tensorflow as tf
import util
import os
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example

import sys
sys.path.append('/home/sgkelley/pixel_link')

import config

        

def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')#[0:10];
    print("%d images found in %s"%(len(image_names), data_path));
    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = [];
            bboxes = []
            labels = [];
            labels_text = [];
            path = util.io.join_path(data_path, image_name);
            print("\tconverting image: %d/%d %s"%(idx, len(image_names), image_name));
            image_data = tf.gfile.FastGFile(path, 'r').read()
            
            image = util.img.imread(path, rgb = True);
            shape = image.shape
            h, w = shape[0:2];
            h *= 1.0;
            w *= 1.0;
            image_name = util.str.split(image_name, '.')[0];
            
            gt_name = 'annotation_' + image_name[14:] + '.txt';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            lines = util.io.read_lines(gt_filepath);
                
            for line in lines:
                line = util.str.remove_all(line, '\xef\xbb\xbf')
                gt = util.str.split(line, ',');
                oriented_box = [int(gt[i]) for i in range(8)];
                oriented_box = np.asarray(oriented_box) / ([w, h] * 4);
                oriented_bboxes.append(oriented_box);
                
                xs = oriented_box.reshape(4, 2)[:, 0]                
                ys = oriented_box.reshape(4, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])

                # might be wrong here, but it doesn't matter because the label is not going to be used in detection
                labels_text.append(gt[-1]); 
                ignored = util.str.contains(gt[-1], '###')
                if ignored:
                    labels.append(config.ignore_label);
                else:
                    labels.append(config.text_label)
            example = convert_to_example(image_data, image_name, labels, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())
        
if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('/mnt/nfs/work1/elm/sgkelley/shishir/cropped')
    train_split_dir = util.io.join_path(root_dir, 'train_split')
    test_split_dir = util.io.join_path(root_dir, 'test_split')

    os.mkdir(os.path.join(train_split_dir, 'tfrecords'))
    os.mkdir(os.path.join(test_split_dir, 'tfrecords'))

    training_data_dir = util.io.join_path(train_split_dir, 'images')
    output_dir = util.io.join_path(os.path.join(train_split_dir, 'tfrecords'))
    annotations_parent_directory = util.io.join_path(train_split_dir, 'annotations')
    for annotation_folder in list(filter(lambda folder: os.path.isdir(os.path.join(annotations_parent_directory, folder)), os.listdir(annotations_parent_directory))):
        training_gt_dir = util.io.join_path(train_split_dir, 'annotations', annotation_folder)
        cvt_to_tfrecords(output_path = util.io.join_path(output_dir, annotation_folder + '_train.tfrecord'), data_path = training_data_dir, gt_path = training_gt_dir)
    
    
    output_dir = util.io.join_path(os.path.join(train_split_dir, 'tfrecords'))
    test_data_dir = util.io.join_path(test_split_dir, 'images')
    annotations_parent_directory = util.io.join_path(test_split_dir, 'annotations')
    for annotation_folder in list(filter(lambda folder: os.path.isdir(os.path.join(annotations_parent_directory, folder)), os.listdir(annotations_parent_directory))):
        test_gt_dir = util.io.join_path(test_split_dir, 'annotations', annotation_folder)
        cvt_to_tfrecords(output_path = util.io.join_path(output_dir, annotation_folder + '_test.tfrecord'), data_path = test_data_dir, gt_path = test_gt_dir)