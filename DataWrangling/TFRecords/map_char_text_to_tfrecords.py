#encoding=utf-8
import numpy as np;
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example

import sys
sys.path.append('/home/sgkelley/pixel_link')

import config


def get_splits_make_box_files(root_dir):
    raw_annotations = open('/home/sgkelley/pixel_link/datasets/cropped_annotations_angles_-90to90step5_fixed.txt', 'r')
    lines = raw_annotations.readlines()

    annotations = []
    for i in range(len(lines)):
        line = lines[i]
        if line[0] == '/':
            filename = line.split('/')[-1]
            split_imgname = filename.split(".")[0].split("_")
            angle = split_imgname[1]
            if angle == '0':
                boxes = [filename.strip()]
                num = int(lines[i + 1].strip())
                for j in range(1, num + 1):
                    idx = i + 1 + j
                    raw_box = list(map(lambda entry:  float(entry),lines[idx].split(" ")))
                    [x_min, y_max, width, height] = raw_box
                    x1 = x_min; y1 = y_max - height
                    x2 = x_min + width; y2 = y_max - height
                    x3 = x_min; y3 = y_max
                    x4 = x_min + width; y4 = y_max
                    boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
                annotations.append(boxes)
    
    annotations = np.array(annotations)
    np.random.shuffle(annotations)
    eighty_percent = int(annotations.size * 0.8)
    train_annotations = annotations[:eighty_percent]
    test_annotations = annotations[eighty_percent:]

    output_dir = util.io.get_absolute_path('/home/sgkelley/pixel_link/datasets/')
    train_gt_path = util.io.join_path(output_dir, 'train_gt')
    test_gt_path = util.io.join_path(output_dir, 'test_gt')

    ## make the actual directories and files
    ## TODO: create the files in the respective directories
    util.io.mkdir(train_gt_path)
    for annotation in train_annotations:
        new_filename = 'gt_' + annotation[0]
        with open(util.io.join_path( train_gt_path, new_filename + '.txt'), 'w+') as f:
            for line in annotation[1:]:
                f.write(",".join(list(map(str, line))) + '\n')

    util.io.mkdir(test_gt_path)
    for annotation in train_annotations:
        new_filename = 'gt_' + annotation[0]
        with open(util.io.join_path(test_gt_path, new_filename + '.txt'), 'w+') as f:
            for line in annotation[1:]:
                f.write(",".join(list(map(str, line))) + '\n')


    train_filenames = [annotation[0] for annotation in train_annotations]
    test_filenames = [annotation[0] for annotation in test_annotations]

    return train_filenames, test_filenames, train_gt_path, test_gt_path
        

def cvt_to_tfrecords(output_path , data_path, filenames, gt_path):
    # image_names = util.io.ls(data_path, '.jpg')#[0:10];
    image_names = filenames
    print "%d images found in %s"%(len(image_names), data_path);
    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = [];
            bboxes = []
            labels = [];
            labels_text = [];
            path = util.io.join_path(data_path, image_name);
            print "\tconverting image: %d/%d %s"%(idx, len(image_names), image_name);
            image_data = tf.gfile.FastGFile(path, 'r').read()
            
            image = util.img.imread(path, rgb = True);
            shape = image.shape
            h, w = shape[0:2];
            h *= 1.0;
            w *= 1.0;
            # image_name = util.str.split(image_name, '.')[0];
            gt_name = 'gt_' + image_name + '.txt';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            lines = util.io.read_lines(gt_filepath);
                
            for line in lines:
                line = util.str.remove_all(line, '\xef\xbb\xbf')
                gt = util.str.split(line, ',');
                oriented_box = [int(float(gt[i])) for i in range(8)];
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
    # root_dir = util.io.get_absolute_path('~/dataset/ICDAR2015/Challenge4/')

    root_dir = util.io.get_absolute_path('/home/sgkelley/pixel_link')
    output_dir = util.io.get_absolute_path('/home/sgkelley/pixel_link/datasets/tf_records')
    util.io.mkdir(output_dir);

    train_filenames, test_filenames, train_gt_path, test_gt_path = get_splits_make_box_files(root_dir)

    maps_dir = util.io.get_absolute_path('/home/sgkelley/data/maps/angles_-90to90step5/jpg_converts/cropped_img')
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'maps_train.tfrecord'), data_path = maps_dir, filenames = train_filenames, gt_path = train_gt_path)
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'maps_test.tfrecord'), data_path = maps_dir, filenames = test_filenames, gt_path = test_gt_path)