import tensorflow as tf
import sys

path_to_events_file = sys.argv[1]

for e in tf.train.summary_iterator(path_to_events_file):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(v.simple_value)