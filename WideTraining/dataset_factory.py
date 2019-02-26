"""A factory-pattern class which returns classification image/label pairs."""
from datasets import dataset_utils

class DatasetConfig():
    def __init__(self, file_pattern, split_sizes):
        self.file_pattern = file_pattern
        self.split_sizes = split_sizes
        
icdar2013 = DatasetConfig(
        file_pattern = '*_%s.tfrecord', 
        split_sizes = {
            'train': 229,
            'test': 233
        }
)
icdar2015 = DatasetConfig(
        file_pattern = 'icdar2015_%s.tfrecord', 
        split_sizes = {
            'train': 1000,
            'test': 500
        }
)
td500 = DatasetConfig(
        file_pattern = '*_%s.tfrecord', 
        split_sizes = {
            'train': 300,
            'test': 200
        }
)
tr400 = DatasetConfig(
        file_pattern = 'tr400_%s.tfrecord', 
        split_sizes = {
            'train': 400
        }
)
scut = DatasetConfig(
    file_pattern = 'scut_%s.tfrecord',
    split_sizes = {
        'train': 1715
    }
)

synthtext = DatasetConfig(
    file_pattern = '*.tfrecord',
#     file_pattern = 'SynthText_*.tfrecord',
    split_sizes = {
        'train': 858750
    }
)

maptext = DatasetConfig(
    file_pattern = 'char_anots_%s_train.tfrecord',
    split_sizes = {
        '0': 20976,
        '1': 20976,
        '2': 20976,
        '3': 20976,
        '4': 20976,
        '5': 20976,
        '6': 20976,
        '7': 20976,
        '8': 20976,
        '9': 20976,
        'a': 20976,
        'b': 20976,
        'c': 20976,
        'd': 20976,
        'e': 20976,
        'f': 20976,
        'g': 20976,
        'h': 20976,
        'i': 20976,
        'j': 20976,
        'k': 20976,
        'l': 20976,
        'm': 20976,
        'n': 20976,
        'o': 20976,
        'p': 20976,
        'q': 20976,
        'r': 20976,
        's': 20976,
        't': 20976,
        'u': 20976,
        'v': 20976,
        'w': 20976,
        'x': 20976,
        'y': 20976,
        'z': 20976,
        'A': 20976,
        'B': 20976,
        'C': 20976,
        'D': 20976,
        'E': 20976,
        'F': 20976,
        'G': 20976,
        'H': 20976,
        'I': 20976,
        'J': 20976,
        'K': 20976,
        'L': 20976,
        'M': 20976,
        'N': 20976,
        'O': 20976,
        'P': 20976,
        'Q': 20976,
        'R': 20976,
        'S': 20976,
        'T': 20976,
        'U': 20976,
        'V': 20976,
        'W': 20976,
        'X': 20976,
        'Y': 20976,
        'Z': 20976
    }
)

datasets_map = {
    'icdar2013':icdar2013,
    'icdar2015':icdar2015,
    'scut':scut,
    'td500':td500,
    'tr400':tr400,
    'synthtext':synthtext,
    'maptext': maptext,
}


def get_dataset(dataset_name, split_name, dataset_dir, reader=None):
    """Given a dataset dataset_name and a split_name returns a Dataset.
    Args:
        dataset_name: String, the dataset_name of the dataset.
        split_name: A train/test split dataset_name.
        dataset_dir: The directory where the dataset files are stored.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    Returns:
        A `Dataset` class.
    Raises:
        ValueError: If the dataset `dataset_name` is unknown.
    """
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    dataset_config = datasets_map[dataset_name];
    file_pattern = dataset_config.file_pattern
    num_samples = dataset_config.split_sizes[split_name]
    return dataset_utils.get_split(split_name, dataset_dir,file_pattern, num_samples, reader)