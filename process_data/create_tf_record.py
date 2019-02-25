from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import sys

import tensorflow as tf

from utils import dataset_util

flags = tf.app.flags
tf.flags.DEFINE_string('image_dir', 'image', 'The image directory')
tf.flags.DEFINE_string('output_dir', '/tmp/record', 'The output tf_record directory')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

# Seed for repeatability
_RANDOM_SEED = 0

# The number of shards per dataset split
_NUM_SHARDS = 5

# Rate of validation/all
_RATE_VAL_ALL = 0.1

class ImageReader(object):
    """Helper class that provides Tensorflow image coding utilties"""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        # Get the image height and width
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        # Get the image data
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _get_filenames_and_classes(image_dir):
    """Returns a list of filenames and inferred class names.

    Args:
        dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
        A list of image file paths, relative to `dataset_dir` and the list of
        subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for filename in os.listdir(image_dir):
        path = os.path.join(image_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)

def _get_record_filename(output_dir, split_name, shard_id):
    """Return the output record filenames"""
    output_filename = 'images_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(output_dir, output_filename)

def _convert_record(split_name, filenames, class_names_to_ids, output_dir):
    """Convert the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
        dataset_dir: The directory where the converted tfrecord are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_record_filename(output_dir, split_name, shard_id)
                # Create tfrecord writer
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                        sys.stdout.flush()
                        #Get the image data
                        image_data = tf.gfile.GFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_util.image_to_tfexample(image_data, b'jpeg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def _record_exists(output_dir):
    """Judge whether the record exists"""
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_record_filename(output_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True

def main(_):
    assert FLAGS.image_dir, '`image_dir ` missing'
    assert FLAGS.output_dir, '`output_dir` missing'
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    image_dir = FLAGS.image_dir
    output_dir = FLAGS.output_dir

    if _record_exists(output_dir):
        print('Record files already exist')
        return

    photo_filenames, class_names = _get_filenames_and_classes(image_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Divide into train and test record
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)

    photo_nums = len(photo_filenames)
    validation_nums = int(photo_nums * _RATE_VAL_ALL)

    training_filenames = photo_filenames[validation_nums:]
    validation_filenames = photo_filenames[:validation_nums]

    #convert the training and validation record
    _convert_record('train', training_filenames, class_names_to_ids, output_dir)
    _convert_record('validation', validation_filenames, class_names_to_ids, output_dir)

    # Finally, write the label file
    label_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_util.write_label_file(label_to_class_names, image_dir)

    tf.logging.info("Translate complete")

if __name__ == '__main__':
    tf.app.run()