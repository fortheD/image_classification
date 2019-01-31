from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
RECORD_BYTES = DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 10
NUM_DATA_FILES = 5

def get_filenames(is_training, data_dir):
    """Returns a list of filenames."""
    assert tf.gfile.Exists(data_dir), (
        'Run cifar10_download_and_extract.py first to download and extract the '
        'CIFAR-10 data.')
    if is_training:
        return [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, NUM_DATA_FILES + 1)
        ]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]

def get_dataset(is_training, data_dir):
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, RECORD_BYTES)
    return dataset

def preprocess_image(image, is_training):
    if is_training:
        imgae = tf.image.resize_image_with_crop_or_pad(image, HEIGHT+8, WIDTH+8)
        image = tf.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])
        image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image

def parse_record(raw_record, is_training, dtype):
    '''Parse image and label from a raw record'''
    # Convert bytes to a vector of uint8
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, then convert from uint8 to int32
    label = tf.cast(record_vector[0], tf.int32)

    # The remaining bytes after label represent the image, we reshape from
    # [depth * height * width] to [depth, height, width]
    depth_major = tf.reshape(record_vector[1:RECORD_BYTES], [NUM_CHANNELS, HEIGHT, WIDTH])

    #Convert from [depth, height, width] to [height, width, depth], cast the data type
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    image = preprocess_image(image, is_training)
    image = tf.cast(image, dtype)

    return image, label

def train(directory):
    dataset = get_dataset(is_training=True, data_dir=directory)
    return dataset.map(lambda record:parse_record(record, True, tf.float32))

def test(directory):
    dataset = get_dataset(is_training=False, data_dir=directory)
    return dataset.map(lambda record:parse_record(record, False, tf.float32))

if __name__ == "__main__":
    dataset = test('/tmp/cifar10_data/cifar-10-batches-bin')
    next_element = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        print(sess.run(next_element))