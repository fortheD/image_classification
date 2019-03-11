from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from utils import dataset_util

flags = tf.app.flags
tf.flags.DEFINE_string('tfrecord_dir', '/home/leike/proj/MOTData/record', 'The tf_record directory')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def get_recordnames(is_training, record_dir):
    """""Returns a list of tensorflow record names"""
    assert tf.gfile.Exists(record_dir), ('The record directory do not exists')

    train_record_names = []
    val_record_names = []

    for filename in os.listdir(record_dir):
        if 'train' in filename:
            train_record_names.append(os.path.join(record_dir, filename))
        elif 'validation' in filename:
            val_record_names.append(os.path.join(record_dir, filename))
    
    if is_training:
        return train_record_names
    return val_record_names

def get_dataset(is_training, record_dir):
    record_names = get_recordnames(is_training, record_dir)
    dataset = tf.data.TFRecordDataset(record_names)
    return dataset

def preprocess_image(image, height, width, is_training):
    image = tf.cast(image, tf.float32)
    image.set_shape([None, None, 3])
    if is_training:
        imgae = tf.image.resize_image_with_crop_or_pad(image, height+8, width+8)
        image = tf.random_crop(image, [height, width, 3])
        image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_images(image, [40, 40])
    image = tf.image.per_image_standardization(image)
    return image

def _parse_function(example_proto, is_training):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/format": tf.FixedLenFeature((), tf.string, default_value="jpeg"),
                "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                "image/height": tf.FixedLenFeature((), tf.int64, default_value=0),
                "image/width": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_image(parsed_features["image/encoded"])
    label = parsed_features["image/class/label"]
    height = tf.cast(parsed_features["image/height"], tf.int32)
    width = tf.cast(parsed_features["image/width"], tf.int32)
    image = preprocess_image(image, height, width, is_training)
    return image, label

def train(record_dir):
    dataset = get_dataset(is_training=True, record_dir=record_dir)
    return dataset.map(lambda record:_parse_function(record, True))

def val(record_dir):
    dataset = get_dataset(is_training=False, record_dir=record_dir)
    return dataset.map(lambda record:_parse_function(record, False))

def _read_tf_record(record_dir):
    dataset = val(record_dir)
    next_element = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        image, img_format = sess.run(next_element)
        print(image)
        print(img_format)

def main(_):
    assert FLAGS.tfrecord_dir, '`tfrecord_dir` missing'
    record_dir = FLAGS.tfrecord_dir
    _read_tf_record(record_dir)

if __name__ == '__main__':
    tf.app.run()