import tensorflow as tf
import os
import PIL.Image
import io
import cv2

from utils import dataset_util

flags = tf.app.flags
tf.flags.DEFINE_string('input_record', 'output/train.record', 'The name of record')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def _parse_function(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/format": tf.FixedLenFeature((), tf.string, default_value="jpeg")}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_image(parsed_features["image/encoded"])
    img_format = parsed_features["image/format"]
    return image, img_format 

def _read_tf_record(input_record):
    dataset = tf.data.TFRecordDataset(input_record)
    dataset = dataset.map(_parse_function)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        image, img_format = sess.run(next_element)


def main(_):
    assert FLAGS.input_record, '`input_record` missing'
    _read_tf_record(FLAGS.input_record)


if __name__ == '__main__':
    tf.app.run()