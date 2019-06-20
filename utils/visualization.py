from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.model import ClassifyModel

flags = tf.app.flags

tf.flags.DEFINE_string('origin_model_path', '/tmp/train/VGG16.h5', 'Origin model path')
tf.flags.DEFINE_string('saved_path', '/tmp/plotmodel/', 'The model image path')

FLAGS = flags.FLAGS

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    model = tf.keras.models.load_model(FLAGS.origin_model_path)

    if not tf.gfile.IsDirectory(FLAGS.saved_path):
        tf.gfile.MakeDirs(FLAGS.saved_path)
    SAVED_PATH = FLAGS.saved_path + "model.png"
    tf.keras.utils.plot_model(model, SAVED_PATH)
    tf.logging.info('model plot complete')
    return

if __name__ == '__main__':
    tf.app.run()