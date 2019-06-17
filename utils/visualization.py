from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.model import ClassifyModel

flags = tf.app.flags

tf.flags.DEFINE_string('model_name', 'DenseNet121', 'The model you want to visualize')
tf.flags.DEFINE_string('saved_path', '/tmp/plotmodel/', 'The model image path')

FLAGS = flags.FLAGS

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_name = FLAGS.model_name

    classify_model = ClassifyModel(input_shape=(224,224,3), model_name=model_name, classes=1000, data_format="channels_last")
    model = classify_model.keras_model()

    if not tf.gfile.IsDirectory(FLAGS.saved_path):
        tf.gfile.MakeDirs(FLAGS.saved_path)
    SAVED_PATH = FLAGS.saved_path + model_name + ".png"
    tf.keras.utils.plot_model(model, SAVED_PATH)
    tf.logging.info('model plot complete')
    return

if __name__ == '__main__':
    tf.app.run()