from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags

"""
Tensorflow version 1.14 can call tf.keras.experimental.export_saved_model
to convert a tf.keras.Model to saved_model format
"""

tf.flags.DEFINE_string('export_dir', '/tmp/export', 'The export model directory')
tf.flags.DEFINE_string('origin_model_path', '/tmp/train/VGG16.h5', 'Origin model path')

FLAGS = flags.FLAGS

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    image = tf.placeholder(tf.float32, [None, 224, 224, 3])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'input_1': image,})
    session_config = tf.ConfigProto(allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(session_config=session_config)
    estimator = tf.keras.estimator.model_to_estimator(keras_model_path=FLAGS.origin_model_path, model_dir="train", config=run_config)
    estimator.export_savedmodel(FLAGS.export_dir, input_fn, checkpoint_path="train/keras/keras_model.ckpt", strip_default_attrs=True)
    tf.logging.info('export model to saved_model format complete')
    return

if __name__ == '__main__':
    tf.app.run()