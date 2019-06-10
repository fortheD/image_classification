from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from train_and_eval import model_fn
from estimator.create_estimator import Classifier

flags = tf.app.flags
tf.flags.DEFINE_string('checkpoint_dir', '/home/leike/proj/track_pedestrian/classifier', 'The checkpoint directory')
tf.flags.DEFINE_string('img_path', '/home/leike/proj/traffic_sign/predict_image/2/f93426970f90479dac9384b65c376054.jpg', 'The image path')

FLAGS = flags.FLAGS

def input_fn():
    img_path = FLAGS.img_path
    image_data = tf.gfile.GFile(img_path, 'rb').read()
    decode_image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize_images(decode_image, [112, 112])
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, axis=0)
    return image

def run(flags):
    warm_start_dir = flags.checkpoint_dir
    model_dir = '/tmp/train'
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    classifier = Classifier("ResNet50", 20, data_format)
    estimator = tf.estimator.Estimator(
        model_fn=classifier.model_fn,
        model_dir=model_dir,
        params={'image_nums':15000}
    )

    # estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params={'data_format':data_format})
    results = estimator.predict(input_fn)
    first = False
    for result in results:
        if not first:
            tf.logging.info('result %s' % result)
            first = True
        break

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    run(FLAGS)

if __name__ == '__main__':
    tf.app.run()