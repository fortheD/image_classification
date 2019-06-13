from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from estimator.create_estimator import Classifier

from nets.vgg16 import VGG16

from nets.model import ClassifyModel

flags = tf.app.flags
tf.flags.DEFINE_string('checkpoint_dir', '/home/leike/proj/track_pedestrian/classifier', 'The checkpoint directory')
tf.flags.DEFINE_string('img_path', '/home/leike/proj/traffic_sign/predict_image/10/9e9485c6bf334e9f8ac275b453b1dc85.jpg', 'The image path')

FLAGS = flags.FLAGS

def input_fn():
    img_path = FLAGS.img_path
    image_data = tf.gfile.GFile(img_path, 'rb').read()
    decode_image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize_images(decode_image, [224, 224])
    image = tf.image.per_image_standardization(image)
    image = tf.transpose(image, [2, 0, 1])
    image = tf.expand_dims(image, axis=0)
    return image

def run(flags):
    warm_start_dir = flags.checkpoint_dir
    model_dir = '/tmp/train'
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    input_shape = (224, 224, 3)
    classify_model = ClassifyModel(input_shape=input_shape, model_name="ResNet50", classes=20, data_format=data_format)
    model = classify_model.keras_model()
    SAVED_MODEL_PATH = '/home/leike/resnet.h5'
    model.load_weights(SAVED_MODEL_PATH)
    inputs = input_fn()
    result = model.predict(inputs, steps=1)
    print(result)
    return

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    run(FLAGS)

if __name__ == '__main__':
    tf.app.run()