# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# /usr/bin/env python2.7

"""Send PNG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import cv2
import time
import os

import numpy as np

tf.app.flags.DEFINE_string('server', '0.0.0.0:11412',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/home/leike/proj/traffic_sign/predict_image/10/9e9485c6bf334e9f8ac275b453b1dc85.jpg', 'path to image in PNG format')

FLAGS = tf.app.flags.FLAGS

def per_image_standardization(image):
    image = np.cast['float32'](image)
    image_mean = np.mean(image, axis=(-1, -2, -3), keepdims=True)

    variance = (np.mean(np.square(image), axis=(-1, -2, -3), keepdims=True) - np.square(image_mean))
    variance = (np.abs(variance) + variance)/2
    stddev = np.sqrt(variance)

    shapes = image.shape
    nums = 1
    for shape in shapes:
        nums *= shape
    min_stddev = 1/np.sqrt((np.cast['float32'](nums)))
    pixel_value_scale = np.max((stddev, min_stddev))
    pixel_value_offset = image_mean

    image = np.subtract(image, pixel_value_offset)
    image = image/pixel_value_scale
    return image

def main(_):
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    image_path = FLAGS.image
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img,(224, 224))
    image_data = per_image_standardization(resized_img)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'vgg'
    request.inputs['input_1'].CopyFrom(
        tf.make_tensor_proto(image_data, dtype=tf.float32, shape=[1,224,224,3]))
    result = stub.Predict(request, 100.0)  # 100 secs timeout
    output = list(result.outputs['predictions'].float_val)
    print(output)
    cv2.waitKey(0)

if __name__ == '__main__':
  tf.app.run()
