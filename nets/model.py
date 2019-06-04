from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.resnetv1 import ResNetV1

classification_models = ["Xception",
                           "VGG16",
                           "VGG19",
                           "ResNet50",
                           "ResNet101",
                           "ResNet152",
                           "ResNet50V2",
                           "ResNet101V2",
                           "ResNet152V2",
                           "ResNetXt50",
                           "ResNetXt101",
                           "InceptionV3",
                           "InceptionResNetV2",
                           "MobileNet",
                           "MobileNetV2",
                           "DenseNet121",
                           "DenseNet169",
                           "DenseNet201",
                           "NASNetMobile",
                           "NASNetLarge"]

class ClassifyModel(object):
    def __init__(self, model_name, classes, data_format):
        """
        model_name: model_name should be supported in classification_models list
        classes: The classification task classes
        data_format: channel_first or channel_last, channel_first will run faster in GPU
        """
        self.model = self.build(model_name, classes, data_format)

    def build(self, model_name, classes, data_format):
        if model_name == "ResNet50":
            model = ResNetV1('resnet50', classes, data_format)
        return model

    def __call__(self, input_tensor, training):
        return self.model(input_tensor, training)