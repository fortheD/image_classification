from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.vgg16 import VGG16
from nets.vgg19 import VGG19
from nets.resnetv1 import ResNetV1
from nets.resnetv2 import ResNetV2
from nets.inceptionv3 import InceptionV3

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
    def __init__(self, input_shape, model_name, classes, data_format):
        """
        model_name: model_name should be supported in classification_models list
        classes: The classification task classes
        data_format: channels_first or channels_last, channels_first will run faster in GPU
        """
        super(ClassifyModel, self).__init__()
        if data_format == "channels_first":
            input_shape = (input_shape[2], input_shape[0], input_shape[1])
        inputs = tf.keras.layers.Input(shape=input_shape)
        self.model = self.build(model_name, inputs, classes, data_format)

    def build(self, model_name, inputs, classes, data_format):
        if model_name == "VGG16":
            model = VGG16(inputs, classes, data_format)
        elif model_name == "VGG19":
            model = VGG19(inputs, classes, data_format)
        elif model_name == "ResNet50":
            model = ResNetV1('resnet50', inputs, classes, data_format)
        elif model_name == "ResNet101":
            model = ResNetV1('resnet101', inputs, classes, data_format)
        elif model_name == "ResNet152":
            model = ResNetV1('resnet152', inputs, classes, data_format)
        elif model_name == "ResNet50V2":
            model = ResNetV2("resnet50", inputs, classes, data_format)
        elif model_name == "ResNet101V2":
            model = ResNetV2("resnet101", inputs, classes, data_format)
        elif model_name == "ResNet152V2":
            model = ResNetV2("resnet152", inputs, classes, data_format)
        elif model_name == "InceptionV3":
            model = InceptionV3(inputs, classes, data_format)
        return model

    def keras_model(self):
        return self.model