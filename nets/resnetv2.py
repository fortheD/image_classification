# -*- coding: utf-8 -*-
'''ResNetV2 model for Keras.

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def identity_block(input_tensor, kernel_size, filters, stage, block, data_format):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, data_format, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(input_tensor)
    shortcut = Activation('relu')(shortcut)
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    return x

def ResNetV2(architecture, inputs, classes):
    """
    architecture: Can be resnet50, resnet101, resnet152
    inputs: model inputs
    classes: The classification task classes
    """
    assert architecture in ['resnet50', 'resnet101', 'resnet152']
    data_format = K.image_data_format()

    bn_axis = 3
    if data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        bn_axis = 1

    x = ZeroPadding2D((3, 3), data_format=data_format)(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', data_format=data_format)(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), data_format=data_format)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', data_format=data_format, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, data_format=data_format, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, data_format=data_format, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, data_format=data_format, block='a')
    block_count = {"resnet50": 3, "resnet101": 3, "resnet152": 7}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [128, 128, 512], stage=3, data_format=data_format, block=chr(98+i))


    x = conv_block(x, 3, [256, 256, 1024], stage=4, data_format=data_format, block='a')
    block_count = {"resnet50": 5, "resnet101": 22, "resnet152": 37}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, data_format=data_format, block=chr(98+i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, data_format=data_format, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, data_format=data_format, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, data_format=data_format, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool', data_format=data_format)(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc')(x)

    # Create model.
    model = Model(inputs, x, name=architecture)
    return model