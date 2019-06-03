from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Add
from tensorflow.keras.layers import ZeroPadding2D, MaxPool2D
from tensorflow.keras.layers import Dense

class ResNetV1(object):
    '''
    The class of resnet(version1)
    '''
    def __init__(self, architecture, classes, data_format):
        """
        architecture: Can be resnet50, resnet101, resnet152
        classes: The classification task classes
        data_format: channel_first or channel_last, channel_first will run faster in GPU
        training: True for training, False for inference
        """
        super(ResNetV1, self).__init__()
        assert architecture in ['resnet50', 'resnet101', 'resnet152']
        assert data_format in ['channels_first', 'channels_last']
        self.architecture = architecture
        self.classes = classes
        self.data_format = data_format

    def identity_block(self, input_tensor, kernel_size, filters, stage, block, data_format,
                    use_bias=True, training=True):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        assert len(filters) == 3
        filter1, filter2, filter3 =  filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filter1, (1, 1), name=conv_name_base+'2a', use_bias=use_bias, data_format=data_format)(input_tensor)
        x = BatchNormalization(axis=(-1, 1)[self.data_format=="channels_first"], name=bn_name_base+'2a')(x, training=training)
        x = Activation('relu')(x)

        x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base+'2b', use_bias=use_bias, data_format=data_format)(x)
        x = BatchNormalization(axis=(-1, 1)[self.data_format=="channels_first"], name=bn_name_base+'2b')(x, training=training)
        x = Activation('relu')(x)

        x = Conv2D(filter3, (1, 1), name=conv_name_base+'2c', use_bias=use_bias, data_format=data_format)(x)
        x = BatchNormalization(axis=(-1, 1)[self.data_format=="channels_first"], name=bn_name_base+'2c')(x, training=training)

        x = Add()([x, input_tensor])
        x = Activation('relu', name='res'+str(stage)+block+'_out')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, data_format,
                strides=(2, 2), use_bias=True, training=True):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        assert len(filters) == 3
        filter1, filter2, filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base+'2a', use_bias=use_bias, data_format=data_format)(input_tensor)
        x = BatchNormalization(axis=(-1, 1)[self.data_format=="channels_first"], name=bn_name_base+'2a')(x, training=training)
        x = Activation('relu')(x)

        x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base+'2b', use_bias=use_bias, data_format=data_format)(x)
        x = BatchNormalization(axis=(-1, 1)[self.data_format=="channels_first"], name=bn_name_base+'2b')(x, training=training)
        x = Activation('relu')(x)

        x = Conv2D(filter3, (1, 1), name=conv_name_base+'2c', use_bias=use_bias, data_format=data_format)(x)
        x = BatchNormalization(axis=(-1, 1)[self.data_format=="channels_first"], name=bn_name_base+'2c')(x, training=training)

        shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base+'1', use_bias=use_bias, data_format=data_format)(input_tensor)
        shortcut = BatchNormalization(axis=(-1, 1)[self.data_format=="channels_first"], name=bn_name_base+'1')(shortcut, training=training)

        x = Add()([x, shortcut])
        x = Activation('relu', name='res'+str(stage)+block+'_out')(x)
        return x

    def __call__(self, input_tensor, training):
        """Build a ResNet Graph
        """
        # Stage 1
        x = ZeroPadding2D((3, 3), data_format=self.data_format)(input_tensor)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True, data_format=self.data_format)(x)
        x = BatchNormalization(axis=(-1, 1)[self.data_format=="channels_first"], name='bn_conv1')(x, training=training)
        x = Activation('relu')(x)
        C1 = x = MaxPool2D((3, 3), strides=(2, 2), padding="same", data_format=self.data_format)(x)
        # Stage 2
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', data_format=self.data_format, strides=(1, 1), training=training)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b', data_format=self.data_format, training=training)
        C2 = x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c', data_format=self.data_format, training=training)
        # Stage 3
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a', data_format=self.data_format, training=training)
        block_count = {"resnet50": 3, "resnet101": 3, "resnet152": 7}[self.architecture]
        for i in range(block_count):
            x = self.identity_block(x, 3, [128, 128, 512], stage=3, block=chr(98+i), data_format=self.data_format, training=training)
        C3 = x
        # Stage 4
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a', data_format=self.data_format, training=training)
        block_count = {"resnet50": 5, "resnet101": 22, "resnet152": 37}[self.architecture]
        for i in range(block_count):
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98+i), data_format=self.data_format, training=training)
        C4 = x
        # Stage 5
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a', data_format=self.data_format, training=training)
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b', data_format=self.data_format, training=training)
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c', data_format=self.data_format, training=training)
        C5 = x
        # Post process
        x = tf.reduce_mean(x, axis=[2, 3] if self.data_format == "channels_first" else [1, 2])
        x = Dense(self.classes)(x)
        return x