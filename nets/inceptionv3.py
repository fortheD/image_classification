from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Add
from tensorflow.keras.layers import ZeroPadding2D, MaxPool2D, AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(inputs, training, data_format, name):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == "channels_first" else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, name=name, fused=True)

class InceptionV3(object):
    """
    The class of inception(version3)
    """
    def __init__(self, classes, data_format):
        """
        classes: The classification task classes
        data_format: channel_first or channel_last, channel_first will run faster in GPU
        """
        super(InceptionV3, self).__init__()
        assert data_format in ['channels_first', 'channels_last']
        self.classes = classes
        self.data_format = data_format

    def conv2d_bn(self, input_tensor, filters, num_row, num_col, padding='same', strides=(1, 1), training=True, name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, data_format=self.data_format, name=conv_name)(input_tensor)
        x = batch_norm(x, training=training, data_format=self.data_format, name=bn_name)
        x = Activation('relu', name=name)(x)
        return x

    def __call__(self, input_tensor, training):
        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            input_tensor = tf.transpose(input_tensor, [0, 3, 1, 2])

        x = self.conv2d_bn(input_tensor, 32, 3, 3,  padding='valid', strides=(2,2), training=training)
        x = self.conv2d_bn(x, 32, 3, 3, padding='valid', training=training)
        x = self.conv2d_bn(x, 64, 3, 3, training=training)
        x = MaxPool2D((3, 3), strides=(2, 2), data_format=self.data_format)(x)

        x = self.conv2d_bn(x, 80, 1, 1, padding='valid', training=training)
        x = self.conv2d_bn(x, 192, 3, 3, padding='valid', training=training)
        x = MaxPool2D((3, 3), strides=(2, 2), data_format=self.data_format)(x)

        # mixed 0, 1, 2: 35 * 35 * 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, training=training)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1, training=training)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, training=training)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, training=training)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, training=training)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, training=training)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=self.data_format)(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1, training=training)

        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                        axis=1 if self.data_format == "channels_first" else 3,
                        name='mixed0')

        # mixed 1: 35 * 35 * 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, training=training)
    
        branch5x5 = self.conv2d_bn(x, 48, 1, 1, training=training)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, training=training)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, training=training)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, training=training)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, training=training)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=self.data_format)(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1, training=training)

        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                        axis=1 if self.data_format == "channels_first" else 3,
                        name='mixed1')

        # mixed 2
        branch1x1 = self.conv2d_bn(x, 64, 1, 1, training=training)
    
        branch5x5 = self.conv2d_bn(x, 48, 1, 1, training=training)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5, training=training)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, training=training)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, training=training)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, training=training)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=self.data_format)(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1, training=training)

        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                        axis=1 if self.data_format == "channels_first" else 3,
                        name='mixed2')

        # mixed 3
        branch3x3 = self.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid', training=training)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1, training=training)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, training=training)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid', training=training)

        branch_pool = MaxPool2D((3, 3), strides=(2, 2), data_format=self.data_format)(x)

        x = concatenate([branch3x3, branch3x3dbl, branch_pool],
                        axis=1 if self.data_format == "channels_first" else 3,
                        name='mixed3')

        # mixed 4
        branch1x1 = self.conv2d_bn(x, 192, 1, 1, training=training)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1, training=training)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7, training=training)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, training=training)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1, training=training)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1, training=training)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7, training=training)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1, training=training)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, training=training)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=self.data_format)(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                        axis=1 if self.data_format == "channels_first" else 3,
                        name='mixed4')

        # mixed 5,6
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, 1, 1, training=training)

            branch7x7 = self.conv2d_bn(x, 160, 1, 1, training=training)
            branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7, training=training)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, training=training)

            branch7x7dbl = self.conv2d_bn(x, 160, 1, 1, training=training)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1, training=training)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 1, 7, training=training)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1, training=training)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, training=training)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=self.data_format)(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                            axis=1 if self.data_format == "channels_first" else 3,
                            name='mixed' + str(5+i))

        # mixed 7
        branch1x1 = self.conv2d_bn(x, 192, 1, 1, training=training)

        branch7x7 = self.conv2d_bn(x, 192, 1, 1, training=training)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7, training=training)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1, training=training)

        branch7x7dbl = self.conv2d_bn(x, 192, 1, 1, training=training)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1, training=training)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, training=training)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1, training=training)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7, training=training)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=self.data_format)(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                        axis=1 if self.data_format == "channels_first" else 3,
                        name='mixed7')

        # mixed 8
        branch3x3 = self.conv2d_bn(x, 192, 1, 1, training=training)
        branch3x3 = self.conv2d_bn(branch3x3, 320, 3, 3, strides=(2,2), padding='valid')

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1, training=training)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7, training=training)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1, training=training)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2,2), padding='valid')

        branch_pool = MaxPool2D((3, 3), strides=(2, 2), data_format=self.data_format)(x)
        x = concatenate([branch3x3, branch7x7x3, branch_pool],
                        axis=1 if self.data_format == "channels_first" else 3,
                        name='mixed8')

        # mixed 9
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 320, 1, 1, training=training)

            branch3x3 = self.conv2d_bn(x, 384, 1, 1, training=training)
            branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = concatenate([branch3x3_1, branch3x3_2],
                                    axis=1 if self.data_format == "channels_first" else 3,
                                    name='mixed9_'+str(i))

            branch3x3dbl = self.conv2d_bn(x, 448, 1, 1, training=training)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3, training=training)
            branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3, training=training)
            branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1, training=training)
            branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                        axis=1 if self.data_format == "channels_first" else 3)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=self.data_format)(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1, training=training)
            x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                            axis=1 if self.data_format == "channels_first" else 3,
                            name='mixed'+str(9+i))

        x = GlobalAveragePooling2D(data_format=self.data_format, name='avg_pool')(x)
        x = Dense(self.classes, activation='softmax', name='predictions')(x)

        return x