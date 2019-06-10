from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense, Flatten

class VGG19(object):
    """
    The class of vgg19
    """
    def __init__(self, classes, data_format):
        """
        classes: The classification task classes
        data_format: channel_first or channel_last, channel_first will run faster in GPU
        """        
        super(VGG19, self).__init__()
        assert data_format in ['channels_first', 'channels_last']
        self.classes = classes
        self.data_format = data_format

    def __call__(self, input_tensor):
        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            input_tensor = tf.transpose(input_tensor, [0, 3, 1, 2])

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block1_conv1')(input_tensor)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block1_conv2')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), data_format=self.data_format, name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block2_conv2')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), data_format=self.data_format, name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block3_conv4')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), data_format=self.data_format, name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block4_conv4')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), data_format=self.data_format, name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=self.data_format, name='block5_conv4')(x)        
        x = MaxPool2D((2, 2), strides=(2, 2), data_format=self.data_format, name='block5_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

        return x