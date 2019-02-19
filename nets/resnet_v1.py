from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == "channels_first" else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                    Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

def build_block_v1(inputs, filters, training, projection_shortcut, strides, data_format):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs=inputs, training=training, data_format=data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs=inputs, training=training, data_format=data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters*4, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs=inputs, training=training, data_format=data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs

def block_layer(inputs, filters, block_fn, blocks, strides, training, name, data_format):

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs, filters*4, kernel_size=1, strides=strides, data_format=data_format)

    # only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)

def post_process(inputs, num_classes, data_format):
    inputs = tf.reduce_mean(inputs, axis=[2, 3] if data_format == "channels_first" else [1, 2], keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')

    inputs = tf.squeeze(inputs, axis=[2, 3] if data_format == "channels_first" else [1, 2])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs


class resnet_v1(object):
    '''Class of resnet v1 model'''
    
    def __init__(self, resnet_size, num_class, data_format, resnet_version):
        self.resnet_size = resnet_size
        self.num_class = num_class
        self.data_format = data_format
        self.resnet_version = resnet_version
    
    def __call__(self, inputs, training):
        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats            
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        if self.resnet_size == 50:
            return self.build_resnet50(inputs, training, self.num_class)
        elif self.resnet_size == 101:
            return self.build_resnet101(inputs, training, self.num_class)
        elif self.resnet_size == 152:
            return self.build_resnet152(inputs, training, self.num_class)
        else:
            raise ValueError("The resnet size %d is not supported" % self.resnet_size)

    def build_resnet50(self, inputs, training, num_classes):
        '''Assume the input shape is [None, 224, 224, 3]'''
        with tf.variable_scope("block1"):
            inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3, strides=1, data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            if self.resnet_version == 1:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

        with tf.variable_scope("block2"):
            inputs = block_layer(inputs, filters=64, block_fn=build_block_v1, blocks=3, strides=2, training=training, name="scale2", data_format=self.data_format)
        
        with tf.variable_scope("block3"):
            inputs = block_layer(inputs, filters=128, block_fn=build_block_v1, blocks=4, strides=2, training=training, name="scale3", data_format=self.data_format)

        with tf.variable_scope("block4"):
            inputs = block_layer(inputs, filters=256, block_fn=build_block_v1, blocks=6, strides=2, training=training, name="scale4", data_format=self.data_format)

        with tf.variable_scope("block5"):
            inputs = block_layer(inputs, filters=512, block_fn=build_block_v1, blocks=3, strides=2, training=training, name="scale5", data_format=self.data_format)
        
        outputs = post_process(inputs, num_classes=num_classes, data_format=self.data_format)
        return outputs
    
    def build_resnet101(self, inputs, training, num_class):
        '''Assume the input shape is [None, 224, 224, 3]'''
        with tf.variable_scope("block1"):
            inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3, strides=1, data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            if self.resnet_version == 1:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

        with tf.variable_scope("block2"):
            inputs = block_layer(inputs, filters=64, block_fn=build_block_v1, blocks=3, strides=2, training=training, name="scale2", data_format=self.data_format)
        
        with tf.variable_scope("block3"):
            inputs = block_layer(inputs, filters=128, block_fn=build_block_v1, blocks=4, strides=2, training=training, name="scale3", data_format=self.data_format)

        with tf.variable_scope("block4"):
            inputs = block_layer(inputs, filters=256, block_fn=build_block_v1, blocks=23, strides=2, training=training, name="scale4", data_format=self.data_format)

        with tf.variable_scope("block5"):
            inputs = block_layer(inputs, filters=512, block_fn=build_block_v1, blocks=3, strides=2, training=training, name="scale5", data_format=self.data_format)
        
        outputs = post_process(inputs, num_classes=num_classes, data_format=self.data_format)
        return outputs
    
    def build_resnet152(self, inputs, training, num_class):
        '''Assume the input shape is [None, 224, 224, 3]'''
        with tf.variable_scope("block1"):
            inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3, strides=1, data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            if self.resnet_version == 1:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

        with tf.variable_scope("block2"):
            inputs = block_layer(inputs, filters=64, block_fn=build_block_v1, blocks=3, strides=2, training=training, name="scale2", data_format=self.data_format)
        
        with tf.variable_scope("block3"):
            inputs = block_layer(inputs, filters=128, block_fn=build_block_v1, blocks=8, strides=2, training=training, name="scale3", data_format=self.data_format)

        with tf.variable_scope("block4"):
            inputs = block_layer(inputs, filters=256, block_fn=build_block_v1, blocks=36, strides=2, training=training, name="scale4", data_format=self.data_format)

        with tf.variable_scope("block5"):
            inputs = block_layer(inputs, filters=512, block_fn=build_block_v1, blocks=3, strides=2, training=training, name="scale5", data_format=self.data_format)
        
        outputs = post_process(inputs, num_classes=num_classes, data_format=self.data_format)
        return outputs

if __name__ == "__main__":
    model = resnet_v1(152, 10, "channels_first", resnet_version=1)
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    output = model(inputs, training=True)
    print(output.get_shape().as_list())