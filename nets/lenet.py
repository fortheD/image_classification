import tensorflow as tf

class lenet(object):

    def __init__(self, num_classes, data_format):
        self.num_classes = num_classes
        self.data_format = data_format

    def __call__(self, inputs):
        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU.
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        return self.build(inputs, self.num_classes)

    def build(self, inputs, num_classes):
        # Assume the input size is [None, 28, 28, 1]
        conv1 = tf.layers.conv2d(inputs, filters=6, kernel_size=5, kernel_initializer=tf.variance_scaling_initializer(), data_format=self.data_format, name="conv_1") #[None, 24, 24, 6]
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, data_format=self.data_format) #[None, 12, 12, 6]
        conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, kernel_initializer=tf.variance_scaling_initializer(), data_format=self.data_format, name="conv_2") #[None, 8, 8, 6]
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, data_format=self.data_format) #[None, 4, 4, 16]

        fcIn = tf.reduce_mean(pool2, axis=[2, 3] if self.data_format == "channels_first" else [1, 2]) #[None, 16]
        fc1 = tf.layers.dense(fcIn, 120, name="fc1") #[None, 120]
        fc2 = tf.layers.dense(fc1, 84, name="fc2") #[None, 84]
        fc3 = tf.layers.dense(fc2, num_classes, name="fc3") #[None, num_classes]
        return fc3