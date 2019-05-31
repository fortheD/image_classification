import tensorflow as tf

class vgg(object):

    def __init__(self, num_classes, data_format, vgg_version=16):
        self.num_classes = num_classes
        self.data_format = data_format
        self.vgg_version = vgg_version
    
    def __call__(self, inputs, training):
        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU.
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        if self.vgg_version == 16:
            return self.build_vgg16(inputs, self.data_format, self.num_classes, training)
        elif self.vgg_version == 19:
            return self.build_vgg19(inputs, self.data_format, self.num_classes, training)
        else:
            raise ValueError("The vgg version should be 16 or 19")
    
    def convLayer(self, inputs, filters, data_format, name):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, padding='SAME', data_format=data_format,
            kernel_initializer=tf.variance_scaling_initializer(), name=name)
    
    def maxpoolLayer(self, inputs, data_format, name):
        return tf.layers.max_pooling2d(inputs, pool_size=2, strides=2, data_format=data_format, name=name)

    def build_vgg16(self, inputs, data_format, num_classes, training):
        conv1_1 = self.convLayer(inputs, 64, data_format, "conv1_1")
        conv1_2 = self.convLayer(conv1_1, 64, data_format, "conv1_2")
        pool1 = self.maxpoolLayer(conv1_2, data_format, "pool1")

        conv2_1 = self.convLayer(pool1, 128, data_format, "conv2_1")
        conv2_2 = self.convLayer(conv2_1, 128, data_format, "conv2_2")
        pool2 = self.maxpoolLayer(conv2_2, data_format, "pool2")

        conv3_1 = self.convLayer(pool2, 256, data_format, "conv3_1")
        conv3_2 = self.convLayer(conv3_1, 256, data_format, "conv3_2")
        conv3_3 = self.convLayer(conv3_2, 256, data_format, "conv3_3")
        pool3 = self.maxpoolLayer(conv3_3, data_format, "pool3")

        conv4_1 = self.convLayer(pool3, 512, data_format, "conv4_1")
        conv4_2 = self.convLayer(conv4_1, 512, data_format, "conv4_2")
        conv4_3 = self.convLayer(conv4_2, 512, data_format, "conv4_3")
        pool4 = self.maxpoolLayer(conv4_3, data_format, "pool4")

        conv5_1 = self.convLayer(pool4, 512, data_format, "conv5_1")
        conv5_2 = self.convLayer(conv5_1, 512, data_format, "conv5_2")
        conv5_3 = self.convLayer(conv5_2, 512, data_format, "conv5_3")
        pool5 = self.maxpoolLayer(conv5_3, data_format, "pool5")

        fcIn = tf.reduce_mean(pool5, axis=[2, 3] if self.data_format == "channels_first" else [1, 2])
        fc6 = tf.layers.dense(fcIn, 4096, name="fc6")
        dropout1 = tf.layers.dropout(fc6, rate=0.7, training=training)

        fc7 = tf.layers.dense(dropout1, 4096, name="fc7")
        dropout2 = tf.layers.dropout(fc7, rate=0.7, training=training)

        fc8 = tf.layers.dense(dropout2, num_classes, name="fc8")
        return fc8

    def build_vgg19(self, inputs, data_format, num_classes, training):
        conv1_1 = self.convLayer(inputs, 64, data_format, "conv1_1")
        conv1_2 = self.convLayer(conv1_1, 64, data_format, "conv1_2")
        pool1 = self.maxpoolLayer(conv1_2, data_format, "pool1")

        conv2_1 = self.convLayer(pool1, 128, data_format, "conv2_1")
        conv2_2 = self.convLayer(conv2_1, 128, data_format, "conv2_2")
        pool2 = self.maxpoolLayer(conv2_2, data_format, "pool2")

        conv3_1 = self.convLayer(pool2, 256, data_format, "conv3_1")
        conv3_2 = self.convLayer(conv3_1, 256, data_format, "conv3_2")
        conv3_3 = self.convLayer(conv3_2, 256, data_format, "conv3_3")
        conv3_4 = self.convLayer(conv3_3, 256, data_format, "conv3_4")
        pool3 = self.maxpoolLayer(conv3_4, data_format, "pool3")

        conv4_1 = self.convLayer(pool3, 512, data_format, "conv4_1")
        conv4_2 = self.convLayer(conv4_1, 512, data_format, "conv4_2")
        conv4_3 = self.convLayer(conv4_2, 512, data_format, "conv4_3")
        conv4_4 = self.convLayer(conv4_3, 512, data_format, "conv4_4")
        pool4 = self.maxpoolLayer(conv4_4, data_format, "pool4")

        conv5_1 = self.convLayer(pool4, 512, data_format, "conv5_1")
        conv5_2 = self.convLayer(conv5_1, 512, data_format, "conv5_2")
        conv5_3 = self.convLayer(conv5_2, 512, data_format, "conv5_3")
        conv5_4 = self.convLayer(conv5_3, 512, data_format, "conv5_4")
        pool5 = self.maxpoolLayer(conv5_4, data_format, "pool5")

        fcIn = tf.reduce_mean(pool5, axis=[2, 3] if self.data_format == "channels_first" else [1, 2])
        fc6 = tf.layers.dense(fcIn, 4096, name="fc6")
        dropout1 = tf.layers.dropout(fc6, rate=0.7, training=training)

        fc7 = tf.layers.dense(dropout1, 4096, name="fc7")
        dropout2 = tf.layers.dropout(fc7, rate=0.7, training=training)

        fc8 = tf.layers.dense(dropout2, num_classes, name="fc8")
        return fc8