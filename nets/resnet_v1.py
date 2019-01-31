import tensorflow as tf

class resnet_v1(object):
    '''Class of resnet v1 model'''
    
    def __init__(self, resnet_size, num_class, data_format):
        self.resnet_size = resnet_size
        self.num_class = num_class
        self.data_format = data_format
    
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
    
    def _conv_bn_leakyLayer(self, inputs, filters, kernel_size=[3,3], strides=[1,1], padding="SAME", training=False, active=True, name=None):
        '''A combination of conv2d, batch_normalization, leaky_relu'''

        #Get the input layer's channels
        channels = int(inputs.get_shape()[-1])
        with tf.variable_scope(name):
            #Conv2d layer
            conv2d = tf.layers.conv2d(inputs, filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=self.data_format)
            #Batch normalization layer
            bn_layer = tf.layers.batch_normalization(conv2d, axis=1 if self.data_format == 'channels_first' else 3, training=training)
            #active layer
            if active:
                active_layer = tf.nn.leaky_relu(bn_layer)
                return active_layer
            else:
                return bn_layer
    
    def _bottleneckLayer(self, inputs, filters, output_layers, pooling=False, training=False, name=None):
        with tf.variable_scope(name):
            if pooling:
                layer1 = self._conv_bn_leakyLayer(inputs, filters, kernel_size=[1,1], strides=[2,2], training=training, name="conv_bn_leaky_1")
            else:
                layer1 = self._conv_bn_leakyLayer(inputs, filters, kernel_size=[1,1], training=training, name="conv_bn_leaky_1")
            layer2 = self._conv_bn_leakyLayer(layer1, filters, kernel_size=[3,3], training=training, name="conv_bn_leaky_2")
            layer3 = self._conv_bn_leakyLayer(layer2, output_layers, kernel_size=[1,1], training=training, active=False, name="conv_bn")
        return layer3

    def shortcut(self, input1, input2, name):
        with tf.name_scope(name):
            output = input1 + input2
            return output
    
    def build_resnet50(self, inputs, training, num_class):
        '''Assume the input shape is [None, 224, 224, 3]'''
        with tf.variable_scope("scale1"):
            conv1_1 = self._conv_bn_leakyLayer(inputs, 64, kernel_size=[7,7], strides=[2,2], training=training, name="block1") #[None, 112, 112, 3]
            pool1 = tf.layers.max_pooling2d(conv1_1, pool_size=[3,3], strides=[2,2], padding='same', data_format=self.data_format, name='pool1') #[None, 56, 56, 3]

        with tf.variable_scope("scale2"):
            conv2_1_branch1 = self._bottleneckLayer(pool1, 64, 256, training=training, name="block1") #[None ,56, 56, 256]
            conv2_1_conv_branch2 = tf.layers.conv2d(pool1, 256, kernel_size=[1,1], data_format=self.data_format, name="identify_conv") #[None ,56, 56, 256]
            conv2_1_bn_branch2 = tf.layers.batch_normalization(conv2_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,56, 56, 256] 
            conv2_1_add = self.shortcut(conv2_1_branch1, conv2_1_bn_branch2, name="shortcut1") #[None ,56, 56, 256]
            conv2_1 = tf.nn.relu(conv2_1_add) #[None ,56, 56, 256]

            conv2_2_branch1 = self._bottleneckLayer(conv2_1, 64, 256, training=training, name="block2") #[None ,56, 56, 256]
            conv2_2_add = self.shortcut(conv2_2_branch1, conv2_1, name="shortcut2") #[None ,56, 56, 256]
            conv2_2 = tf.nn.relu(conv2_2_add) #[None ,56, 56, 256]

            conv2_3_branch1 = self._bottleneckLayer(conv2_2, 64, 256, training=training, name="block3") #[None ,56, 56, 256]
            conv2_3_add =self.shortcut(conv2_3_branch1, conv2_2, name="shortcut3") #[None ,56, 56, 256]
            conv2_3 = tf.nn.relu(conv2_3_add) #[None ,56, 56, 256]

        with tf.variable_scope("scale3"):
            conv3_1_branch1 = self._bottleneckLayer(conv2_3, 128, 512, pooling=True, training=training, name="block1") #[None ,28, 28, 512]
            conv3_1_conv_branch2 = tf.layers.conv2d(conv2_3, 512, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,28, 28, 512]
            conv3_1_bn_branch2 = tf.layers.batch_normalization(conv3_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,28, 28, 512]
            conv3_1_add = self.shortcut(conv3_1_branch1, conv3_1_bn_branch2, name="shortcut4") #[None ,28, 28, 512]
            conv3_1 = tf.nn.relu(conv3_1_add) #[None ,28, 28, 512]

            conv3_2_branch1 = self._bottleneckLayer(conv3_1, 128, 512, training=training, name="block2") #[None ,28, 28, 512]
            conv3_2_add = self.shortcut(conv3_2_branch1, conv3_1, name="shortcut5") #[None ,28, 28, 512]
            conv3_2 = tf.nn.relu(conv3_2_add) #[None ,28, 28, 512]

            conv3_3_branch1 = self._bottleneckLayer(conv3_2, 128, 512, training=training, name="block3") #[None ,28, 28, 512]
            conv3_3_add =self.shortcut(conv3_3_branch1, conv3_2, name="shortcut6") #[None ,28, 28, 512]
            conv3_3 = tf.nn.relu(conv3_3_add) #[None ,28, 28, 512]

            conv3_4_branch1 = self._bottleneckLayer(conv3_3, 128, 512, training=training, name="block4") #[None ,28, 28, 512]
            conv3_4_add =self.shortcut(conv3_4_branch1, conv3_3, name="shortcut7") #[None ,28, 28, 512]
            conv3_4 = tf.nn.relu(conv3_4_add) #[None ,28, 28, 512]

        with tf.variable_scope("scale4"):
            conv4_1_branch1 = self._bottleneckLayer(conv3_4, 256, 1024, pooling=True, training=training, name="block1") #[None ,14, 14, 1024]
            conv4_1_conv_branch2 = tf.layers.conv2d(conv3_4, 1024, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,14, 14, 1024]
            conv4_1_bn_branch2 = tf.layers.batch_normalization(conv4_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,14, 14, 1024]
            conv4_1_add = self.shortcut(conv4_1_branch1, conv4_1_bn_branch2, name="shortcut8") #[None ,14, 14, 1024]
            conv4_1 = tf.nn.relu(conv4_1_add) #[None ,14, 14, 1024]

            conv4_2_branch1 = self._bottleneckLayer(conv4_1, 256, 1024, training=training, name="block2") #[None ,14, 14, 1024]
            conv4_2_add = self.shortcut(conv4_2_branch1, conv4_1, name="shortcut9") #[None ,14, 14, 1024]
            conv4_2 = tf.nn.relu(conv4_2_add) #[None ,14, 14, 1024]

            conv4_3_branch1 = self._bottleneckLayer(conv4_2, 256, 1024, training=training, name="block3") #[None ,14, 14, 1024]
            conv4_3_add = self.shortcut(conv4_3_branch1, conv4_2, name="shortcut10") #[None ,14, 14, 1024]
            conv4_3 = tf.nn.relu(conv4_3_add) #[None ,14, 14, 1024]

            conv4_4_branch1 = self._bottleneckLayer(conv4_3, 256, 1024, training=training, name="block4") #[None ,14, 14, 1024]
            conv4_4_add = self.shortcut(conv4_4_branch1, conv4_3, name="shortcut11") #[None ,14, 14, 1024]
            conv4_4 = tf.nn.relu(conv4_4_add) #[None ,14, 14, 1024]

            conv4_5_branch1 = self._bottleneckLayer(conv4_4, 256, 1024, training=training, name="block5") #[None ,14, 14, 1024]
            conv4_5_add = self.shortcut(conv4_5_branch1, conv4_4, name="shortcut12") #[None ,14, 14, 1024]
            conv4_5 = tf.nn.relu(conv4_5_add) #[None ,14, 14, 1024]

            conv4_6_branch1 = self._bottleneckLayer(conv4_5, 256, 1024, training=training, name="block6") #[None ,14, 14, 1024]
            conv4_6_add = self.shortcut(conv4_6_branch1, conv4_5, name="shortcut13") #[None ,14, 14, 1024]
            conv4_6 = tf.nn.relu(conv4_6_add) #[None ,14, 14, 1024]
        
        with tf.variable_scope("scale5"):
            conv5_1_branch1 = self._bottleneckLayer(conv4_6, 512, 2048, pooling=True, training=training, name="block1") #[None ,7, 7, 2048]
            conv5_1_conv_branch2 = tf.layers.conv2d(conv4_6, 2048, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,7, 7, 2048]
            conv5_1_bn_branch2 = tf.layers.batch_normalization(conv5_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,7, 7, 2048]
            conv5_1_add = self.shortcut(conv5_1_branch1, conv5_1_bn_branch2, name="shortcut14") #[None ,7, 7, 2048]
            conv5_1 = tf.nn.relu(conv5_1_add) #[None ,7, 7, 2048]

            conv5_2_branch1 = self._bottleneckLayer(conv5_1, 512, 2048, training=training, name="block2") #[None ,7, 7, 2048]
            conv5_2_add = self.shortcut(conv5_2_branch1, conv5_1, name="shortcut15") #[None ,7, 7, 2048]
            conv5_2 = tf.nn.relu(conv5_2_add) #[None ,7, 7, 2048]

            conv5_3_branch1 = self._bottleneckLayer(conv5_2, 512, 2048, training=training, name="block3") #[None ,7, 7, 2048]
            conv5_3_add = self.shortcut(conv5_3_branch1, conv5_2, name="shortcut16") #[None ,7, 7, 2048]
            conv5_3 = tf.nn.relu(conv5_3_add) #[None ,7, 7, 2048]
        
        with tf.variable_scope("fc"):
            avg_pool = tf.reduce_mean(conv5_3, axis=[2,3] if self.data_format == 'channels_first' else [1,2], keepdims=True) #[None, 1, 1, 2048]
            flatten_layer = tf.layers.flatten(avg_pool, name="flattenLayer") #[None, 2048]
            output = tf.layers.dense(flatten_layer, num_class, name="fcLayer")#[None, num_class]
        return output
    
    def build_resnet101(self, inputs, training, num_class):
        '''Assume the input shape is [None, 224, 224, 3]'''
        with tf.variable_scope("scale1"):
            conv1_1 = self._conv_bn_leakyLayer(inputs, 64, kernel_size=[7,7], strides=[2,2], training=training, name="block1") #[None, 112, 112, 3]
            pool1 = tf.layers.max_pooling2d(conv1_1, pool_size=[3,3], strides=[2,2], padding='same', data_format=self.data_format, name='pool1') #[None, 56, 56, 3]

        with tf.variable_scope("scale2"):
            conv2_1_branch1 = self._bottleneckLayer(pool1, 64, 256, training=training, name="block1") #[None ,56, 56, 256]
            conv2_1_conv_branch2 = tf.layers.conv2d(pool1, 256, kernel_size=[1,1], data_format=self.data_format, name="identify_conv") #[None ,56, 56, 256]
            conv2_1_bn_branch2 = tf.layers.batch_normalization(conv2_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,56, 56, 256] 
            conv2_1_add = self.shortcut(conv2_1_branch1, conv2_1_bn_branch2, name="shortcut1") #[None ,56, 56, 256]
            conv2_1 = tf.nn.relu(conv2_1_add) #[None ,56, 56, 256]

            conv2_2_branch1 = self._bottleneckLayer(conv2_1, 64, 256, training=training, name="block2") #[None ,56, 56, 256]
            conv2_2_add = self.shortcut(conv2_2_branch1, conv2_1, name="shortcut2") #[None ,56, 56, 256]
            conv2_2 = tf.nn.relu(conv2_2_add) #[None ,56, 56, 256]

            conv2_3_branch1 = self._bottleneckLayer(conv2_2, 64, 256, training=training, name="block3") #[None ,56, 56, 256]
            conv2_3_add =self.shortcut(conv2_3_branch1, conv2_2, name="shortcut3") #[None ,56, 56, 256]
            conv2_3 = tf.nn.relu(conv2_3_add) #[None ,56, 56, 256]

        with tf.variable_scope("scale3"):
            conv3_1_branch1 = self._bottleneckLayer(conv2_3, 128, 512, pooling=True, training=training, name="block1") #[None ,28, 28, 512]
            conv3_1_conv_branch2 = tf.layers.conv2d(conv2_3, 512, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,28, 28, 512]
            conv3_1_bn_branch2 = tf.layers.batch_normalization(conv3_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,28, 28, 512]
            conv3_1_add = self.shortcut(conv3_1_branch1, conv3_1_bn_branch2, name="shortcut4") #[None ,28, 28, 512]
            conv3_1 = tf.nn.relu(conv3_1_add) #[None ,28, 28, 512]

            conv3_2_branch1 = self._bottleneckLayer(conv3_1, 128, 512, training=training, name="block2") #[None ,28, 28, 512]
            conv3_2_add = self.shortcut(conv3_2_branch1, conv3_1, name="shortcut5") #[None ,28, 28, 512]
            conv3_2 = tf.nn.relu(conv3_2_add) #[None ,28, 28, 512]

            conv3_3_branch1 = self._bottleneckLayer(conv3_2, 128, 512, training=training, name="block3") #[None ,28, 28, 512]
            conv3_3_add =self.shortcut(conv3_3_branch1, conv3_2, name="shortcut6") #[None ,28, 28, 512]
            conv3_3 = tf.nn.relu(conv3_3_add) #[None ,28, 28, 512]

            conv3_4_branch1 = self._bottleneckLayer(conv3_3, 128, 512, training=training, name="block4") #[None ,28, 28, 512]
            conv3_4_add =self.shortcut(conv3_4_branch1, conv3_3, name="shortcut7") #[None ,28, 28, 512]
            conv3_4 = tf.nn.relu(conv3_4_add) #[None ,28, 28, 512]

        with tf.variable_scope("scale4"):
            conv4_1_branch1 = self._bottleneckLayer(conv3_4, 256, 1024, pooling=True, training=training, name="block1") #[None ,14, 14, 1024]
            conv4_1_conv_branch2 = tf.layers.conv2d(conv3_4, 1024, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,14, 14, 1024]
            conv4_1_bn_branch2 = tf.layers.batch_normalization(conv4_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,14, 14, 1024]
            conv4_1_add = self.shortcut(conv4_1_branch1, conv4_1_bn_branch2, name="shortcut8") #[None ,14, 14, 1024]
            conv4_1 = tf.nn.relu(conv4_1_add) #[None ,14, 14, 1024]

            conv4_2_branch1 = self._bottleneckLayer(conv4_1, 256, 1024, training=training, name="block2") #[None ,14, 14, 1024]
            conv4_2_add = self.shortcut(conv4_2_branch1, conv4_1, name="shortcut9") #[None ,14, 14, 1024]
            conv4_2 = tf.nn.relu(conv4_2_add) #[None ,14, 14, 1024]

            conv4_3_branch1 = self._bottleneckLayer(conv4_2, 256, 1024, training=training, name="block3") #[None ,14, 14, 1024]
            conv4_3_add = self.shortcut(conv4_3_branch1, conv4_2, name="shortcut10") #[None ,14, 14, 1024]
            conv4_3 = tf.nn.relu(conv4_3_add) #[None ,14, 14, 1024]

            conv4_4_branch1 = self._bottleneckLayer(conv4_3, 256, 1024, training=training, name="block4") #[None ,14, 14, 1024]
            conv4_4_add = self.shortcut(conv4_4_branch1, conv4_3, name="shortcut11") #[None ,14, 14, 1024]
            conv4_4 = tf.nn.relu(conv4_4_add) #[None ,14, 14, 1024]

            conv4_5_branch1 = self._bottleneckLayer(conv4_4, 256, 1024, training=training, name="block5") #[None ,14, 14, 1024]
            conv4_5_add = self.shortcut(conv4_5_branch1, conv4_4, name="shortcut12") #[None ,14, 14, 1024]
            conv4_5 = tf.nn.relu(conv4_5_add) #[None ,14, 14, 1024]

            conv4_6_branch1 = self._bottleneckLayer(conv4_5, 256, 1024, training=training, name="block6") #[None ,14, 14, 1024]
            conv4_6_add = self.shortcut(conv4_6_branch1, conv4_5, name="shortcut13") #[None ,14, 14, 1024]
            conv4_6 = tf.nn.relu(conv4_6_add) #[None ,14, 14, 1024]

            conv4_7_branch1 = self._bottleneckLayer(conv4_6, 256, 1024, training=training, name="block7") #[None ,14, 14, 1024]
            conv4_7_add = self.shortcut(conv4_7_branch1, conv4_6, name="shortcut14") #[None ,14, 14, 1024]
            conv4_7 = tf.nn.relu(conv4_7_add) #[None ,14, 14, 1024]

            conv4_8_branch1 = self._bottleneckLayer(conv4_7, 256, 1024, training=training, name="block8") #[None ,14, 14, 1024]
            conv4_8_add = self.shortcut(conv4_8_branch1, conv4_7, name="shortcut15") #[None ,14, 14, 1024]
            conv4_8 = tf.nn.relu(conv4_8_add) #[None ,14, 14, 1024]

            conv4_9_branch1 = self._bottleneckLayer(conv4_8, 256, 1024, training=training, name="block9") #[None ,14, 14, 1024]
            conv4_9_add = self.shortcut(conv4_9_branch1, conv4_8, name="shortcut16") #[None ,14, 14, 1024]
            conv4_9 = tf.nn.relu(conv4_9_add) #[None ,14, 14, 1024]

            conv4_10_branch1 = self._bottleneckLayer(conv4_9, 256, 1024, training=training, name="block10") #[None ,14, 14, 1024]
            conv4_10_add = self.shortcut(conv4_10_branch1, conv4_9, name="shortcut17") #[None ,14, 14, 1024]
            conv4_10 = tf.nn.relu(conv4_10_add) #[None ,14, 14, 1024]

            conv4_11_branch1 = self._bottleneckLayer(conv4_10, 256, 1024, training=training, name="block11") #[None ,14, 14, 1024]
            conv4_11_add = self.shortcut(conv4_11_branch1, conv4_10, name="shortcut18") #[None ,14, 14, 1024]
            conv4_11 = tf.nn.relu(conv4_11_add) #[None ,14, 14, 1024]

            conv4_12_branch1 = self._bottleneckLayer(conv4_11, 256, 1024, training=training, name="block12") #[None ,14, 14, 1024]
            conv4_12_add = self.shortcut(conv4_12_branch1, conv4_11, name="shortcut19") #[None ,14, 14, 1024]
            conv4_12 = tf.nn.relu(conv4_12_add) #[None ,14, 14, 1024]

            conv4_13_branch1 = self._bottleneckLayer(conv4_12, 256, 1024, training=training, name="block13") #[None ,14, 14, 1024]
            conv4_13_add = self.shortcut(conv4_13_branch1, conv4_12, name="shortcut20") #[None ,14, 14, 1024]
            conv4_13 = tf.nn.relu(conv4_13_add) #[None ,14, 14, 1024]

            conv4_14_branch1 = self._bottleneckLayer(conv4_13, 256, 1024, training=training, name="block14") #[None ,14, 14, 1024]
            conv4_14_add = self.shortcut(conv4_14_branch1, conv4_13, name="shortcut21") #[None ,14, 14, 1024]
            conv4_14 = tf.nn.relu(conv4_14_add) #[None ,14, 14, 1024]

            conv4_15_branch1 = self._bottleneckLayer(conv4_14, 256, 1024, training=training, name="block15") #[None ,14, 14, 1024]
            conv4_15_add = self.shortcut(conv4_15_branch1, conv4_14, name="shortcut22") #[None ,14, 14, 1024]
            conv4_15 = tf.nn.relu(conv4_15_add) #[None ,14, 14, 1024]

            conv4_16_branch1 = self._bottleneckLayer(conv4_15, 256, 1024, training=training, name="block16") #[None ,14, 14, 1024]
            conv4_16_add = self.shortcut(conv4_16_branch1, conv4_15, name="shortcut23") #[None ,14, 14, 1024]
            conv4_16 = tf.nn.relu(conv4_16_add) #[None ,14, 14, 1024]

            conv4_17_branch1 = self._bottleneckLayer(conv4_16, 256, 1024, training=training, name="block17") #[None ,14, 14, 1024]
            conv4_17_add = self.shortcut(conv4_17_branch1, conv4_16, name="shortcut24") #[None ,14, 14, 1024]
            conv4_17 = tf.nn.relu(conv4_17_add) #[None ,14, 14, 1024]

            conv4_18_branch1 = self._bottleneckLayer(conv4_17, 256, 1024, training=training, name="block18") #[None ,14, 14, 1024]
            conv4_18_add = self.shortcut(conv4_18_branch1, conv4_17, name="shortcut25") #[None ,14, 14, 1024]
            conv4_18 = tf.nn.relu(conv4_18_add) #[None ,14, 14, 1024]

            conv4_19_branch1 = self._bottleneckLayer(conv4_18, 256, 1024, training=training, name="block19") #[None ,14, 14, 1024]
            conv4_19_add = self.shortcut(conv4_19_branch1, conv4_18, name="shortcut26") #[None ,14, 14, 1024]
            conv4_19 = tf.nn.relu(conv4_19_add) #[None ,14, 14, 1024]

            conv4_20_branch1 = self._bottleneckLayer(conv4_19, 256, 1024, training=training, name="block20") #[None ,14, 14, 1024]
            conv4_20_add = self.shortcut(conv4_20_branch1, conv4_19, name="shortcut27") #[None ,14, 14, 1024]
            conv4_20 = tf.nn.relu(conv4_20_add) #[None ,14, 14, 1024]

            conv4_21_branch1 = self._bottleneckLayer(conv4_20, 256, 1024, training=training, name="block21") #[None ,14, 14, 1024]
            conv4_21_add = self.shortcut(conv4_21_branch1, conv4_20, name="shortcut28") #[None ,14, 14, 1024]
            conv4_21 = tf.nn.relu(conv4_21_add) #[None ,14, 14, 1024]

            conv4_22_branch1 = self._bottleneckLayer(conv4_21, 256, 1024, training=training, name="block22") #[None ,14, 14, 1024]
            conv4_22_add = self.shortcut(conv4_22_branch1, conv4_21, name="shortcut29") #[None ,14, 14, 1024]
            conv4_22 = tf.nn.relu(conv4_22_add) #[None ,14, 14, 1024]

            conv4_23_branch1 = self._bottleneckLayer(conv4_22, 256, 1024, training=training, name="block23") #[None ,14, 14, 1024]
            conv4_23_add = self.shortcut(conv4_23_branch1, conv4_22, name="shortcut30") #[None ,14, 14, 1024]
            conv4_23 = tf.nn.relu(conv4_23_add) #[None ,14, 14, 1024]
        
        with tf.variable_scope("scale5"):
            conv5_1_branch1 = self._bottleneckLayer(conv4_23, 512, 2048, pooling=True, training=training, name="block1") #[None ,7, 7, 2048]
            conv5_1_conv_branch2 = tf.layers.conv2d(conv4_23, 2048, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,7, 7, 2048]
            conv5_1_bn_branch2 = tf.layers.batch_normalization(conv5_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,7, 7, 2048]
            conv5_1_add = self.shortcut(conv5_1_branch1, conv5_1_bn_branch2, name="shortcut31") #[None ,7, 7, 2048]
            conv5_1 = tf.nn.relu(conv5_1_add) #[None ,7, 7, 2048]

            conv5_2_branch1 = self._bottleneckLayer(conv5_1, 512, 2048, training=training, name="block2") #[None ,7, 7, 2048]
            conv5_2_add = self.shortcut(conv5_2_branch1, conv5_1, name="shortcut32") #[None ,7, 7, 2048]
            conv5_2 = tf.nn.relu(conv5_2_add) #[None ,7, 7, 2048]

            conv5_3_branch1 = self._bottleneckLayer(conv5_2, 512, 2048, training=training, name="block3") #[None ,7, 7, 2048]
            conv5_3_add = self.shortcut(conv5_3_branch1, conv5_2, name="shortcut33") #[None ,7, 7, 2048]
            conv5_3 = tf.nn.relu(conv5_3_add) #[None ,7, 7, 2048]
        
        with tf.variable_scope("fc"):
            avg_pool = tf.reduce_mean(conv5_3, axis=[2,3] if self.data_format == 'channels_first' else [1,2], keepdims=True) #[None, 1, 1, 2048]
            flatten_layer = tf.layers.flatten(avg_pool, name="flattenLayer") #[None, 2048]
            output = tf.layers.dense(flatten_layer, num_class, name="fcLayer")#[None, num_class]
        return output
    
    def build_resnet152(self, inputs, training, num_class):
        '''Assume the input shape is [None, 224, 224, 3]'''
        with tf.variable_scope("scale1"):
            conv1_1 = self._conv_bn_leakyLayer(inputs, 64, kernel_size=[7,7], strides=[2,2], training=training, name="block1") #[None, 112, 112, 3]
            pool1 = tf.layers.max_pooling2d(conv1_1, pool_size=[3,3], strides=[2,2], padding='same', data_format=self.data_format, name='pool1') #[None, 56, 56, 3]

        with tf.variable_scope("scale2"):
            conv2_1_branch1 = self._bottleneckLayer(pool1, 64, 256, training=training, name="block1") #[None ,56, 56, 256]
            conv2_1_conv_branch2 = tf.layers.conv2d(pool1, 256, kernel_size=[1,1], data_format=self.data_format, name="identify_conv") #[None ,56, 56, 256]
            conv2_1_bn_branch2 = tf.layers.batch_normalization(conv2_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,56, 56, 256] 
            conv2_1_add = self.shortcut(conv2_1_branch1, conv2_1_bn_branch2, name="shortcut1") #[None ,56, 56, 256]
            conv2_1 = tf.nn.relu(conv2_1_add) #[None ,56, 56, 256]

            conv2_2_branch1 = self._bottleneckLayer(conv2_1, 64, 256, training=training, name="block2") #[None ,56, 56, 256]
            conv2_2_add = self.shortcut(conv2_2_branch1, conv2_1, name="shortcut2") #[None ,56, 56, 256]
            conv2_2 = tf.nn.relu(conv2_2_add) #[None ,56, 56, 256]

            conv2_3_branch1 = self._bottleneckLayer(conv2_2, 64, 256, training=training, name="block3") #[None ,56, 56, 256]
            conv2_3_add =self.shortcut(conv2_3_branch1, conv2_2, name="shortcut3") #[None ,56, 56, 256]
            conv2_3 = tf.nn.relu(conv2_3_add) #[None ,56, 56, 256]

        with tf.variable_scope("scale3"):
            conv3_1_branch1 = self._bottleneckLayer(conv2_3, 128, 512, pooling=True, training=training, name="block1") #[None ,28, 28, 512]
            conv3_1_conv_branch2 = tf.layers.conv2d(conv2_3, 512, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,28, 28, 512]
            conv3_1_bn_branch2 = tf.layers.batch_normalization(conv3_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,28, 28, 512]
            conv3_1_add = self.shortcut(conv3_1_branch1, conv3_1_bn_branch2, name="shortcut4") #[None ,28, 28, 512]
            conv3_1 = tf.nn.relu(conv3_1_add) #[None ,28, 28, 512]

            conv3_2_branch1 = self._bottleneckLayer(conv3_1, 128, 512, training=training, name="block2") #[None ,28, 28, 512]
            conv3_2_add = self.shortcut(conv3_2_branch1, conv3_1, name="shortcut5") #[None ,28, 28, 512]
            conv3_2 = tf.nn.relu(conv3_2_add) #[None ,28, 28, 512]

            conv3_3_branch1 = self._bottleneckLayer(conv3_2, 128, 512, training=training, name="block3") #[None ,28, 28, 512]
            conv3_3_add =self.shortcut(conv3_3_branch1, conv3_2, name="shortcut6") #[None ,28, 28, 512]
            conv3_3 = tf.nn.relu(conv3_3_add) #[None ,28, 28, 512]

            conv3_4_branch1 = self._bottleneckLayer(conv3_3, 128, 512, training=training, name="block4") #[None ,28, 28, 512]
            conv3_4_add =self.shortcut(conv3_4_branch1, conv3_3, name="shortcut7") #[None ,28, 28, 512]
            conv3_4 = tf.nn.relu(conv3_4_add) #[None ,28, 28, 512]

            conv3_5_branch1 = self._bottleneckLayer(conv3_4, 128, 512, training=training, name="block5") #[None ,28, 28, 512]
            conv3_5_add =self.shortcut(conv3_5_branch1, conv3_4, name="shortcut8") #[None ,28, 28, 512]
            conv3_5 = tf.nn.relu(conv3_5_add) #[None ,28, 28, 512]

            conv3_6_branch1 = self._bottleneckLayer(conv3_5, 128, 512, training=training, name="block6") #[None ,28, 28, 512]
            conv3_6_add =self.shortcut(conv3_6_branch1, conv3_5, name="shortcut9") #[None ,28, 28, 512]
            conv3_6 = tf.nn.relu(conv3_6_add) #[None ,28, 28, 512]

            conv3_7_branch1 = self._bottleneckLayer(conv3_6, 128, 512, training=training, name="block7") #[None ,28, 28, 512]
            conv3_7_add =self.shortcut(conv3_7_branch1, conv3_6, name="shortcut10") #[None ,28, 28, 512]
            conv3_7 = tf.nn.relu(conv3_7_add) #[None ,28, 28, 512]

            conv3_8_branch1 = self._bottleneckLayer(conv3_7, 128, 512, training=training, name="block8") #[None ,28, 28, 512]
            conv3_8_add =self.shortcut(conv3_8_branch1, conv3_7, name="shortcut11") #[None ,28, 28, 512]
            conv3_8 = tf.nn.relu(conv3_8_add) #[None ,28, 28, 512]

        with tf.variable_scope("scale4"):
            conv4_1_branch1 = self._bottleneckLayer(conv3_8, 256, 1024, pooling=True, training=training, name="block1") #[None ,14, 14, 1024]
            conv4_1_conv_branch2 = tf.layers.conv2d(conv3_8, 1024, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,14, 14, 1024]
            conv4_1_bn_branch2 = tf.layers.batch_normalization(conv4_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,14, 14, 1024]
            conv4_1_add = self.shortcut(conv4_1_branch1, conv4_1_bn_branch2, name="shortcut12") #[None ,14, 14, 1024]
            conv4_1 = tf.nn.relu(conv4_1_add) #[None ,14, 14, 1024]

            conv4_2_branch1 = self._bottleneckLayer(conv4_1, 256, 1024, training=training, name="block2") #[None ,14, 14, 1024]
            conv4_2_add = self.shortcut(conv4_2_branch1, conv4_1, name="shortcut13") #[None ,14, 14, 1024]
            conv4_2 = tf.nn.relu(conv4_2_add) #[None ,14, 14, 1024]

            conv4_3_branch1 = self._bottleneckLayer(conv4_2, 256, 1024, training=training, name="block3") #[None ,14, 14, 1024]
            conv4_3_add = self.shortcut(conv4_3_branch1, conv4_2, name="shortcut14") #[None ,14, 14, 1024]
            conv4_3 = tf.nn.relu(conv4_3_add) #[None ,14, 14, 1024]

            conv4_4_branch1 = self._bottleneckLayer(conv4_3, 256, 1024, training=training, name="block4") #[None ,14, 14, 1024]
            conv4_4_add = self.shortcut(conv4_4_branch1, conv4_3, name="shortcut15") #[None ,14, 14, 1024]
            conv4_4 = tf.nn.relu(conv4_4_add) #[None ,14, 14, 1024]

            conv4_5_branch1 = self._bottleneckLayer(conv4_4, 256, 1024, training=training, name="block5") #[None ,14, 14, 1024]
            conv4_5_add = self.shortcut(conv4_5_branch1, conv4_4, name="shortcut16") #[None ,14, 14, 1024]
            conv4_5 = tf.nn.relu(conv4_5_add) #[None ,14, 14, 1024]

            conv4_6_branch1 = self._bottleneckLayer(conv4_5, 256, 1024, training=training, name="block6") #[None ,14, 14, 1024]
            conv4_6_add = self.shortcut(conv4_6_branch1, conv4_5, name="shortcut17") #[None ,14, 14, 1024]
            conv4_6 = tf.nn.relu(conv4_6_add) #[None ,14, 14, 1024]

            conv4_7_branch1 = self._bottleneckLayer(conv4_6, 256, 1024, training=training, name="block7") #[None ,14, 14, 1024]
            conv4_7_add = self.shortcut(conv4_7_branch1, conv4_6, name="shortcut18") #[None ,14, 14, 1024]
            conv4_7 = tf.nn.relu(conv4_7_add) #[None ,14, 14, 1024]

            conv4_8_branch1 = self._bottleneckLayer(conv4_7, 256, 1024, training=training, name="block8") #[None ,14, 14, 1024]
            conv4_8_add = self.shortcut(conv4_8_branch1, conv4_7, name="shortcut19") #[None ,14, 14, 1024]
            conv4_8 = tf.nn.relu(conv4_8_add) #[None ,14, 14, 1024]

            conv4_9_branch1 = self._bottleneckLayer(conv4_8, 256, 1024, training=training, name="block9") #[None ,14, 14, 1024]
            conv4_9_add = self.shortcut(conv4_9_branch1, conv4_8, name="shortcut20") #[None ,14, 14, 1024]
            conv4_9 = tf.nn.relu(conv4_9_add) #[None ,14, 14, 1024]

            conv4_10_branch1 = self._bottleneckLayer(conv4_9, 256, 1024, training=training, name="block10") #[None ,14, 14, 1024]
            conv4_10_add = self.shortcut(conv4_10_branch1, conv4_9, name="shortcut21") #[None ,14, 14, 1024]
            conv4_10 = tf.nn.relu(conv4_10_add) #[None ,14, 14, 1024]

            conv4_11_branch1 = self._bottleneckLayer(conv4_10, 256, 1024, training=training, name="block11") #[None ,14, 14, 1024]
            conv4_11_add = self.shortcut(conv4_11_branch1, conv4_10, name="shortcut22") #[None ,14, 14, 1024]
            conv4_11 = tf.nn.relu(conv4_11_add) #[None ,14, 14, 1024]

            conv4_12_branch1 = self._bottleneckLayer(conv4_11, 256, 1024, training=training, name="block12") #[None ,14, 14, 1024]
            conv4_12_add = self.shortcut(conv4_12_branch1, conv4_11, name="shortcut23") #[None ,14, 14, 1024]
            conv4_12 = tf.nn.relu(conv4_12_add) #[None ,14, 14, 1024]

            conv4_13_branch1 = self._bottleneckLayer(conv4_12, 256, 1024, training=training, name="block13") #[None ,14, 14, 1024]
            conv4_13_add = self.shortcut(conv4_13_branch1, conv4_12, name="shortcut24") #[None ,14, 14, 1024]
            conv4_13 = tf.nn.relu(conv4_13_add) #[None ,14, 14, 1024]

            conv4_14_branch1 = self._bottleneckLayer(conv4_13, 256, 1024, training=training, name="block14") #[None ,14, 14, 1024]
            conv4_14_add = self.shortcut(conv4_14_branch1, conv4_13, name="shortcut25") #[None ,14, 14, 1024]
            conv4_14 = tf.nn.relu(conv4_14_add) #[None ,14, 14, 1024]

            conv4_15_branch1 = self._bottleneckLayer(conv4_14, 256, 1024, training=training, name="block15") #[None ,14, 14, 1024]
            conv4_15_add = self.shortcut(conv4_15_branch1, conv4_14, name="shortcut26") #[None ,14, 14, 1024]
            conv4_15 = tf.nn.relu(conv4_15_add) #[None ,14, 14, 1024]

            conv4_16_branch1 = self._bottleneckLayer(conv4_15, 256, 1024, training=training, name="block16") #[None ,14, 14, 1024]
            conv4_16_add = self.shortcut(conv4_16_branch1, conv4_15, name="shortcut27") #[None ,14, 14, 1024]
            conv4_16 = tf.nn.relu(conv4_16_add) #[None ,14, 14, 1024]

            conv4_17_branch1 = self._bottleneckLayer(conv4_16, 256, 1024, training=training, name="block17") #[None ,14, 14, 1024]
            conv4_17_add = self.shortcut(conv4_17_branch1, conv4_16, name="shortcut28") #[None ,14, 14, 1024]
            conv4_17 = tf.nn.relu(conv4_17_add) #[None ,14, 14, 1024]

            conv4_18_branch1 = self._bottleneckLayer(conv4_17, 256, 1024, training=training, name="block18") #[None ,14, 14, 1024]
            conv4_18_add = self.shortcut(conv4_18_branch1, conv4_17, name="shortcut29") #[None ,14, 14, 1024]
            conv4_18 = tf.nn.relu(conv4_18_add) #[None ,14, 14, 1024]

            conv4_19_branch1 = self._bottleneckLayer(conv4_18, 256, 1024, training=training, name="block19") #[None ,14, 14, 1024]
            conv4_19_add = self.shortcut(conv4_19_branch1, conv4_18, name="shortcut30") #[None ,14, 14, 1024]
            conv4_19 = tf.nn.relu(conv4_19_add) #[None ,14, 14, 1024]

            conv4_20_branch1 = self._bottleneckLayer(conv4_19, 256, 1024, training=training, name="block20") #[None ,14, 14, 1024]
            conv4_20_add = self.shortcut(conv4_20_branch1, conv4_19, name="shortcut31") #[None ,14, 14, 1024]
            conv4_20 = tf.nn.relu(conv4_20_add) #[None ,14, 14, 1024]

            conv4_21_branch1 = self._bottleneckLayer(conv4_20, 256, 1024, training=training, name="block21") #[None ,14, 14, 1024]
            conv4_21_add = self.shortcut(conv4_21_branch1, conv4_20, name="shortcut32") #[None ,14, 14, 1024]
            conv4_21 = tf.nn.relu(conv4_21_add) #[None ,14, 14, 1024]

            conv4_22_branch1 = self._bottleneckLayer(conv4_21, 256, 1024, training=training, name="block22") #[None ,14, 14, 1024]
            conv4_22_add = self.shortcut(conv4_22_branch1, conv4_21, name="shortcut33") #[None ,14, 14, 1024]
            conv4_22 = tf.nn.relu(conv4_22_add) #[None ,14, 14, 1024]

            conv4_23_branch1 = self._bottleneckLayer(conv4_22, 256, 1024, training=training, name="block23") #[None ,14, 14, 1024]
            conv4_23_add = self.shortcut(conv4_23_branch1, conv4_22, name="shortcut34") #[None ,14, 14, 1024]
            conv4_23 = tf.nn.relu(conv4_23_add) #[None ,14, 14, 1024]

            conv4_24_branch1 = self._bottleneckLayer(conv4_23, 256, 1024, training=training, name="block24") #[None ,14, 14, 1024]
            conv4_24_add = self.shortcut(conv4_24_branch1, conv4_23, name="shortcut35") #[None ,14, 14, 1024]
            conv4_24 = tf.nn.relu(conv4_24_add) #[None ,14, 14, 1024]

            conv4_25_branch1 = self._bottleneckLayer(conv4_24, 256, 1024, training=training, name="block25") #[None ,14, 14, 1024]
            conv4_25_add = self.shortcut(conv4_25_branch1, conv4_24, name="shortcut36") #[None ,14, 14, 1024]
            conv4_25 = tf.nn.relu(conv4_25_add) #[None ,14, 14, 1024]

            conv4_26_branch1 = self._bottleneckLayer(conv4_25, 256, 1024, training=training, name="block26") #[None ,14, 14, 1024]
            conv4_26_add = self.shortcut(conv4_26_branch1, conv4_25, name="shortcut37") #[None ,14, 14, 1024]
            conv4_26 = tf.nn.relu(conv4_26_add) #[None ,14, 14, 1024]

            conv4_27_branch1 = self._bottleneckLayer(conv4_26, 256, 1024, training=training, name="block27") #[None ,14, 14, 1024]
            conv4_27_add = self.shortcut(conv4_27_branch1, conv4_26, name="shortcut38") #[None ,14, 14, 1024]
            conv4_27 = tf.nn.relu(conv4_27_add) #[None ,14, 14, 1024]

            conv4_28_branch1 = self._bottleneckLayer(conv4_27, 256, 1024, training=training, name="block28") #[None ,14, 14, 1024]
            conv4_28_add = self.shortcut(conv4_28_branch1, conv4_27, name="shortcut39") #[None ,14, 14, 1024]
            conv4_28 = tf.nn.relu(conv4_28_add) #[None ,14, 14, 1024]

            conv4_29_branch1 = self._bottleneckLayer(conv4_28, 256, 1024, training=training, name="block29") #[None ,14, 14, 1024]
            conv4_29_add = self.shortcut(conv4_29_branch1, conv4_28, name="shortcut40") #[None ,14, 14, 1024]
            conv4_29 = tf.nn.relu(conv4_29_add) #[None ,14, 14, 1024]

            conv4_30_branch1 = self._bottleneckLayer(conv4_29, 256, 1024, training=training, name="block30") #[None ,14, 14, 1024]
            conv4_30_add = self.shortcut(conv4_30_branch1, conv4_29, name="shortcut41") #[None ,14, 14, 1024]
            conv4_30 = tf.nn.relu(conv4_30_add) #[None ,14, 14, 1024]

            conv4_31_branch1 = self._bottleneckLayer(conv4_30, 256, 1024, training=training, name="block31") #[None ,14, 14, 1024]
            conv4_31_add = self.shortcut(conv4_31_branch1, conv4_30, name="shortcut42") #[None ,14, 14, 1024]
            conv4_31 = tf.nn.relu(conv4_31_add) #[None ,14, 14, 1024]

            conv4_32_branch1 = self._bottleneckLayer(conv4_31, 256, 1024, training=training, name="block32") #[None ,14, 14, 1024]
            conv4_32_add = self.shortcut(conv4_32_branch1, conv4_31, name="shortcut43") #[None ,14, 14, 1024]
            conv4_32 = tf.nn.relu(conv4_32_add) #[None ,14, 14, 1024]

            conv4_33_branch1 = self._bottleneckLayer(conv4_32, 256, 1024, training=training, name="block33") #[None ,14, 14, 1024]
            conv4_33_add = self.shortcut(conv4_33_branch1, conv4_32, name="shortcut44") #[None ,14, 14, 1024]
            conv4_33 = tf.nn.relu(conv4_33_add) #[None ,14, 14, 1024]

            conv4_34_branch1 = self._bottleneckLayer(conv4_33, 256, 1024, training=training, name="block34") #[None ,14, 14, 1024]
            conv4_34_add = self.shortcut(conv4_34_branch1, conv4_33, name="shortcut45") #[None ,14, 14, 1024]
            conv4_34 = tf.nn.relu(conv4_34_add) #[None ,14, 14, 1024]

            conv4_35_branch1 = self._bottleneckLayer(conv4_34, 256, 1024, training=training, name="block35") #[None ,14, 14, 1024]
            conv4_35_add = self.shortcut(conv4_35_branch1, conv4_34, name="shortcut46") #[None ,14, 14, 1024]
            conv4_35 = tf.nn.relu(conv4_35_add) #[None ,14, 14, 1024]

            conv4_36_branch1 = self._bottleneckLayer(conv4_35, 256, 1024, training=training, name="block36") #[None ,14, 14, 1024]
            conv4_36_add = self.shortcut(conv4_36_branch1, conv4_35, name="shortcut47") #[None ,14, 14, 1024]
            conv4_36 = tf.nn.relu(conv4_36_add) #[None ,14, 14, 1024]
        
        with tf.variable_scope("scale5"):
            conv5_1_branch1 = self._bottleneckLayer(conv4_36, 512, 2048, pooling=True, training=training, name="block1") #[None ,7, 7, 2048]
            conv5_1_conv_branch2 = tf.layers.conv2d(conv4_36, 2048, kernel_size=[3,3], strides=[2,2], padding="same", data_format=self.data_format, name="identify_conv") #[None ,7, 7, 2048]
            conv5_1_bn_branch2 = tf.layers.batch_normalization(conv5_1_conv_branch2, axis=1 if self.data_format == 'channels_first' else 3, training=training) #[None ,7, 7, 2048]
            conv5_1_add = self.shortcut(conv5_1_branch1, conv5_1_bn_branch2, name="shortcut48") #[None ,7, 7, 2048]
            conv5_1 = tf.nn.relu(conv5_1_add) #[None ,7, 7, 2048]

            conv5_2_branch1 = self._bottleneckLayer(conv5_1, 512, 2048, training=training, name="block2") #[None ,7, 7, 2048]
            conv5_2_add = self.shortcut(conv5_2_branch1, conv5_1, name="shortcut49") #[None ,7, 7, 2048]
            conv5_2 = tf.nn.relu(conv5_2_add) #[None ,7, 7, 2048]

            conv5_3_branch1 = self._bottleneckLayer(conv5_2, 512, 2048, training=training, name="block3") #[None ,7, 7, 2048]
            conv5_3_add = self.shortcut(conv5_3_branch1, conv5_2, name="shortcut50") #[None ,7, 7, 2048]
            conv5_3 = tf.nn.relu(conv5_3_add) #[None ,7, 7, 2048]
        
        with tf.variable_scope("fc"):
            avg_pool = tf.reduce_mean(conv5_3, axis=[2,3] if self.data_format == 'channels_first' else [1,2], keepdims=True) #[None, 1, 1, 2048]
            flatten_layer = tf.layers.flatten(avg_pool, name="flattenLayer") #[None, 2048]
            output = tf.layers.dense(flatten_layer, num_class, name="fcLayer")#[None, num_class]
        return output

if __name__ == "__main__":
    model = resnet_v1(152, 10, "channels_first")
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    output = model(inputs, training=True)
    print(output.get_shape().as_list())