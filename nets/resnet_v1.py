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
            conv3_1_conv_branch2 = tf.layers.conv2d(conv2_3, 512, kernel_size=[3,3], strides=[2,2], padding="same", name="identify_conv") #[None ,28, 28, 512]
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
            conv4_1_conv_branch2 = tf.layers.conv2d(conv3_4, 1024, kernel_size=[3,3], strides=[2,2], padding="same", name="identify_conv") #[None ,14, 14, 1024]
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
            conv5_1_conv_branch2 = tf.layers.conv2d(conv4_6, 2048, kernel_size=[3,3], strides=[2,2], padding="same", name="identify_conv") #[None ,7, 7, 2048]
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
            avg_pool = tf.reduce_mean(conv5_3, axis=[2,3] if self.data_format == 'channels_first' else [1,2], keepdims=True)
            flatten_layer = tf.layers.flatten(avg_pool, name="flattenLayer") #[None, 2048]
            output = tf.layers.dense(flatten_layer, num_class, name="fcLayer")
        return output
    
    def build_resnet101(self, input, training, num_class):
        pass
    
    def build_resnet152(self, input, training, num_class):
        pass

if __name__ == "__main__":
    model = resnet_v1(50, 10, "channels_last")
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    output = model(inputs, training=True)
    print(output.get_shape().as_list())