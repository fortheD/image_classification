from abc import ABCMeta, abstractmethod

import tensorflow as tf

class net:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def convLayer(self, input, feature_num, name, feature=[3, 3], stride=[1, 1], padding="SAME"):
        channels = int(input.get_shape()[-1])
        with tf.variable_scope(name):
            w = tf.get_variable("w", shape = [feature[0], feature[1], channels, feature_num])
            b = tf.get_variable("b", shape = [feature_num])
            conv2d = tf.nn.conv2d(input, w, strides = [1, stride[0], stride[1], 1], padding = padding)
            out = tf.nn.bias_add(conv2d, b)
            activation_out = tf.nn.relu(out)
            return activation_out

    def maxpoolLayer(self, input, name, feature=[2, 2], stride=[2, 2], padding="SAME"):
        return tf.nn.max_pool(input, ksize=[1, feature[0], feature[1], 1], 
                                strides=[1,stride[0],stride[1],1],
                                padding=padding,name=name)

    def dropout(self, input, keep_pro=0.7, name=None):
        return  tf.nn.dropout(input, keep_pro, name)

    def fcLayer(self, input, inputShape, outputShape, name):
        with tf.variable_scope(name) as scope:
            w = tf.get_variable("w", shape= [inputShape, outputShape])
            b = tf.get_variable("b", [outputShape])
            out = tf.nn.xw_plus_b(input, w, b, name=scope.name)
            return tf.nn.relu(out)

    def conv_bn_leakyLayer(self, input, feature_num, name, feature=[3,3], stride=[1,1], padding="SAME", training=False):
        channels = int(input.get_shape()[-1])
        with tf.variable_scope(name):
            w = tf.get_variable("w", shape = [feature[0], feature[1], channels, feature_num])
            conv2d = tf.nn.conv2d(input, w, strides = [1, stride[0], stride[1], 1], padding = padding)
            bn_layer = tf.layers.batch_normalization(conv2d, training=training)
            active_layer = tf.nn.leaky_relu(bn_layer)
            return active_layer
    
    def bottleneckLayer(self, input, input_layers, output_layers, name):
        layer1 = self.conv_bn_leakyLayer(input, input_layers, name+"_1", feature=[1,1])
        layer2 = self.conv_bn_leakyLayer(layer1, input_layers, name+"_2", feature=[3,3])
        layer3 = self.conv_bn_leakyLayer(layer2, output_layers, name+"_3", feature=[1,1])
        return layer3
    
    def zeropaddingLayer(self, input, add_rows, add_cols):
        padding = tf.constant([[0,0], [0, add_rows], [0, add_cols], [0, 0]])
        padding_layer = tf.pad(input, padding)
        return padding_layer
    
    def shortcut(self, input1, input2, name):
        with tf.name_scope(name):
            output = input1 + input2
            return output
    
    @abstractmethod
    def build(self, input, class_num, **kwargs):
        '''
        Please implement in subclass
        '''

    def loadModelFromNPY(self, model_path, sess):
        """load model"""
        wDict = np.load(model_path, encoding = "bytes").item()
        #for layers in model
        for name in wDict:
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            #bias
                            sess.run(tf.get_variable('b', trainable = False).assign(p))
                        else:
                            #weights
                            sess.run(tf.get_variable('w', trainable = False).assign(p))