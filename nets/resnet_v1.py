import tensorflow as tf
import net

class resnet_v1(net.net):
    
    def __init__(self):
        super(resnet_v1, self).__init__()
    
    def build(self, input, class_num, **kwargs):
        if kwargs.has_key('type'):
            if kwargs['type'] == 'resnet50':
                self.build_resnet50(input, class_num)
        else:
            self.build_resnet50(input, class_num)
    
    def build_resnet50(self, input, class_num):
        assert input.get_shape().as_list()[1:] == [224, 224, 3]

        conv1_1 = self.conv_bn_leakyLayer(input, 64, name="conv1_1", feature=[7, 7], stride=[2, 2]) #[None, 112, 112, 3]
        pool1 = self.maxpoolLayer(conv1_1, name="pool1", feature=[3,3]) #[None, 56, 56, 3]

        conv2_1_branch1 = self.bottleneckLayer(pool1, 64, 256, name="conv2_1")
        conv2_1_conv_branch2 = tf.layers.conv2d(pool1, 256, [1,1], use_bias=False)
        conv2_1_bn_branch2 = tf.layers.batch_normalization(conv2_1_conv_branch2)
        conv2_1_add = self.shortcut(conv2_1_branch1, conv2_1_bn_branch2, name="shortcut1")
        conv2_1 = tf.nn.leaky_relu(conv2_1_add)

        conv2_2_branch1 = self.bottleneckLayer(conv2_1, 64, 256, name="conv2_2")
        conv2_2_add = self.shortcut(conv2_2_branch1, conv2_1, name="shortcut2")
        conv2_2 = tf.nn.leaky_relu(conv2_2_add)

        conv2_3_branch1 = self.bottleneckLayer(conv2_2, 64, 256, name="conv2_3")
        conv2_3_add =self.shortcut(conv2_3_branch1, conv2_2, name="shortcut3")
        conv2_3 = tf.nn.leaky_relu(conv2_3_add)

        conv3_1_branch1 = self.bottleneckLayer(conv2_3, 128, 512, name="conv3_1", pooling=True)
        conv3_1_conv_branch2 = tf.layers.conv2d(conv2_3, 512, [1,1], strides=[2,2], use_bias=False)
        conv3_1_bn_branch2 = tf.layers.batch_normalization(conv3_1_conv_branch2)
        conv3_1_add = self.shortcut(conv3_1_branch1, conv3_1_bn_branch2, name="shortcut4")
        conv3_1 = tf.nn.leaky_relu(conv3_1_add)

        conv3_2_branch1 = self.bottleneckLayer(conv3_1, 128, 512, name="conv3_2")
        conv3_2_add = self.shortcut(conv3_2_branch1, conv3_1, name="shortcut5")
        conv3_2 = tf.nn.leaky_relu(conv3_2_add)

        conv3_3_branch1 = self.bottleneckLayer(conv3_2, 128, 512, name="conv3_3")
        conv3_3_add =self.shortcut(conv3_3_branch1, conv3_2, name="shortcut6")
        conv3_3 = tf.nn.leaky_relu(conv3_3_add)

        conv3_4_branch1 = self.bottleneckLayer(conv3_3, 128, 512, name="conv3_4")
        conv3_4_add =self.shortcut(conv3_4_branch1, conv3_3, name="shortcut7")
        conv3_4 = tf.nn.leaky_relu(conv3_4_add)

        conv4_1_branch1 = self.bottleneckLayer(conv3_4, 256, 1024, name="conv4_1", pooling=True)
        conv4_1_conv_branch2 = tf.layers.conv2d(conv3_4, 1024, [1,1], strides=[2,2], use_bias=False)
        conv4_1_bn_branch2 = tf.layers.batch_normalization(conv4_1_conv_branch2)
        conv4_1_add = self.shortcut(conv4_1_branch1, conv4_1_bn_branch2, name="shortcut8")
        conv4_1 = tf.nn.leaky_relu(conv4_1_add)

        conv4_2_branch1 = self.bottleneckLayer(conv4_1, 256, 1024, name="conv4_2")
        conv4_2_add = self.shortcut(conv4_2_branch1, conv4_1, name="shortcut9")
        conv4_2 = tf.nn.leaky_relu(conv4_2_add)

        conv4_3_branch1 = self.bottleneckLayer(conv4_2, 256, 1024, name="conv4_3")
        conv4_3_add = self.shortcut(conv4_3_branch1, conv4_2, name="shortcut10")
        conv4_3 = tf.nn.leaky_relu(conv4_3_add)

        conv4_4_branch1 = self.bottleneckLayer(conv4_3, 256, 1024, name="conv4_4")
        conv4_4_add = self.shortcut(conv4_4_branch1, conv4_3, name="shortcut11")
        conv4_4 = tf.nn.leaky_relu(conv4_4_add)

        conv4_5_branch1 = self.bottleneckLayer(conv4_4, 256, 1024, name="conv4_5")
        conv4_5_add = self.shortcut(conv4_5_branch1, conv4_4, name="shortcut12")
        conv4_5 = tf.nn.leaky_relu(conv4_5_add)

        conv4_6_branch1 = self.bottleneckLayer(conv4_5, 256, 1024, name="conv4_6")
        conv4_6_add = self.shortcut(conv4_6_branch1, conv4_5, name="shortcut13")
        conv4_6 = tf.nn.leaky_relu(conv4_6_add)

        conv5_1_branch1 = self.bottleneckLayer(conv4_6, 512, 2048, name="conv5_1", pooling=True)
        conv5_1_conv_branch2 = tf.layers.conv2d(conv4_6, 2048, [1,1], strides=[2,2], use_bias=False)
        conv5_1_bn_branch2 = tf.layers.batch_normalization(conv5_1_conv_branch2)
        conv5_1_add = self.shortcut(conv5_1_branch1, conv5_1_bn_branch2, name="shortcut14")
        conv5_1 = tf.nn.leaky_relu(conv5_1_add)

        conv5_2_branch1 = self.bottleneckLayer(conv5_1, 512, 2048, name="conv5_2")
        conv5_2_add = self.shortcut(conv5_2_branch1, conv5_1, name="shortcut15")
        conv5_2 = tf.nn.leaky_relu(conv5_2_add)

        conv5_3_branch1 = self.bottleneckLayer(conv5_2, 512, 2048, name="conv5_3")
        conv5_3_add = self.shortcut(conv5_3_branch1, conv5_2, name="shortcut16")
        conv5_3 = tf.nn.leaky_relu(conv5_3_add)

        avg_pool = tf.nn.avg_pool(conv5_3, [1, 7, 7, 1], [1, 1, 1, 1], padding="VALID")
        flatten_layer = tf.layers.flatten(avg_pool)
        output = tf.layers.dense(flatten_layer, class_num)

        print(output.get_shape().as_list())
    
    def build_resnet101(self, input, class_num):
        pass
    
    def build_resnet152(self, input, class_num):
        pass