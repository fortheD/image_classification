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
        conv2_2_add =self.shortcut(conv2_3_branch1, conv2_2, name="shortcut3")
        conv2_3 = tf.nn.leaky_relu(conv2_2_add)

        print(conv2_1.get_shape().as_list())
    
    def build_resnet101(self, input, class_num):
        pass
    
    def build_resnet152(self, input, class_num):
        pass