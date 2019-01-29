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
    
    def build_resnet50(self, input, class_num, training=False):
        assert input.get_shape().as_list()[1:] == [224, 224, 3]

        with tf.variable_scope("scale1"):
            conv1_1 = self.conv_bn_leakyLayer(input, 64, name="block1", feature=[7, 7], stride=[2, 2]) #[None, 112, 112, 3]
            pool1 = self.maxpoolLayer(conv1_1, name="pool1", feature=[3,3]) #[None, 56, 56, 3]
        
        with tf.variable_scope("scale2"):
            conv2_1_branch1 = self.bottleneckLayer(pool1, 64, 256, name="block1") #[None ,56, 56, 256]
            conv2_1_conv_branch2 = self.convLayer(pool1, 256, name="identify_conv", feature=[1,1], active=False) #[None ,56, 56, 256]
            conv2_1_bn_branch2 = self.bnLayer(conv2_1_conv_branch2, training, name="identify_bn") #[None ,56, 56, 256]
            conv2_1_add = self.shortcut(conv2_1_branch1, conv2_1_bn_branch2, name="shortcut1") #[None ,56, 56, 256]
            conv2_1 = self.activeLayer(conv2_1_add) #[None ,56, 56, 256]

            conv2_2_branch1 = self.bottleneckLayer(conv2_1, 64, 256, name="block2") #[None ,56, 56, 256]
            conv2_2_add = self.shortcut(conv2_2_branch1, conv2_1, name="shortcut2") #[None ,56, 56, 256]
            conv2_2 = self.activeLayer(conv2_2_add) #[None ,56, 56, 256]

            conv2_3_branch1 = self.bottleneckLayer(conv2_2, 64, 256, name="block3") #[None ,56, 56, 256]
            conv2_3_add =self.shortcut(conv2_3_branch1, conv2_2, name="shortcut3") #[None ,56, 56, 256]
            conv2_3 = self.activeLayer(conv2_3_add) #[None ,56, 56, 256]
        
        with tf.variable_scope("scale3"):
            conv3_1_branch1 = self.bottleneckLayer(conv2_3, 128, 512, name="block1", pooling=True) #[None ,28, 28, 512]
            conv3_1_conv_branch2 = self.convLayer(conv2_3, 512,  name="identify_conv", feature=[1,1], strides=[2,2], active=False) #[None ,28, 28, 512]
            conv3_1_bn_branch2 = self.bnLayer(conv3_1_conv_branch2, training, name="identify_bn") #[None ,28, 28, 512]
            conv3_1_add = self.shortcut(conv3_1_branch1, conv3_1_bn_branch2, name="shortcut4") #[None ,28, 28, 512]
            conv3_1 = self.activeLayer(conv3_1_add) #[None ,28, 28, 512]

            conv3_2_branch1 = self.bottleneckLayer(conv3_1, 128, 512, name="block2") #[None ,28, 28, 512]
            conv3_2_add = self.shortcut(conv3_2_branch1, conv3_1, name="shortcut5") #[None ,28, 28, 512]
            conv3_2 = self.activeLayer(conv3_2_add) #[None ,28, 28, 512]

            conv3_3_branch1 = self.bottleneckLayer(conv3_2, 128, 512, name="block3") #[None ,28, 28, 512]
            conv3_3_add =self.shortcut(conv3_3_branch1, conv3_2, name="shortcut6") #[None ,28, 28, 512]
            conv3_3 = self.activeLayer(conv3_3_add) #[None ,28, 28, 512]

            conv3_4_branch1 = self.bottleneckLayer(conv3_3, 128, 512, name="block4") #[None ,28, 28, 512]
            conv3_4_add =self.shortcut(conv3_4_branch1, conv3_3, name="shortcut7") #[None ,28, 28, 512]
            conv3_4 = self.activeLayer(conv3_4_add) #[None ,28, 28, 512]

        with tf.variable_scope("scale4"):
            conv4_1_branch1 = self.bottleneckLayer(conv3_4, 256, 1024, name="block1", pooling=True) #[None ,14, 14, 1024]
            conv4_1_conv_branch2 = self.convLayer(conv3_4, 1024, name="identify_conv", feature=[1,1], strides=[2,2], active=False) #[None ,14, 14, 1024]
            conv4_1_bn_branch2 = self.bnLayer(conv4_1_conv_branch2, training, name="identify_bn") #[None ,14, 14, 1024]
            conv4_1_add = self.shortcut(conv4_1_branch1, conv4_1_bn_branch2, name="shortcut8") #[None ,14, 14, 1024]
            conv4_1 = self.activeLayer(conv4_1_add) #[None ,14, 14, 1024]

            conv4_2_branch1 = self.bottleneckLayer(conv4_1, 256, 1024, name="block2") #[None ,14, 14, 1024]
            conv4_2_add = self.shortcut(conv4_2_branch1, conv4_1, name="shortcut9") #[None ,14, 14, 1024]
            conv4_2 = self.activeLayer(conv4_2_add) #[None ,14, 14, 1024]

            conv4_3_branch1 = self.bottleneckLayer(conv4_2, 256, 1024, name="block3") #[None ,14, 14, 1024]
            conv4_3_add = self.shortcut(conv4_3_branch1, conv4_2, name="shortcut10") #[None ,14, 14, 1024]
            conv4_3 = self.activeLayer(conv4_3_add) #[None ,14, 14, 1024]

            conv4_4_branch1 = self.bottleneckLayer(conv4_3, 256, 1024, name="block4") #[None ,14, 14, 1024]
            conv4_4_add = self.shortcut(conv4_4_branch1, conv4_3, name="shortcut11") #[None ,14, 14, 1024]
            conv4_4 = self.activeLayer(conv4_4_add) #[None ,14, 14, 1024]

            conv4_5_branch1 = self.bottleneckLayer(conv4_4, 256, 1024, name="block5") #[None ,14, 14, 1024]
            conv4_5_add = self.shortcut(conv4_5_branch1, conv4_4, name="shortcut12") #[None ,14, 14, 1024]
            conv4_5 = self.activeLayer(conv4_5_add) #[None ,14, 14, 1024]

            conv4_6_branch1 = self.bottleneckLayer(conv4_5, 256, 1024, name="block6") #[None ,14, 14, 1024]
            conv4_6_add = self.shortcut(conv4_6_branch1, conv4_5, name="shortcut13") #[None ,14, 14, 1024]
            conv4_6 = self.activeLayer(conv4_6_add) #[None ,14, 14, 1024]
        
        with tf.variable_scope("scale5"):
            conv5_1_branch1 = self.bottleneckLayer(conv4_6, 512, 2048, name="block1", pooling=True) #[None ,7, 7, 2048]
            conv5_1_conv_branch2 = self.convLayer(conv4_6, 2048, name="identify_conv", feature=[1,1], strides=[2,2], active=False) #[None ,7, 7, 2048]
            conv5_1_bn_branch2 = self.bnLayer(conv5_1_conv_branch2, training, name="identify_bn") #[None ,7, 7, 2048]
            conv5_1_add = self.shortcut(conv5_1_branch1, conv5_1_bn_branch2, name="shortcut14") #[None ,7, 7, 2048]
            conv5_1 = self.activeLayer(conv5_1_add) #[None ,7, 7, 2048]

            conv5_2_branch1 = self.bottleneckLayer(conv5_1, 512, 2048, name="block2") #[None ,7, 7, 2048]
            conv5_2_add = self.shortcut(conv5_2_branch1, conv5_1, name="shortcut15") #[None ,7, 7, 2048]
            conv5_2 = self.activeLayer(conv5_2_add) #[None ,7, 7, 2048]

            conv5_3_branch1 = self.bottleneckLayer(conv5_2, 512, 2048, name="block3") #[None ,7, 7, 2048]
            conv5_3_add = self.shortcut(conv5_3_branch1, conv5_2, name="shortcut16") #[None ,7, 7, 2048]
            conv5_3 = self.activeLayer(conv5_3_add) #[None ,7, 7, 2048]
        
        with tf.variable_scope("fc"):
            avg_pool = self.avgpoolLayer(conv5_3, [7, 7], [1, 1], padding="SAME", name="avgpoolLayer") #[None ,1, 1, 2048]
            flatten_layer = self.flattenLayer(avg_pool, name="flattenLayer") #[None, 2048]
            output = self.fcLayer(flatten_layer, class_num, name="fcLayer") #[None, class_num]

        print(output.get_shape().as_list())
    
    def build_resnet101(self, input, class_num):
        pass
    
    def build_resnet152(self, input, class_num):
        pass