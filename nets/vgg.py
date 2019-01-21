import tensorflow as tf
import net

class vgg(net.net):
    def __init__(self):
        super(vgg, self).__init__()

    def build(self, input, class_num, **kwargs):
        if kwargs.has_key('type'):
            if kwargs['type'] == 'vgg16':
                self.build_vgg16(input, class_num)
            elif kwargs['type'] == 'vgg19':
                self.build_vgg19(input, class_num)
            else:
                print('please confirm the type name(vgg16 or vgg19)')
        else:
            self.build_vgg16(input, class_num)

    def build_vgg16(self, input, class_num):
        conv1_1 = self.convLayer(input, 64, "conv1_1")
        conv1_2 = self.convLayer(conv1_1, 64, "conv1_2")
        pool1 = self.maxpoolLayer(conv1_2, "pool1")

        conv2_1 = self.convLayer(pool1, 128, "conv2_1")
        conv2_2 = self.convLayer(conv2_1, 128, "conv2_2")
        pool2 = self.maxpoolLayer(conv2_2, "pool2")

        conv3_1 = self.convLayer(pool2, 256, "conv3_1")
        conv3_2 = self.convLayer(conv3_1, 256, "conv3_2")
        conv3_3 = self.convLayer(conv3_2, 256, "conv3_3")
        pool3 = self.maxpoolLayer(conv3_3, "pool3")

        conv4_1 = self.convLayer(pool3, 512, "conv4_1")
        conv4_2 = self.convLayer(conv4_1, 512, "conv4_2")
        conv4_3 = self.convLayer(conv4_2, 512, "conv4_3")
        pool4 = self.maxpoolLayer(conv4_3, "pool4")

        conv5_1 = self.convLayer(pool4, 512, "conv5_1")
        conv5_2 = self.convLayer(conv5_1, 512, "conv5_2")
        conv5_3 = self.convLayer(conv5_2, 512, "conv5_3")
        pool5 = self.maxpoolLayer(conv5_3, "pool5")

        fcIn = tf.reshape(pool5, [-1, 7*7*512])
        fc6 = self.fcLayer(fcIn, 7*7*512, 4096, "fc6")
        dropout1 = self.dropout(fc6, 0.7)

        fc7 = self.fcLayer(dropout1, 4096, 4096, "fc7")
        dropout2 = self.dropout(fc7, 0.7)

        fc8 = self.fcLayer(dropout2, 4096, class_num, "fc8")
        return fc8        

    def build_vgg19(self, input, class_num):
        assert input.get_shape().as_list()[1:] == [224, 224, 3]

        conv1_1 = self.convLayer(input, 64, "conv1_1")
        conv1_2 = self.convLayer(conv1_1, 64, "conv1_2")
        pool1 = self.maxpoolLayer(conv1_2, "pool1")

        conv2_1 = self.convLayer(pool1, 128, "conv2_1")
        conv2_2 = self.convLayer(conv2_1, 128, "conv2_2")
        pool2 = self.maxpoolLayer(conv2_2, "pool2")

        conv3_1 = self.convLayer(pool2, 256, "conv3_1")
        conv3_2 = self.convLayer(conv3_1, 256, "conv3_2")
        conv3_3 = self.convLayer(conv3_2, 256, "conv3_3")
        conv3_4 = self.convLayer(conv3_3, 256, "conv3_4")
        pool3 = self.maxpoolLayer(conv3_4, "pool3")

        conv4_1 = self.convLayer(pool3, 512, "conv4_1")
        conv4_2 = self.convLayer(conv4_1, 512, "conv4_2")
        conv4_3 = self.convLayer(conv4_2, 512, "conv4_3")
        conv4_4 = self.convLayer(conv4_3, 512, "conv4_4")
        pool4 = self.maxpoolLayer(conv4_4, "pool4")

        conv5_1 = self.convLayer(pool4, 512, "conv5_1")
        conv5_2 = self.convLayer(conv5_1, 512, "conv5_2")
        conv5_3 = self.convLayer(conv5_2, 512, "conv5_3")
        conv5_4 = self.convLayer(conv5_3, 512, "conv5_4")
        pool5 = self.maxpoolLayer(conv5_4, "pool5")

        fcIn = tf.reshape(pool5, [-1, 7*7*512])
        fc6 = self.fcLayer(fcIn, 7*7*512, 4096, "fc6")
        dropout1 = self.dropout(fc6, 0.7)

        fc7 = self.fcLayer(dropout1, 4096, 4096, "fc7")
        dropout2 = self.dropout(fc7, 0.7)

        fc8 = self.fcLayer(dropout2, 4096, class_num, "fc8")
        return fc8


    