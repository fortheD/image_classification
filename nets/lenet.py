import tensorflow as tf
import net

class lenet(net.net):

    def __init__(self):
        super(lenet, self).__init__()

    def build(self, input, class_num, **kwargs):
        assert input.get_shape().as_list()[1:] == [32, 32, 1]
        conv1 = self.convLayer(input, 6, name="conv_1", feature=[5, 5], padding="VALID") #[None, 28, 28, 6]
        pool1 = self.maxpoolLayer(conv1, "pool1") #[None, 14, 14, 6]
        conv2 = self.convLayer(pool1, 16, name="conv_2", feature=[5, 5], padding="VALID") #[None, 10, 10, 16]
        pool2 = self.maxpoolLayer(conv2, "pool2") #[None, 5, 5, 16]

        fcIn = tf.reshape(pool2, [-1, 5*5*16])
        fc1 = self.fcLayer(fcIn, 120, name="fc1")
        fc2 = self.fcLayer(fc1, 84, name="fc2")
        fc3 = self.fcLayer(fc2, class_num, name="fc3")
        return fc3