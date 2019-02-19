from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import lenet
from nets import vgg
from nets import resnet_v1

networks_map = {'lenet': lenet.lenet,
                'vgg': vgg.vgg,
                'resnet_v1': resnet_v1.resnet_v1,
               }

def get_network_fn(name, size, num_classes, data_format="channels_first", is_training=False):
    if name not in networks_map:
        raise ValueError('Name if network unknown %s' % name)
    model = networks_map[name](size, num_classes, data_format)
    def network_fn(inputs):
        return model(inputs, is_training)
    return network_fn