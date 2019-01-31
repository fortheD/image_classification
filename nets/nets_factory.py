from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import lenet
from nets import vgg
from nets import resnet_v1

networks_map = {'lenet': lenet.lenet,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'resnet_v1_50': resnet_v1.resnet_v1_50,
                'resnet_v1_101': resnet_v1.resnet_v1_101,
                'resnet_v1_152': resnet_v1.resnet_v1_152,
               }

def get_network_fn(name, num_classes, is_training=False):
    if name not in networks_map:
        raise ValueError('Name if network unknown %s' % name)
    func = networks_map[name]