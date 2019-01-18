import tensorflow as tf

def build_vgg(input, class_num):
    assert input.get_shape().as_list()[1:] == [224, 224, 1]
    