import tensorflow as tf

def convLayer(input, featureNum, name, feature=[3, 3], stride=[1, 1], padding="SAME"):
    channels = int(input.get_shape()[-1])
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape = [feature[0], feature[1], channels, featureNum])
        b = tf.get_variable("b", shape = [featureNum])
        conv2d = tf.nn.conv2d(input, w, strides = [1, stride[0], stride[1], 1], padding = padding)
        out = tf.nn.bias_add(conv2d, b)
        activation_out = tf.nn.relu(out)
        return activation_out

def maxpoolLayer(input, name, feature=[2, 2], stride=[2, 2], padding="SAME"):
    return tf.nn.max_pool(input, ksize=[1, feature[0], feature[1], 1], 
                             strides=[1,stride[0],stride[1],1],
                             padding=padding,name=name)

def dropout(input, keepPro, name=None):
    return  tf.nn.dropout(input, keepPro, name)

def fcLayer(input, inputShape, outputShape, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape= [inputShape, outputShape])
        b = tf.get_variable("b", [outputShape])
        out = tf.nn.xw_plus_b(input, w, b, name=scope.name)
        return tf.nn.relu(out)

def build_vgg(input, class_num):
    assert input.get_shape().as_list()[1:] == [224, 224, 3]

    conv1_1 = convLayer(input, 64, "conv1_1")
    conv1_2 = convLayer(conv1_1, 64, "conv1_2")
    pool1 =maxpoolLayer(conv1_2, "pool1")

    conv2_1 = convLayer(pool1, 128, "conv2_1")
    conv2_2 = convLayer(conv2_1, 128, "conv2_2")
    pool2 = maxpoolLayer(conv2_2, "pool2")

    conv3_1 = convLayer(pool2, 256, "conv3_1")
    conv3_2 = convLayer(conv3_1, 256, "conv3_2")
    conv3_3 = convLayer(conv3_2, 256, "conv3_3")
    conv3_4 = convLayer(conv3_3, 256, "conv3_4")
    pool3 = maxpoolLayer(conv3_4, "pool3")

    conv4_1 = convLayer(pool3, 512, "conv4_1")
    conv4_2 = convLayer(conv4_1, 512, "conv4_2")
    conv4_3 = convLayer(conv4_2, 512, "conv4_3")
    conv4_4 = convLayer(conv4_3, 512, "conv4_4")
    pool4 = maxpoolLayer(conv4_4, "pool4")

    conv5_1 = convLayer(pool4, 512, "conv5_1")
    conv5_2 = convLayer(conv5_1, 512, "conv5_2")
    conv5_3 = convLayer(conv5_2, 512, "conv5_3")
    conv5_4 = convLayer(conv5_3, 512, "conv5_4")
    pool5 = maxpoolLayer(conv5_4, "pool5")

    fcIn = tf.reshape(pool5, [-1, 7*7*512])
    fc6 = fcLayer(fcIn, 7*7*512, 4096, "fc6")
    dropout1 = dropout(fc6, 0.7)

    fc7 = fcLayer(dropout1, 4096, 4096, "fc7")
    dropout2 = dropout(fc7, 0.7)

    fc8 = fcLayer(dropout2, 4096, class_num, "fc8")

def loadModel(model_path, sess):
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


    