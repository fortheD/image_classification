from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import vgg 

tf.app.flags.DEFINE_string(
    'output_file', 'output/net.pb', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.output_file:
        raise ValueError('You must supply the path to save to with --output_file')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:

        placeholder = tf.placeholder(name='input', dtype=tf.float32,shape=[None, 224, 224, 3])
        vggnet = vgg.vgg()
        vggnet.build(placeholder, 3, type='vgg19')

        tf.summary.FileWriter("output", graph)

        graph_def = graph.as_graph_def()
        with tf.gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())


if __name__ == '__main__':
    tf.app.run()
