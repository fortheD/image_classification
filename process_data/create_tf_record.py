import tensorflow as tf
import os
import PIL.Image
import io

from utils import dataset_util

flags = tf.app.flags
tf.flags.DEFINE_string('image_dir', '', 'The image directory')
tf.flags.DEFINE_string('output_dir', 'output', 'The output tf_record directory')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def _create_tf_record(image_dir, output_record):
    image_files = os.listdir(image_dir)
    writer = tf.python_io.TFRecordWriter(output_record)
    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        feature_dict = {
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),}
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))                
        writer.write(example.SerializeToString())
    tf.logging.info("translate complete")

def main(_):
    assert FLAGS.image_dir, '`image_dir ` missing'
    assert FLAGS.output_dir, '`output_dir` missing'
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    output_record = os.path.join(FLAGS.output_dir, 'train.record')
    _create_tf_record(FLAGS.image_dir, output_record)

if __name__ == '__main__':
    tf.app.run()