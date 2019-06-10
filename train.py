from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf

from utils import dataset_util
from process_data.create_tf_record import record_exists, get_filenames_and_classes
from process_data.create_tf_record import _RANDOM_SEED, _NUM_SHARDS, _RATE_VAL_ALL
from process_data.create_tf_record import convert_record
from process_data.read_tf_record import train, val

from estimator.create_estimator import Classifier

flags = tf.app.flags

tf.flags.DEFINE_string('image_dir', '', 'The image directory')
tf.flags.DEFINE_string('tfrecord_dir', '/tmp/record', 'Temporary directory of record file')
tf.flags.DEFINE_string('model_dir', '/tmp/train', 'Saved model directory')
tf.flags.DEFINE_string('export_dir', '', 'The export model directory')
tf.flags.DEFINE_integer('batch_size', '16', 'The training dataset batch size')
tf.flags.DEFINE_integer('train_epochs', '200', 'The training epochs')

FLAGS = flags.FLAGS

def run(flags):
    """
    Create tensorflow record file from image directory
    """
    assert flags.image_dir, '`image_dir ` missing'
    assert flags.tfrecord_dir, '`tfrecord_dir` missing'
    assert flags.model_dir, '`model_dir` missing'
    if not tf.gfile.IsDirectory(flags.tfrecord_dir):
        tf.gfile.MakeDirs(flags.tfrecord_dir)

    # Define nerve network input shape
    input_shape = (299, 299, 3)

    image_dir = flags.image_dir
    record_dir = flags.tfrecord_dir

    photo_filenames, class_names = get_filenames_and_classes(image_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test record
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)

    photo_nums = len(photo_filenames)
    validation_nums = int(photo_nums * _RATE_VAL_ALL)

    training_filenames = photo_filenames[validation_nums:]
    validation_filenames = photo_filenames[:validation_nums]

    if record_exists(record_dir):
        tf.logging.info('Record files already exist')
    else:
        # Convert the training and validation record
        convert_record('train', training_filenames, class_names_to_ids, record_dir)
        convert_record('validation', validation_filenames, class_names_to_ids, record_dir)

        # Finally, write the label file
        label_to_class_names = dict(zip(range(len(class_names)), class_names))
        dataset_util.write_label_file(label_to_class_names, image_dir)

    tf.logging.info("Translate complete")

    """"
    Begin to train a classifier
    """
    session_config = tf.ConfigProto(allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(session_config=session_config)

    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    classifier = Classifier("InceptionV3", len(class_names), data_format)
    estimator = tf.estimator.Estimator(
        model_fn=classifier.model_fn,
        model_dir=flags.model_dir,
        config=run_config,
        params={'image_nums':len(training_filenames)}
    )

    #Set up training input function
    def train_input_fn():
        ds = train(record_dir, input_shape)
        ds = ds.cache().shuffle(buffer_size=20000).batch(flags.batch_size)
        ds = ds.repeat(1)
        return ds
    
    #Set up evaluation input function
    def eval_input_fn():
        return val(record_dir, input_shape).batch(flags.batch_size)

    for _ in range(flags.train_epochs):
        estimator.train(input_fn=train_input_fn)
        eval_results = estimator.evaluate(input_fn=eval_input_fn)
        tf.logging.info('\nEvaluation results:\n\t%s\n' % eval_results)

    #Export the model as saved_model format
    if flags.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            {'image': image,}
        )
        estimator.export_savedmodel(flags.export_dir, input_fn, strip_default_attrs=True)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    run(FLAGS)

if __name__ == '__main__':
    tf.app.run()