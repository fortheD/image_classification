from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import dataset

LEARNING_RATE = 1e-3

flags = tf.app.flags
tf.flags.DEFINE_string('data_dir', '/tmp/mnist_data', 'The data directory')
tf.flags.DEFINE_string('model_dir', '/tmp/mnist_model', 'The model directory')
tf.flags.DEFINE_string('export_dir', '', 'The export model directory')
tf.flags.DEFINE_integer('batch_size', '100', 'The training dataset batch size')
tf.flags.DEFINE_integer('train_epochs', '40', 'The training epochs')
tf.flags.DEFINE_integer('epochs_between_evals', '1', 'The number of training epochs to run between evaluation')

FLAGS = flags.FLAGS

def create_model(data_format):
    """Model to recognize digits in the MNIST dataset.

    Network structure is equivalent to:
    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
    and
    https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

    But uses the tf.keras API.

    Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
        typically faster on GPUs while 'channels_last' is typically faster on
        CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats

    Returns:
    A tf.keras.Model.
    """
    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [28, 28, 1]

    l = tf.keras.layers
    max_pool = l.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(28 * 28,)),
            l.Conv2D(
                32,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Conv2D(
                64,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dropout(0.4),
            l.Dense(10)
        ])

def model_fn(features, labels, mode, params):
    model = create_model(params['data_format'])
    image = features
    if isinstance(image, dict):
        image = features['image']
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        #save accuracy scalar to Tensorboard output
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy':tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1)),
            })
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

def clean_dir(dir):
    if tf.gfile.Exists(dir):
        tf.gfile.DeleteRecursively(dir)

def run_mnist(flags):
    clean_dir(flags.data_dir)
    clean_dir(flags.model_dir)

    session_config = tf.ConfigProto(allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(session_config=session_config)

    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=flags.model_dir,
        config=run_config,
        params={'data_format':data_format}
    )

    #Set up training input function
    def train_input_fn():
        ds = dataset.train(flags.data_dir)
        ds = ds.cache().shuffle(buffer_size=50000).batch(flags.batch_size)
        ds = ds.repeat(flags.epochs_between_evals)
        return ds
    
    #Set up evaluation input function
    def eval_input_fn():
        return dataset.test(flags.data_dir).batch(flags.batch_size).make_one_shot_iterator().get_next()
    
    for _ in range(flags.train_epochs // flags.epochs_between_evals):
        mnist_classifier.train(input_fn=train_input_fn)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        tf.logging.info('\nEvaluation results:\n\t%s\n' % eval_results)

        if eval_results['accuracy'] > 0.99:
            break
    
    #Export the model
    if flags.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            {'image': image,}
        )
        mnist_classifier.export_savedmodel(flags.export_dir, input_fn, strip_default_attrs=True)
    



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)    
    run_mnist(FLAGS)

if __name__ == '__main__':
    tf.app.run()
