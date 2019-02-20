from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import dataset

from nets.resnet import resnet

LEARNING_RATE = 1e-4

flags = tf.app.flags
tf.flags.DEFINE_string('data_dir', '/tmp/cifar10_data/cifar-10-batches-bin', 'The data directory')
tf.flags.DEFINE_string('model_dir', '/tmp/cifar10_model', 'The model directory')
tf.flags.DEFINE_string('export_dir', '', 'The export model directory')
tf.flags.DEFINE_integer('batch_size', '128', 'The training dataset batch size')
tf.flags.DEFINE_integer('train_epochs', '250', 'The training epochs')
tf.flags.DEFINE_integer('epochs_between_evals', '1', 'The number of training epochs to run between evaluation')

FLAGS = flags.FLAGS

################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
    base_lr=0.1, warmup=False):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    if warmup:
      warmup_steps = int(batches_per_epoch * 5)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
      return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
    return lr

  return learning_rate_fn

  # Learning rate schedule follows arXiv:1512.03385 for ResNet-56 and under.
learning_rate_fn = learning_rate_with_decay(
    batch_size=128, batch_denom=128,
    num_images=50000, boundary_epochs=[91, 136, 182],
    decay_rates=[1, 0.1, 0.01, 0.001])

def model_fn(features, labels, mode, params):
    weight_decay = 2e-4

    model = resnet(56, 10, params['data_format'], resnet_version=2)  
    
    image = features
    if isinstance(image, dict):
        image = features['image']
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = learning_rate_fn(tf.train.get_or_create_global_step())
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        tf.summary.scalar('learning_rate', learning_rate)

        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # Add weight decay to the loss.
        l2_loss = weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
        loss = loss + l2_loss
        accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

        tf.identity(learning_rate, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        #save accuracy scalar to Tensorboard output
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Add weight decay to the loss.
        l2_loss = weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
        loss = loss + l2_loss

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

def run_cifar(flags):
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
        image = tf.placeholder(tf.float32, [None, 32, 32, 3])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            {'image': image,}
        )
        mnist_classifier.export_savedmodel(flags.export_dir, input_fn, strip_default_attrs=True)
    



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)    
    run_cifar(FLAGS)

if __name__ == '__main__':
    tf.app.run()
