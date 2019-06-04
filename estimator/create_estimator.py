from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.model import ClassifyModel

def learning_rate_with_decay(num_images, batch_size=16, batch_denom=16, boundary_epochs=[50, 100, 150],
                            decay_rates=[1, 0.1, 0.01, 0.001], base_lr=0.1, warmup=False):
    """Get a learning rate that decays step-wise as training progresses
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
        lr = tf.train.piecewise_constant_decay(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        return lr
    return learning_rate_fn

class Classifier():
    def __init__(self, model_name, classes, data_format):
        self.model = ClassifyModel(model_name, classes, data_format)

    def model_fn(self, features, labels, mode, params):
        """
        features: The input image tensor
        labels: Training datasets labels
        model: TRAIN, EVAL, PREDICT
        params: A dict {'data_format':data_format}
        """
        weight_decay = 2e-4

        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate_fn = learning_rate_with_decay(num_images=params['image_nums'])
            learning_rate = learning_rate_fn(tf.train.get_or_create_global_step())
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            tf.summary.scalar('learning_rate', learning_rate)

            logits = self.model(features, training=True)
            train_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            # Add weight decay to the loss.
            # l2_loss = weight_decay * tf.add_n(
            #     # loss is computed using fp32 for numerical stability.
            #     [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
            # train_loss = train_loss + l2_loss
            accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

            tf.identity(train_loss, "train_loss")
            tf.identity(accuracy[1], name='train_accuracy')

            # Save accuracy scalar to Tensorboard output
            tf.summary.scalar('train_loss', train_loss)
            tf.summary.scalar('train_accuracy', accuracy[1])

            train_op = optimizer.minimize(train_loss, tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=train_loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            logits = self.model(features, training=True)
            eval_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            # Save eval loss to Tensorboard output
            tf.summary.scalar('eval_loss', eval_loss)

            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=eval_loss,
                eval_metric_ops={
                'accuracy':tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1)),
                })

        elif mode == tf.estimator.ModeKeys.PREDICT:
            logits = self.model(features, training=False)
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